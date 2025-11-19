import os
import json
from pathlib import Path
import requests
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from vector_db import ChromaDBStore, FAISSStore
from extractors_simple import extract_text_from_multiple_files, split_chunk_overlap
from document_downloader import DocumentDownloader

load_dotenv()


class CrawlerRAGPipeline:
    """Integrated RAG pipeline for crawler data + downloaded documents"""
    
    def __init__(
        self,
        embedding_model_name="toshk0/nomic-embed-text-v2-moe:Q6_K",
        vector_db_type="FAISS",
        vector_db_path="indexes/crawler_vectordb",
        chunk_size=1000,
        chunk_overlap=200,
        ollama_base_url="http://localhost:11434",
        max_workers=30  # Number of concurrent embedding requests
    ):
        self.embedding_model_name = embedding_model_name
        self.vector_db_type = vector_db_type
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_base_url = ollama_base_url
        self.max_workers = max_workers
        self.progress_lock = Lock()  # For thread-safe progress updates
        
        # Initialize embedding model
        print(f"🔄 Using Ollama embedding model: {embedding_model_name}")
        self._test_ollama_connection()
        print(f"✅ Ollama model ready with {max_workers} concurrent workers")
        
        # Initialize vector database
        self.vector_db = self._load_or_create_vector_db()
        
        # Document downloader
        self.downloader = DocumentDownloader()
    
    def _test_ollama_connection(self):
        """Test connection to Ollama and model availability"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model_name,
                    "prompt": "test"
                },
                timeout=10
            )
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {self.ollama_base_url}")
            else:
                print(f"⚠️  Warning: Ollama returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"   Make sure Ollama is running and model '{self.embedding_model_name}' is pulled")
            raise
    
    def _get_single_embedding(self, text, index):
        """Get embedding for a single text (helper for concurrent processing)"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model_name,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                embedding = response.json()["embedding"]
                return (index, embedding, None)
            else:
                error_msg = f"Status {response.status_code}"
                return (index, None, error_msg)
        except Exception as e:
            return (index, None, str(e))
    
    def _get_ollama_embeddings(self, texts):
        """Get embeddings from Ollama for a list of texts using concurrent requests"""
        embeddings = [None] * len(texts)
        errors = []
        completed = 0
        
        print(f"🧮 Generating embeddings for {len(texts)} texts with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._get_single_embedding, text, i): i 
                for i, text in enumerate(texts)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                index, embedding, error = future.result()
                
                if embedding is not None:
                    embeddings[index] = embedding
                else:
                    errors.append((index, error))
                    # Use zero vector as fallback
                    # Determine embedding dimension from first successful embedding
                    valid_embeddings = [e for e in embeddings if e is not None]
                    dim = len(valid_embeddings[0]) if valid_embeddings else 768
                    embeddings[index] = [0.0] * dim
                
                completed += 1
                
                # Thread-safe progress update
                with self.progress_lock:
                    if completed % 10 == 0 or completed == len(texts):
                        print(f"   Progress: {completed}/{len(texts)} embeddings generated")
        
        if errors:
            print(f"⚠️  {len(errors)} embeddings failed. Using zero vectors as fallback.")
            for idx, error in errors[:3]:  # Show first 3 errors
                print(f"   - Text {idx}: {error}")
        
        return np.array(embeddings)
    
    def _load_or_create_vector_db(self):
        """Load existing vector DB or create new one"""
        if self.vector_db_type == "FAISS":
            vector_db = FAISSStore()
            
            # Try to load existing index - check for .index file
            index_file = Path(str(self.vector_db_path) + ".index")
            if index_file.exists():
                print(f"🔄 Loading existing FAISS index from: {self.vector_db_path}")
                if vector_db.load(str(self.vector_db_path)):
                    print(f"✅ Loaded FAISS index with {vector_db.count()} vectors")
                    return vector_db
                else:
                    print("⚠️  Failed to load index, creating new one")
            else:
                print(f"📝 Creating new FAISS index (no index found at {index_file})")
            
            return vector_db
        else:
            print("📝 Creating ChromaDB collection")
            return ChromaDBStore()
    
    def save_vector_db(self):
        """Save vector database to disk"""
        if self.vector_db_type == "FAISS":
            self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_db.save(str(self.vector_db_path))
            print(f"💾 Saved FAISS index to: {self.vector_db_path}")
        else:
            print("ℹ️  ChromaDB persists automatically")
    
    def load_crawler_data(self, json_path):
        """Load web content from crawler JSON"""
        print(f"\n📖 Loading crawler data from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            if item.get('success') and item.get('content'):
                doc = Document(
                    page_content=item['content'],
                    metadata={
                        'source': item['metadata']['url'],
                        'title': item['metadata']['title'],
                        'source_type': 'web_crawl',
                        'depth': item['metadata']['depth'],
                        'crawl_timestamp': item['metadata']['crawl_timestamp'],
                        'content_length': item['metadata']['content_length']
                    }
                )
                documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} web documents from crawler")
        return documents
    
    def load_document_links(self, json_path):
        """Load document URLs from crawler JSON"""
        print(f"\n📎 Loading document links from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            links = json.load(f)
        
        print(f"✅ Found {len(links)} document links")
        return links
    
    def _download_and_extract_single_document(self, doc_url, index):
        """Download and extract a single document (helper for concurrent processing)"""
        try:
            # Download single document (using existing download_file method)
            downloaded_path = self.downloader.download_file(doc_url)
            
            if not downloaded_path:
                return (index, None, f"Failed to download: {doc_url}")
            
            # Extract text from the document
            docs = extract_text_from_multiple_files([downloaded_path])
            
            if not docs:
                return (index, None, f"Failed to extract text from: {downloaded_path}")
            
            # Enhance metadata
            for doc in docs:
                doc.metadata['download_url'] = doc_url
                doc.metadata['source_type'] = 'downloaded_document'
            
            with self.progress_lock:
                print(f"   ✅ [{index + 1}/{self.total_docs}] Processed: {os.path.basename(downloaded_path)}")
            
            return (index, docs, None)
            
        except Exception as e:
            return (index, None, f"Error processing {doc_url}: {str(e)}")
    
    def download_and_process_documents(self, doc_links):
        """Download documents and extract content concurrently"""
        if not doc_links:
            print("⚠️  No documents to process")
            return []
        
        self.total_docs = len(doc_links)
        print(f"\n📥 Downloading and processing {self.total_docs} documents with {self.max_workers} workers...")
        
        all_documents = []
        errors = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download and extraction tasks
            future_to_index = {
                executor.submit(self._download_and_extract_single_document, url, i): i 
                for i, url in enumerate(doc_links)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                index, docs, error = future.result()
                
                if docs is not None:
                    all_documents.extend(docs)
                else:
                    errors.append((index, error))
                
                completed += 1
        
        if errors:
            print(f"\n⚠️  {len(errors)} documents failed to process:")
            for idx, error in errors[:5]:  # Show first 5 errors
                print(f"   - Document {idx}: {error}")
        
        print(f"\n✅ Successfully processed {len(all_documents)} document pages from {len(doc_links) - len(errors)}/{len(doc_links)} documents")
        return all_documents
    
    def embed_and_store(self, documents, batch_size=10):
        """Embed documents and store in vector database with incremental saving"""
        if not documents:
            print("⚠️  No documents to embed")
            return
        
        print(f"\n🔢 Chunking {len(documents)} documents...")
        chunks = split_chunk_overlap(documents, self.chunk_size, self.chunk_overlap)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Get starting index based on existing vectors
        start_idx = self.vector_db.count()
        print(f"📍 Starting from chunk index {start_idx} (resuming if interrupted)")
        
        # Skip already processed chunks
        if start_idx >= len(chunks):
            print(f"✅ All {len(chunks)} chunks already processed!")
            return
        
        chunks_to_process = chunks[start_idx:]
        print(f"🔄 Processing {len(chunks_to_process)} remaining chunks (skipping first {start_idx})")
        
        print(f"\n🧮 Generating and storing embeddings in batches of {batch_size}...")
        
        # Process in batches
        for batch_start in range(0, len(chunks_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks_to_process))
            batch_chunks = chunks_to_process[batch_start:batch_end]
            
            # Generate embeddings for this batch
            texts = [chunk.page_content for chunk in batch_chunks]
            embeddings = self._get_ollama_embeddings(texts)
            
            # Prepare metadata for this batch
            ids = [f"chunk_{start_idx + batch_start + i}" for i in range(len(batch_chunks))]
            metadatas = [
                {
                    "source": chunk.metadata.get("source", "unknown"),
                    "filename": chunk.metadata.get("filename", "unknown"),
                    "file_type": chunk.metadata.get("file_type", "web"),
                    "source_type": chunk.metadata.get("source_type", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "title": chunk.metadata.get("title", ""),
                    "download_url": chunk.metadata.get("download_url", ""),
                    "chunk_index": batch_start + i
                }
                for i, chunk in enumerate(batch_chunks)
            ]
            
            # Add batch to vector database
            self.vector_db.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Save after each batch
            self.save_vector_db()
            
            print(f"   💾 Batch {batch_start + 1}-{batch_end}/{len(chunks_to_process)} saved. Total vectors: {self.vector_db.count()}")
        
        print(f"✅ Stored {len(chunks_to_process)} new chunks. Total vectors: {self.vector_db.count()}")
    
    def process_crawler_output(self, crawler_output_dir):
        """Process all data from crawler output directory"""
        crawler_dir = Path(crawler_output_dir)
        
        # Load web content
        rag_docs_path = crawler_dir / "rag_documents.json"
        if rag_docs_path.exists():
            web_documents = self.load_crawler_data(rag_docs_path)
            self.embed_and_store(web_documents)
        else:
            print(f"⚠️  Web documents not found: {rag_docs_path}")
        
        # Load and process document links
        doc_links_path = crawler_dir / "doc_links.json"
        if doc_links_path.exists():
            doc_links = self.load_document_links(doc_links_path)
            if doc_links:
                downloaded_documents = self.download_and_process_documents(doc_links)
                self.embed_and_store(downloaded_documents)
        else:
            print(f"⚠️  Document links not found: {doc_links_path}")
        
        # Save vector database
        self.save_vector_db()
        
        print(f"\n✅ Pipeline complete! Total vectors: {self.vector_db.count()}")
    
    def query(self, query_text, top_k=5):
        """Query the vector database"""
        print(f"\n🔍 Searching for: {query_text}")
        
        # Generate query embedding using Ollama
        query_embedding = self._get_ollama_embeddings([query_text])
        
        # Search vector database
        results = self.vector_db.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return results
    
    def display_results(self, results):
        """Display search results with metadata"""
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        print(f"\n📊 Found {len(documents)} results:\n")
        
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            print(f"{'='*80}")
            print(f"Result {i} (Score: {dist:.4f})")
            print(f"{'='*80}")
            print(f"Source Type: {meta.get('source_type', 'unknown')}")
            print(f"Source: {meta.get('source', 'unknown')}")
            if meta.get('filename') != 'unknown':
                print(f"Filename: {meta.get('filename')}")
            if meta.get('title'):
                print(f"Title: {meta.get('title')}")
            if meta.get('page', 0) > 0:
                print(f"Page: {meta.get('page')}")
            print(f"\nContent Preview:")
            print(doc[:500] + "..." if len(doc) > 500 else doc)
            print()


def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = CrawlerRAGPipeline(
        embedding_model_name="toshk0/nomic-embed-text-v2-moe:Q6_K",
        vector_db_type="FAISS",
        vector_db_path="indexes/crawler_vectordb",
        chunk_size=1000,
        chunk_overlap=200,
        ollama_base_url="http://localhost:11434"
    )
    
    # Find latest crawler output
    crawl_data_dir = Path("crawl_data")
    if not crawl_data_dir.exists():
        print("❌ No crawl_data directory found. Please run crawler.py first.")
        return
    
    # Get latest timestamp folder (filter out non-timestamp directories like 'extracted_cache')
    timestamp_dirs = sorted([
        d for d in crawl_data_dir.iterdir() 
        if d.is_dir() and d.name.replace('_', '').isdigit()
    ])
    if not timestamp_dirs:
        print("❌ No crawler output found. Please run crawler.py first.")
        return
    
    latest_dir = timestamp_dirs[-1]
    print(f"📁 Using crawler output from: {latest_dir}")
    
    # Process crawler output
    pipeline.process_crawler_output(latest_dir)
    
    # Interactive query loop
    print(f"\n{'='*80}")
    print("🤖 RAG Query Interface")
    print(f"{'='*80}")
    print("Type your question or 'quit' to exit\n")
    
    while True:
        query = input("❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        results = pipeline.query(query, top_k=3)
        pipeline.display_results(results)


if __name__ == "__main__":
    main()
