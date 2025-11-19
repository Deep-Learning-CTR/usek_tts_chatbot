from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
import asyncio
from crawl4ai import AsyncWebCrawler, LXMLWebScrapingStrategy
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import json
from datetime import datetime
from pathlib import Path
import os
from PIL import Image
import pytesseract
import requests
from io import BytesIO

# Set Tesseract path for Windows (adjust if installed elsewhere)
if os.name == 'nt':  # Windows
    tesseract_path = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def extract_text_from_images(images_data):
    """Extract text from images using OCR"""
    ocr_results = []

    for idx, img_data in enumerate(images_data, 1):
        try:
            img_url = img_data.get('src', '')
            if not img_url:
                continue

            # Handle relative URLs
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif img_url.startswith('/'):
                # This would need the base URL; skip for now
                continue
            elif not img_url.startswith('http'):
                continue

            # Download and process image
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))

                # Perform OCR
                ocr_text = pytesseract.image_to_string(img)

                if ocr_text.strip():
                    ocr_results.append({
                        'image_number': idx,
                        'image_url': img_url,
                        'alt_text': img_data.get('alt', ''),
                        'ocr_text': ocr_text.strip()
                    })
                    print(f"  ✓ OCR extracted from image {idx}: {len(ocr_text)} chars")

        except Exception as e:
            print(f"  ✗ Error processing image {idx}: {e}")
            continue

    return ocr_results


def save_to_json(data, filename):
    """Save crawled data to JSON file optimized for RAG pipeline"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Data saved to JSON: {filename}")
    print(f"✓ Total documents: {len(data)}")


async def main():
    # Minimal browser config compatible with Crawl4AI 0.7.7
    browser_config = BrowserConfig(
        headless=True  # show the browser window (often helps with JS sites)
    )

    # Deep crawl strategy – depth 4 as you wanted
    strategy = BFSDeepCrawlStrategy(
        max_depth=4,
        include_external=False,    # stay on the same domain
        # max_pages=30,
        # score_threshold=0.5,
    )

    # NOTE: main bug before was the typo "networidle"
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        wait_until='domcontentloaded',     # <- valid value now
        delay_before_return_html=2.0,      # wait a bit for JS
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:

        # Start from USEK home (you can change this to other subpages later)
        results = await crawler.arun(
            url="https://www.usek.edu.lb/en/home",
            config=run_config
        )

        print(f"Crawled {len(results)} pages in total")

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"crawl_data/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for RAG pipeline
        documents = []
        ext = ('pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv', 'txt', 'md')
        doc_links = []

        for idx, result in enumerate(results, 1):
            print(f"[{idx}/{len(results)}] Processing: {result.url}")

            # Collect downloadable doc links
            if result.url.split('.')[-1].lower() in ext:
                doc_links.append(result.url)
                continue  # continue to next result

            # Extract images and perform OCR
            ocr_data = []
            if result.media and "images" in result.media:
                images = result.media["images"]
                print(f"  Found {len(images)} images, performing OCR...")
                ocr_data = extract_text_from_images(images)

            # Build OCR text content
            ocr_content = ""
            if ocr_data:
                ocr_content = "\n\n[IMAGE OCR EXTRACTIONS]\n"
                for ocr_item in ocr_data:
                    ocr_content += f"\n--- Image {ocr_item['image_number']} ---\n"
                    if ocr_item['alt_text']:
                        ocr_content += f"Alt Text: {ocr_item['alt_text']}\n"
                    ocr_content += f"OCR Text:\n{ocr_item['ocr_text']}\n"
                    ocr_content += f"Image URL: {ocr_item['image_url']}\n"
                ocr_content += "\n[END IMAGE OCR EXTRACTIONS]\n"

            # Combine markdown content with OCR text
            main_content = result.markdown if result.markdown else ""
            combined_content = main_content + ocr_content

            # Document structure for RAG
            document = {
                'id': f"doc_{timestamp}_{idx}",
                'content': combined_content,
                'metadata': {
                    'url': result.url,
                    'title': result.metadata.get('title', 'No title'),
                    'depth': result.metadata.get('depth', 0),
                    'crawl_timestamp': timestamp,
                    'source': 'usek.edu.lb',
                    'content_length': len(combined_content),
                    'has_content': bool(combined_content.strip()),
                    'has_ocr': len(ocr_data) > 0,
                    'image_count': len(result.media.get("images", [])) if result.media else 0,
                    'ocr_image_count': len(ocr_data),
                },
                'success': result.success,
                'error_message': result.error_message if not result.success else None
            }

            documents.append(document)

        # Save JSONs
        save_to_json(documents, output_dir / "rag_documents.json")
        save_to_json(doc_links, output_dir / "doc_links.json")
        print(f"\n📁 RAG-ready data saved to: {output_dir}")
        print(f"📄 File: rag_documents.json")


if __name__ == "__main__":
    asyncio.run(main())
