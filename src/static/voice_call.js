const socket = io();
const callBtn = document.getElementById('call-btn');
const hangupBtn = document.getElementById('hangup-btn');
const statusText = document.getElementById('status-text');
const orb = document.getElementById('orb');
const transcriptBox = document.getElementById('transcript');
const connectionStatus = document.getElementById('connection-status');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let silenceTimer;
let audioContext;
let analyser;
let dataArray;
let animationId;

// Configuration
const SILENCE_THRESHOLD = 1500; // ms to wait before sending
const MIN_DECIBELS = -50; // Threshold for voice activity

// Socket Events
socket.on('connect', () => {
    connectionStatus.classList.remove('disconnected');
    connectionStatus.classList.add('connected');
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    connectionStatus.classList.remove('connected');
    connectionStatus.classList.add('disconnected');
    updateStatus('Disconnected');
});

socket.on('status', (data) => {
    updateStatus(data.state);
    updateOrbState(data.state);
});

socket.on('transcript', (data) => {
    addMessage(data.role, data.text);
});

socket.on('audio_response', (data) => {
    playAudio(data.audio);
});

socket.on('error', (data) => {
    console.error('Server error:', data.message);
    addMessage('system', 'Error: ' + data.message);
});

// UI Controls
callBtn.addEventListener('click', startCall);
hangupBtn.addEventListener('click', endCall);

async function startCall() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Setup Audio Analysis (for visualization & VAD)
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        analyser.fftSize = 256;
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Setup MediaRecorder
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudio(audioBlob);
            audioChunks = [];
        };

        // Start VAD loop
        detectVoiceActivity();

        // UI Updates
        callBtn.classList.add('hidden');
        hangupBtn.classList.remove('hidden');
        updateStatus('listening');
        updateOrbState('listening');
        addMessage('system', 'Call started');

    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone. Please ensure you have granted permission.');
    }
}

function endCall() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (audioContext) {
        audioContext.close();
    }
    cancelAnimationFrame(animationId);

    callBtn.classList.remove('hidden');
    hangupBtn.classList.add('hidden');
    updateStatus('Ready to call');
    updateOrbState('idle');
    addMessage('system', 'Call ended');
}

// Voice Activity Detection & Visualization
function detectVoiceActivity() {
    analyser.getByteFrequencyData(dataArray);

    // Calculate average volume
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    const average = sum / dataArray.length;
    const volume = (average / 255); // 0.0 to 1.0

    // Update Orb Visuals
    const scale = 1 + volume * 0.5;
    orb.style.transform = `scale(${scale})`;

    // VAD Logic
    if (volume > 0.1) { // Speaking detected
        if (!isRecording) {
            startRecording();
        }
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(stopRecording, SILENCE_THRESHOLD);
    }

    animationId = requestAnimationFrame(detectVoiceActivity);
}

function startRecording() {
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
        isRecording = true;
        mediaRecorder.start();
        console.log('Speaking started...');
        updateOrbState('recording');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        isRecording = false;
        mediaRecorder.stop();
        console.log('Speaking stopped, sending audio...');
        updateOrbState('processing');
    }
}

function sendAudio(blob) {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        socket.emit('audio_chunk', { audio: base64Audio });
    };
}

function playAudio(base64Audio) {
    const audio = new Audio('data:audio/mp3;base64,' + base64Audio);
    updateOrbState('speaking');
    updateStatus('speaking');

    audio.onended = () => {
        updateOrbState('listening');
        updateStatus('listening');
    };

    audio.play().catch(e => console.error("Playback error:", e));
}

// Helper Functions
function updateStatus(state) {
    const statusMap = {
        'listening': 'Listening...',
        'transcribing': 'Transcribing...',
        'thinking': 'Thinking...',
        'speaking': 'Speaking...',
        'recording': 'Listening...',
        'processing': 'Processing...'
    };
    statusText.textContent = statusMap[state] || state;
}

function updateOrbState(state) {
    orb.className = 'orb'; // Reset
    orb.classList.add(state);
}

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    transcriptBox.appendChild(div);
    transcriptBox.scrollTop = transcriptBox.scrollHeight;
}
