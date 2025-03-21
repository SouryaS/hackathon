// Initialize CHARACTERS variable
let CHARACTERS = {};

// Function to initialize characters from template
function initializeCharacters(characters) {
    CHARACTERS = characters;
}

// Character selection and UI state management
let selectedCharacter = 'humble_actor';
let isRecording = false;
let isTextInputMode = false;

// DOM Elements
const elements = {
    startRecording: document.getElementById('startRecording'),
    toggleTextInput: document.getElementById('toggleTextInput'),
    textInputSection: document.getElementById('textInputSection'),
    textInput: document.getElementById('textInput'),
    sendText: document.getElementById('sendText'),
    conversation: document.getElementById('conversation'),
    status: document.getElementById('status')
};

// Character selection handler
function selectCharacter(character, event) {
    selectedCharacter = character;
    document.querySelectorAll('.character-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');

    // Add visual feedback
    const feedback = document.createElement('div');
    feedback.className = 'selection-feedback';
    event.currentTarget.appendChild(feedback);
    setTimeout(() => feedback.remove(), 500);
}

// Toggle between voice and text input modes
function toggleTextInput() {
    isTextInputMode = !isTextInputMode;
    
    if (isTextInputMode) {
        elements.textInputSection.classList.remove('hidden');
        elements.startRecording.classList.add('hidden');
        elements.toggleTextInput.textContent = 'Use Voice Instead';
        elements.toggleTextInput.className = 'btn btn-primary';
        elements.textInput.focus();
    } else {
        elements.textInputSection.classList.add('hidden');
        elements.startRecording.classList.remove('hidden');
        elements.toggleTextInput.textContent = 'Type Instead';
        elements.toggleTextInput.className = 'btn btn-secondary';
    }

    // Add smooth transition
    elements.textInputSection.style.opacity = isTextInputMode ? '1' : '0';
}

// Send text message
async function sendText() {
    const text = elements.textInput.value.trim();
    if (!text) return;
    
    try {
        elements.status.textContent = 'Processing...';
        addMessage('You', text, 'user');
        elements.textInput.value = '';
        
        const response = await fetch('/process_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `text=${encodeURIComponent(text)}&character=${selectedCharacter}`
        });

        const data = await response.json();
        
        if (data.success) {
            addMessage(CHARACTERS[selectedCharacter].name, data.response, 'assistant');
            speakResponse(data.response);
            elements.status.textContent = '';
        } else {
            console.error('Error from server:', data.error);
            elements.status.textContent = `Error: ${data.error || 'Unknown error'}`;
        }
    } catch (error) {
        console.error('Error:', error);
        elements.status.textContent = `Error processing text: ${error.message}`;
    }
}

// Voice recording handler
async function startRecording() {
    if (isRecording) return;

    try {
        elements.startRecording.textContent = 'Recording...';
        elements.startRecording.classList.add('recording');
        elements.startRecording.disabled = true;
        isRecording = true;

        let timeLeft = 10;
        const countdownInterval = setInterval(() => {
            elements.status.textContent = `Recording... ${timeLeft} seconds left`;
            timeLeft--;
            if (timeLeft < 0) clearInterval(countdownInterval);
        }, 1000);

        const response = await fetch('/process_voice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `character=${selectedCharacter}`
        });

        const data = await response.json();
        clearInterval(countdownInterval);
        
        if (data.success) {
            elements.status.textContent = 'Processing...';
            addMessage('You', data.text, 'user');
            addMessage(CHARACTERS[selectedCharacter].name, data.response, 'assistant');
            speakResponse(data.response);
            elements.status.textContent = '';
        } else {
            elements.status.textContent = `Error: ${data.error || 'Unknown error'}`;
        }
    } catch (error) {
        console.error('Error:', error);
        elements.status.textContent = 'Error processing audio';
    } finally {
        elements.startRecording.textContent = 'Start Recording';
        elements.startRecording.classList.remove('recording');
        elements.startRecording.disabled = false;
        isRecording = false;
    }
}

// Add message to conversation
function addMessage(sender, text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    messageDiv.innerHTML = `
        <div class="message-sender">${sender}</div>
        <div class="message-content">${text}</div>
        <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    elements.conversation.insertBefore(messageDiv, elements.conversation.firstChild);

    // Add entrance animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    requestAnimationFrame(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    });
}

// Text-to-speech handler
function speakResponse(text) {
    try {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
    } catch (error) {
        console.error('Error speaking response:', error);
    }
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', () => {
    elements.startRecording.onclick = startRecording;
    elements.toggleTextInput.onclick = toggleTextInput;
    elements.sendText.onclick = sendText;
    elements.textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendText();
    });
});