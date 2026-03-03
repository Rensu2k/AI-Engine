import "./style.css";

// State
let sessionId = null;
let isWaiting = false;

// DOM Elements
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const languageSelect = document.getElementById("languageSelect");
const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");

// Configuration
// Using exact relative path because we will set up the proxy in vite.config.js
const API_URL = "/api";

// Utility: Build a message element
function addMessageToUI(text, sender = "user") {
  const isAi = sender === "ai";

  const msgWrapper = document.createElement("div");
  msgWrapper.className = `message ${isAi ? "ai-message" : "user-message"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${isAi ? "ai-avatar" : "user-avatar"}`;
  avatar.textContent = isAi ? "AI" : "U";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${isAi ? "outline-bubble" : "filled-bubble"}`;
  bubble.textContent = text;

  // Assemble
  msgWrapper.appendChild(avatar);
  msgWrapper.appendChild(bubble);

  // Insert before the typing indicator if it exists, otherwise append
  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    chatWindow.insertBefore(msgWrapper, indicator);
  } else {
    chatWindow.appendChild(msgWrapper);
  }

  scrollToBottom();
}

function addSystemMessage(text) {
  const msgWrapper = document.createElement("div");
  msgWrapper.className = "message system-message";
  msgWrapper.textContent = text;

  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    chatWindow.insertBefore(msgWrapper, indicator);
  } else {
    chatWindow.appendChild(msgWrapper);
  }

  scrollToBottom();
}

// Utility: Show/Hide typing indicator
function showTypingIndicator() {
  if (document.getElementById("typingIndicator")) return;

  const typingWrapper = document.createElement("div");
  typingWrapper.id = "typingIndicator";
  typingWrapper.className = "message ai-message";

  const avatar = document.createElement("div");
  avatar.className = "avatar ai-avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble outline-bubble typing-indicator";
  bubble.innerHTML = `
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  `;

  typingWrapper.appendChild(avatar);
  typingWrapper.appendChild(bubble);
  chatWindow.appendChild(typingWrapper);
  scrollToBottom();
}

function removeTypingIndicator() {
  const indicator = document.getElementById("typingIndicator");
  if (indicator) {
    indicator.remove();
  }
}

// Utility: Scroll to bottom
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// API Call: Chat
async function sendMessageToAPI(message) {
  const language = languageSelect.value;

  const payload = {
    message: message,
    language: language,
  };

  // Only include session ID if we have one
  if (sessionId) {
    payload.session_id = sessionId;
  }

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    // Save session ID for contextual conversations
    if (data.session_id) {
      sessionId = data.session_id;
    }

    return data;
  } catch (error) {
    console.error("Chat error:", error);
    throw error;
  }
}

// API Call: TTS
async function generateTTS(text, langCode) {
  // English male default, or Filipino male
  const voice = langCode === "tl" ? "fil-PH-AngeloNeural" : "en-US-GuyNeural";

  try {
    const response = await fetch(`${API_URL}/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
        voice: voice,
      }),
    });

    if (!response.ok) {
      console.error(`TTS API error: ${response.status}`);
      return;
    }

    // Get the audio stream as a blob
    const blob = await response.blob();

    // Create an object URL from the blob
    const blobUrl = URL.createObjectURL(blob);

    // Play it
    ttsAudioPlayer.src = blobUrl;

    // Use play() wrapped in a promise catch to handle browser auto-play policies
    ttsAudioPlayer.play().catch((e) => {
      console.log("Audio playback was prevented by the browser.", e);
    });

    // Clean up the object URL after the audio finishes playing
    ttsAudioPlayer.onended = () => {
      URL.revokeObjectURL(blobUrl);
    };
  } catch (error) {
    console.error("TTS error:", error);
  }
}

// Event Listener: Form Submit
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (isWaiting) return;

  const message = messageInput.value.trim();
  if (!message) return;

  // 1. Clear input & freeze UI
  messageInput.value = "";
  isWaiting = true;
  messageInput.disabled = true;
  sendBtn.disabled = true;

  // 2. Add User message to UI
  addMessageToUI(message, "user");

  // 3. Show AI typing
  showTypingIndicator();

  try {
    // 4. Send API request
    const response = await sendMessageToAPI(message);

    // 5. Remove typing
    removeTypingIndicator();

    // 6. Output Reply
    addMessageToUI(response.reply, "ai");

    // 7. Request TTS playback in the background
    generateTTS(response.reply, languageSelect.value);
  } catch (error) {
    removeTypingIndicator();
    addSystemMessage(
      "Unable to reach the AI Engine. Ensure the backend server is running on port 8000.",
    );
  } finally {
    // 8. Restore UI
    isWaiting = false;
    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();
  }
});
