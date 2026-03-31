"use client";

import { useState, useRef, useEffect } from 'react';
import { marked } from 'marked';

const API_BASE_URL = 'http://localhost:8000/api';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isTyping?: boolean;
};

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I am the DTS AI Assistant. How can I help you track your documents today?'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [topic, setTopic] = useState('general');
  const [language, setLanguage] = useState('en');
  const [useStreaming, setUseStreaming] = useState(true);
  const [isEngineOnline, setIsEngineOnline] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check backend health
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(res => {
        if (!res.ok) throw new Error();
        setIsEngineOnline(true);
      })
      .catch(() => setIsEngineOnline(false));
  }, []);

  // Configure marked for markdown parsing
  useEffect(() => {
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }, []);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${e.target.scrollHeight}px`;
  };

  const startNewChat = () => {
    setSessionId(null);
    setMessages([
      {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Hello! I am the DTS AI Assistant. Starting a new conversation. How can I help you today?'
      }
    ]);
  };

  const handleStandardResponse = async (payload: any, botMsgId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setSessionId(data.session_id);
      
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, content: data.reply, isTyping: false }
          : msg
      ));
    } catch (error: any) {
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, content: `**Error:** Could not connect to the AI Engine. (${error.message})`, isTyping: false }
          : msg
      ));
    }
  };

  const handleStreamingResponse = async (payload: any, botMsgId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body reader.");
      const decoder = new TextDecoder('utf-8');
      
      let fullReply = '';
      let buffer = '';

      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId ? { ...msg, isTyping: false, content: '' } : msg
      ));

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (let line of lines) {
          line = line.trim();
          if (!line.startsWith('data: ')) continue;

          const dataStr = line.substring(6).trim();

          if (dataStr.startsWith('[DONE]')) {
            const doneJson = dataStr.substring(6);
            try {
              const parsed = JSON.parse(doneJson);
              if (parsed.session_id && !sessionId) setSessionId(parsed.session_id);
            } catch (e) {}
            continue;
          }

          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.error) throw new Error(parsed.error);

            if (parsed.text !== undefined) {
              fullReply += parsed.text;
              
              setMessages(prev => prev.map(msg => 
                msg.id === botMsgId ? { ...msg, content: fullReply } : msg
              ));
            } else if (parsed.session_id !== undefined && !sessionId) {
              setSessionId(parsed.session_id);
            }
          } catch (e) {}
        }
      }
    } catch (error: any) {
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, content: `**Error during stream:** ${error.message}`, isTyping: false }
          : msg
      ));
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const messageText = inputMessage.trim();
    if (!messageText) return;

    // Add user message
    const userMsgId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: messageText }]);
    
    setInputMessage('');
    const textarea = document.getElementById('messageInput');
    if (textarea) textarea.style.height = 'auto';

    // Add bot typing placeholder
    const botMsgId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: botMsgId, role: 'assistant', content: '', isTyping: true }]);

    const payload: any = { message: messageText, language, topic };
    if (sessionId) payload.session_id = sessionId;

    if (useStreaming) {
      await handleStreamingResponse(payload, botMsgId);
    } else {
      await handleStandardResponse(payload, botMsgId);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon"></div>
          <h2>DTS AI</h2>
        </div>
        
        <button className="new-chat-btn" onClick={startNewChat}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New Chat
        </button>
        
        <div className="settings-section">
          <h3>Settings</h3>
          <div className="setting-item">
            <label htmlFor="topicSelect">Topic Context</label>
            <select id="topicSelect" value={topic} onChange={(e) => setTopic(e.target.value)}>
              <option value="general">General Inquiry</option>
              <option value="document_status">Document Status</option>
              <option value="technical_support">Technical Support</option>
            </select>
          </div>
          <div className="setting-item">
            <label htmlFor="languageSelect">Response Language</label>
            <select id="languageSelect" value={language} onChange={(e) => setLanguage(e.target.value)}>
              <option value="en">English</option>
              <option value="tl">Filipino (Tagalog)</option>
            </select>
          </div>
          <div className="setting-item">
            <label className="toggle-label">
              <span>Streaming</span>
              <input type="checkbox" checked={useStreaming} onChange={(e) => setUseStreaming(e.target.checked)} />
              <span className="toggle-slider"></span>
            </label>
          </div>
        </div>
        
        <div className="sidebar-footer">
          <div className={`status-indicator ${isEngineOnline ? 'online' : 'offline'}`}></div>
          <span>Engine {isEngineOnline ? 'Online' : 'Offline'}</span>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-area">
        <header className="chat-header">
          <div className="header-info">
            <h1>AI Assistant</h1>
            <p>Next.js Internal Test Environment</p>
          </div>
        </header>

        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-avatar">{msg.role === 'user' ? 'ME' : 'AI'}</div>
              <div className="message-content">
                {msg.isTyping ? (
                  <div className="typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                ) : (
                  <div dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }} />
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <form className="chat-input-box" onSubmit={handleSubmit}>
            <textarea
              id="messageInput"
              value={inputMessage}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              rows={1}
              required
            ></textarea>
            <button type="submit" className="send-btn" disabled={!inputMessage.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </form>
          <div className="input-footer">
            Powered by Next.js & DTS AI Engine v1.0
          </div>
        </div>
      </main>
    </div>
  );
}
