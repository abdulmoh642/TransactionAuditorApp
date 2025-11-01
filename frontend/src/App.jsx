import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, UploadCloud, Loader2, CheckCircle, AlertTriangle, X, File as FileIcon, Bot, User, CornerDownLeft, Sparkles } from 'lucide-react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
//printf("Environment Variable VITE_API_BASE_URL:", import.meta.env.VITE_API_BASE_URL);
const API_BASE_URL = import.meta.env.VITE_API_URL //|| 'http://localhost:8000';
//const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
if (!API_BASE_URL) {
  console.error("CRITICAL: VITE_API_BASE_URL is not set. The app will not work.");
  // You could render an error page here
}
const API_HEALTH_URL = `${API_BASE_URL}/health`;
const API_CHAT_URL = `${API_BASE_URL}/chat`;
const API_SUMMARIZE_URL = `${API_BASE_URL}/summarize`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState({ online: false, message: 'Connecting...' });
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [totalTokens, setTotalTokens] = useState(0);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  const checkApiHealth = useCallback(async () => {
    try {
      const response = await fetch(API_HEALTH_URL);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      if (data.status === "Agent API is running with FastAPI.") {
        setApiStatus({ online: true, message: 'Online' });
      } else {
        setApiStatus({ online: false, message: 'Offline' });
      }
    } catch (error) {
      console.error("API health check failed:", error);
      let errorMessage = `API health check failed. Error: ${error.message}`;
      setApiStatus({ online: false, message: 'Offline', error: errorMessage });
    }
  }, []);

  useEffect(() => {
    setMessages([
      { role: 'bot', content: "Hello! I'm an expert auditor. I can help you by searching documents, generating summary reports, and performing some calculations. What can I do for you today?" }
    ]);
    
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, [checkApiHealth]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const readFileAsDataURL = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };

  const handleSendMessage = useCallback(async (messageContent) => {
    const text = messageContent.trim();
    if ((!text && attachedFiles.length === 0) || isLoading) return;

    const userMessage = { role: 'user', content: text, files: attachedFiles.map(f => f.name) };
    
    let chatHistory = [];
    setMessages(prevMessages => {
        chatHistory = prevMessages
            .map(m => [
                m.role === 'user' ? m.content : '',
                m.role === 'bot' ? m.content : ''
            ])
            .slice(-10);
        return [...prevMessages, userMessage];
    });

    setIsLoading(true);
    setInput('');
    const currentAttachedFiles = [...attachedFiles];
    setAttachedFiles([]);

    let filesPayload = [];
    try {
      filesPayload = await Promise.all(
        currentAttachedFiles.map(async (file) => {
          const content = await readFileAsDataURL(file);
          return { name: file.name, content: content };
        })
      );
    } catch (error) {
      console.error("Error reading files:", error);
      const errorMessage = { role: 'bot', content: `Error reading file before upload: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
      return;
    }

    const payload = {
      chat_history: chatHistory,
      message: text,
      files: filesPayload
    };

    try {
      const response = await fetch(API_CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Unknown server error" }));
        throw new Error(`HTTP error! status: ${response.status}. ${errData.detail}`);
      }

      const data = await response.json();

      if (data.success) {
        setMessages(prev => [...prev, { role: 'bot', content: data.response }]);
        setTotalTokens(prev => prev + (data.tokens_used_this_turn || 0));
      } else {
        throw new Error(data.detail || "API returned an unsuccessful response.");
      }

    } catch (error) {
      console.error("API call failed:", error);
      const errorMessage = { role: 'bot', content: `Sorry, there was an issue communicating with the API. Error: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, attachedFiles]);

  const handleSummarize = useCallback(async () => {
    setIsLoading(true);
    
    const chatHistory = messages.map(m => [
      m.role === 'user' ? m.content : '',
      m.role === 'bot' ? m.content : ''
    ]);

    const payload = { chat_history: chatHistory };

    try {
      const response = await fetch(API_SUMMARIZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Unknown server error" }));
        throw new Error(`HTTP error! status: ${response.status}. ${errData.detail}`);
      }

      const data = await response.json();

      if (data.success) {
        const summaryMessage = { 
          role: 'bot', 
          content: `✨ **Conversation Summary:**\n\n${data.summary}` 
        };
        setMessages(prev => [...prev, summaryMessage]);
      } else {
        throw new Error(data.detail || "API returned an unsuccessful summary response.");
      }

    } catch (error) {
      console.error("Summarize API call failed:", error);
      const errorMessage = { role: 'bot', content: `Sorry, I failed to generate a summary. Error: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [messages]);

  const handleFileChange = (e) => {
    console.log('File input changed:', e.target.files);
    if (e.target.files) {
      const filesArray = Array.from(e.target.files);
      console.log('Files to attach:', filesArray.map(f => f.name));
      setAttachedFiles(prev => [...prev, ...filesArray]);
      e.target.value = null;
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    console.log('Files dropped:', e.dataTransfer.files);
    if (e.dataTransfer.files) {
      const filesArray = Array.from(e.dataTransfer.files);
      console.log('Files to attach from drop:', filesArray.map(f => f.name));
      setAttachedFiles(prev => [...prev, ...filesArray]);
    }
  };

  const removeFile = (fileName) => {
    setAttachedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  return (
    <div className="flex h-screen w-screen flex-col bg-gray-900 text-gray-200 font-sans">
      <Header status={apiStatus} />
      
      <div className="flex flex-1 overflow-hidden">
        <main className="flex-1 flex flex-col overflow-hidden">
          <ChatMessages messages={messages} chatEndRef={chatEndRef} />
        </main>

        <aside className="w-[300px] lg:w-[350px] bg-gray-950/50 border-l border-gray-700/50 flex flex-col p-4 space-y-4">
          <TokenUsage tokens={totalTokens} />
          <FileAttachmentSidebar
            attachedFiles={attachedFiles}
            fileInputRef={fileInputRef}
            handleFileChange={handleFileChange}
            handleDragOver={handleDragOver}
            handleDrop={handleDrop}
            removeFile={removeFile}
          />
          
          <button
            onClick={handleSummarize}
            disabled={isLoading || messages.length <= 1}
            className="w-full flex justify-center items-center gap-2 rounded-lg bg-indigo-600/80 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500/80 disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            ✨ Summarize Conversation
          </button>

          <button
            onClick={() => {
              setMessages([
                { role: 'bot', content: "Hello! I'm an expert auditor. I can help you by searching documents, generating summary reports, and performing some calculations. What can I do for you today?" }
              ]);
              setAttachedFiles([]);
              setTotalTokens(0);
            }}
            className="w-full flex justify-center items-center gap-2 rounded-lg bg-red-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-red-500 disabled:opacity-50"
          >
            Clear Chat
          </button>
        </aside>
      </div>

      <footer className="border-t border-gray-700/50 p-4 bg-gray-900">
        <ExampleBar onExampleClick={(text) => {
          setInput(text);
          document.getElementById('chat-input')?.focus();
        }} />
        <InputForm
          input={input}
          setInput={setInput}
          isLoading={isLoading}
          handleSendMessage={handleSendMessage}
          attachedFiles={attachedFiles}
        />
      </footer>
    </div>
  );
}

function Header({ status }) {
  return (
    <header className="flex-shrink-0 flex items-center justify-between p-4 border-b border-gray-700/50">
      <div className="flex items-center gap-3">
        <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2 rounded-lg">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><path d="M4.5 16.5c-1.5 1.5-1.5 3.9 0 5.4 1.5 1.5 3.9 1.5 5.4 0 1.5-1.5 1.5-3.9 0-5.4M12.5 11.5l6-6M11.5 1.5l11 11"></path></svg>
        </div>
        <h1 className="text-xl font-semibold text-white">Transaction Auditor Agent</h1>
      </div>
      <div className="flex items-center gap-4 text-sm text-gray-400">
        <span>Model: <span className="text-gray-200">gemini-2.5-flash</span></span>
        <span>Data Access: <span className="text-gray-200">Live</span></span>
        <div className="flex items-center gap-2">
          Agent Status:
          {status.online ? (
            <span className="flex items-center gap-1.5 text-green-400">
              <CheckCircle className="w-4 h-4" /> Online
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-red-400" title={status.error}>
              <AlertTriangle className="w-4 h-4" /> Offline
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

function ChatMessages({ messages, chatEndRef }) {
  const formatContent = (content) => {
    try {
      if (typeof content === 'string' && content.trim().startsWith('{')) {
        const data = JSON.parse(content);
        if (data.report_title && data.findings) {
          let reportHtml = `<h2 class="text-xl font-semibold mb-2">${DOMPurify.sanitize(data.report_title)}</h2>`;
          reportHtml += `<p class="mb-4">${DOMPurify.sanitize(data.summary)}</p>`;
          reportHtml += '<ul class="list-disc pl-5 space-y-2">';
          data.findings.forEach(finding => {
            reportHtml += `<li><strong>${DOMPurify.sanitize(finding.finding_title)}:</strong> ${DOMPurify.sanitize(finding.description)}</li>`;
          });
          reportHtml += '</ul>';
          return reportHtml;
        }
      }
    } catch (e) {
    }
    return DOMPurify.sanitize(marked.parse(content, { gfm: true, breaks: true }));
  };

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6">
      {messages.length === 1 && (
         <div className="flex h-full items-center justify-center">
            <p className="text-gray-500">Conversation history will appear here.</p>
         </div>
      )}
      {messages.map((msg, index) => (
        <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
          <div className={`flex items-start gap-3 max-w-[70%]`}>
            {msg.role === 'bot' && (
              <div className="bg-gray-700 p-2.5 rounded-full flex-shrink-0">
                <Bot className="w-5 h-5 text-indigo-300" />
              </div>
            )}
            <div
              className={`rounded-xl px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-200'
              }`}
            >
              <div
                className="prose prose-sm prose-dark max-w-none text-inherit"
                dangerouslySetInnerHTML={{ __html: formatContent(msg.content) }}
              />
              {msg.files && msg.files.length > 0 && (
                <div className="mt-2 pt-2 border-t border-indigo-400/50">
                  <span className="text-xs font-medium text-indigo-200">Attached: {msg.files.join(', ')}</span>
                </div>
              )}
            </div>
             {msg.role === 'user' && (
              <div className="bg-gray-700 p-2.5 rounded-full flex-shrink-0">
                <User className="w-5 h-5 text-gray-300" />
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={chatEndRef} />
    </div>
  );
}

function TokenUsage({ tokens }) {
  return (
    <div className="bg-gray-800/60 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-indigo-300 mb-3">Current Session Token Usage (Accumulated)</h3>
      <div className="text-4xl font-bold text-white">{tokens}</div>
      <p className="text-xs text-gray-400 mt-1">Total Tokens</p>
    </div>
  );
}

function FileAttachmentSidebar({ attachedFiles, fileInputRef, handleFileChange, handleDragOver, handleDrop, removeFile }) {
  const triggerFileInput = (e) => {
    e.preventDefault();
    console.log('Upload area clicked, triggering file input');
    fileInputRef.current?.click();
  };

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 flex flex-col flex-grow min-h-[200px]">
      <h3 className="text-sm font-semibold text-indigo-300 mb-3">Attach documents for Analysis (Excel/PDF)</h3>
      
      <div
        className="flex-grow border-2 border-dashed border-gray-600 rounded-lg flex flex-col items-center justify-center text-gray-500 hover:border-indigo-500 hover:text-indigo-400 transition-colors cursor-pointer"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={triggerFileInput}
      >
        <UploadCloud className="w-10 h-10 mb-2" />
        <span className="text-sm font-semibold">Drag & Drop File Here</span>
        <span className="text-xs">- or -</span>
        <span className="text-sm font-semibold mt-1">Click to Upload</span>
      </div>
      
      <input
        id="file-upload-input"
        type="file"
        multiple
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".xlsx, .xls, .pdf"
      />

      {attachedFiles.length > 0 && (
        <div className="mt-3 space-y-2 max-h-[150px] overflow-y-auto pr-1">
          <p className="text-xs text-gray-400">{attachedFiles.length} document(s) attached:</p>
          {attachedFiles.map((file, index) => (
            <div key={index} className="flex items-center justify-between bg-gray-700 p-2 rounded-md">
              <div className="flex items-center gap-2 overflow-hidden">
                <FileIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <span className="text-sm text-gray-200 truncate" title={file.name}>{file.name}</span>
              </div>
              <button onClick={() => removeFile(file.name)} className="text-gray-500 hover:text-red-400 flex-shrink-0">
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
      {attachedFiles.length === 0 && (
         <p className="text-xs text-gray-500 text-center mt-3">No documents attached.</p>
      )}
    </div>
  );
}

function InputForm({ input, setInput, isLoading, handleSendMessage, attachedFiles }) { 
  const textAreaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSendMessage(input);
  };

  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
      const scrollHeight = textAreaRef.current.scrollHeight;
      textAreaRef.current.style.height = `${scrollHeight}px`;
    }
  }, [input]);
  
  return (
    <form onSubmit={handleSubmit} className="relative flex items-end gap-3">
      <textarea
        id="chat-input"
        ref={textAreaRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
          }
        }}
        rows={1}
        placeholder="Ask your audit question..."
        className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 pr-12 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none overflow-y-auto"
        style={{ maxHeight: '150px' }}
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={isLoading || (!input.trim() && attachedFiles.length === 0)}
        className="absolute right-3 bottom-[9px] bg-indigo-600 p-2 rounded-lg text-white hover:bg-indigo-500 disabled:bg-gray-600 disabled:opacity-70 transition-colors"
      >
        {isLoading ? (
          <Loader2 className="w-5 h-5 animate-spin" />
        ) : (
          <Send className="w-5 h-5" />
        )}
      </button>
    </form>
  );
}

function ExampleBar({ onExampleClick }) {
  const examples = [
    'What is the square of 42?',
    'Generate a final summary report.',
    'Which transactions require manager approval?',
    'Identify any breaches in transactions.',
  ];

  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-xs font-medium text-gray-400">Examples:</span>
      <div className="flex flex-wrap gap-2">
        {examples.map((ex, i) => (
          <button
            key={i}
            onClick={() => onExampleClick(ex)}
            className="flex items-center gap-1.5 bg-gray-800/80 hover:bg-gray-700/80 border border-gray-700 px-2.5 py-1 rounded-lg text-xs text-gray-300 transition-colors"
          >
            {ex} <CornerDownLeft className="w-3 h-3" />
          </button>
        ))}
      </div>
    </div>
  );
}
