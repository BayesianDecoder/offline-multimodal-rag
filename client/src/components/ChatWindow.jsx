import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';
import MessageBubble from './MessageBubble';
import VoiceInput from './VoiceInput';
import { useSendQuery } from '../hooks/useQuery';

export default function ChatWindow() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);
  const sendQuery = useSendQuery();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const submitQuery = (text) => {
    const query = text.trim();
    if (!query || sendQuery.isPending) return;

    const loadingId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: query },
      { id: loadingId, role: 'assistant', content: '', loading: true }
    ]);
    setInput('');

    sendQuery.mutate(
      { query },
      {
        onSuccess: (data) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === loadingId
                ? {
                    ...m,
                    loading: false,
                    content: data.answer,
                    eval_scores: data.eval_scores,
                    sources: data.sources,
                    cache_hit: data.cache_hit,
                    latency_ms: data.latency_ms
                  }
                : m
            )
          );
        },
        onError: (error) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === loadingId
                ? { ...m, loading: false, content: `❌ ${error.response?.data?.error || error.message}` }
                : m
            )
          );
        }
      }
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    submitQuery(input);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center text-text-muted text-sm">
            Ask a question about the indexed legal documents to get started.
          </div>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-border p-3 flex items-center gap-2">
        <VoiceInput onTranscript={(text) => submitQuery(text)} />
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about BNS sections, judgments, FIRs..."
          className="flex-1 bg-surface border border-border rounded-full px-4 py-2 text-sm outline-none focus:border-accent transition-colors"
        />
        <button
          type="submit"
          disabled={sendQuery.isPending || !input.trim()}
          className="p-2 rounded-full bg-accent hover:bg-accent-hover disabled:opacity-40 transition-colors"
        >
          <Send size={16} />
        </button>
      </form>
    </div>
  );
}
