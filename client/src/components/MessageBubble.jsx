import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check, ThumbsUp, ThumbsDown } from 'lucide-react';
import EvalPanel from './EvalPanel';
import SourceViewer from './SourceViewer';

function LoadingDots() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const hint =
    elapsed >= 60
      ? 'Still thinking — LLM running on-device…'
      : elapsed >= 15
      ? 'Generating answer (this can take 30-90s on first load)…'
      : null;

  return (
    <div className="py-1">
      <div className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-text-muted animate-pulse"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      {hint && <p className="text-[11px] text-text-muted mt-1.5">{hint}</p>}
    </div>
  );
}

export default function MessageBubble({ message }) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const isUser = message.role === 'user';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-tr-sm bg-accent text-white px-4 py-2.5">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-surface border border-border px-4 py-2.5">
        {message.loading ? (
          <LoadingDots />
        ) : (
          <>
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>

            {message.eval_scores && <EvalPanel {...message.eval_scores} />}
            {message.sources && <SourceViewer sources={message.sources} />}

            <div className="flex items-center gap-2 mt-2 pt-2 border-t border-border/60">
              <button
                onClick={handleCopy}
                className="text-text-muted hover:text-text transition-colors"
                title="Copy"
              >
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </button>
              <button
                onClick={() => setFeedback(feedback === 'up' ? null : 'up')}
                className={`transition-colors ${feedback === 'up' ? 'text-emerald-400' : 'text-text-muted hover:text-text'}`}
                title="Good response"
              >
                <ThumbsUp size={14} />
              </button>
              <button
                onClick={() => setFeedback(feedback === 'down' ? null : 'down')}
                className={`transition-colors ${feedback === 'down' ? 'text-red-400' : 'text-text-muted hover:text-text'}`}
                title="Bad response"
              >
                <ThumbsDown size={14} />
              </button>
              {message.cache_hit && (
                <span className="ml-auto text-[10px] text-secondary/70">⚡ cached</span>
              )}
              {typeof message.latency_ms === 'number' && (
                <span className={message.cache_hit ? 'text-[10px] text-text-muted' : 'ml-auto text-[10px] text-text-muted'}>
                  {message.latency_ms}ms
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
