import { useState } from 'react';
import { FileText, X } from 'lucide-react';

export default function SourceViewer({ sources = [] }) {
  const [expanded, setExpanded] = useState(null);

  if (sources.length === 0) return null;

  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {sources.map((source, i) => (
        <button
          key={i}
          onClick={() => setExpanded(expanded === i ? null : i)}
          className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-surface border border-border text-text-muted hover:text-secondary hover:border-secondary transition-colors"
        >
          <FileText size={12} />
          {source.source}
        </button>
      ))}

      {expanded !== null && sources[expanded] && (
        <div className="w-full mt-1 p-3 rounded-lg bg-surface border border-border text-xs relative">
          <button
            onClick={() => setExpanded(null)}
            className="absolute top-2 right-2 text-text-muted hover:text-text"
          >
            <X size={14} />
          </button>
          <p className="font-medium text-text mb-1 pr-6">{sources[expanded].source}</p>
          <p className="text-text-muted whitespace-pre-wrap">{sources[expanded].preview}</p>
          <p className="text-text-muted mt-1">Score: {Number(sources[expanded].score).toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}
