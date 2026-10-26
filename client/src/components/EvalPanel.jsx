import { useState } from 'react';
import { ChevronDown, ChevronUp, CheckCircle2, AlertTriangle } from 'lucide-react';

function scoreColor(score) {
  if (score > 0.6) return 'text-emerald-400';
  if (score >= 0.4) return 'text-amber-400';
  return 'text-red-400';
}

function ScoreRow({ label, score }) {
  return (
    <div className="flex items-center justify-between text-sm py-1">
      <span className="text-text-muted">{label}</span>
      <span className={`font-mono font-medium ${scoreColor(score)}`}>{score.toFixed(2)}</span>
    </div>
  );
}

export default function EvalPanel({
  faithfulness = 0,
  answer_relevance = 0,
  context_recall = 0,
  overall_score = 0,
  verdict = '',
  hallucinated_sentences = []
}) {
  const [open, setOpen] = useState(false);
  const isReliable = overall_score > 0.6;

  return (
    <div className="mt-2 rounded-lg border border-border bg-surface/60 text-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-left"
      >
        <span className="flex items-center gap-1.5 text-text-muted">
          📊 Response Quality
          {isReliable ? (
            <CheckCircle2 size={14} className="text-emerald-400" />
          ) : (
            <AlertTriangle size={14} className="text-amber-400" />
          )}
          <span className={isReliable ? 'text-emerald-400' : 'text-amber-400'}>{verdict}</span>
        </span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="px-3 pb-3 border-t border-border pt-2">
          <ScoreRow label="Faithfulness" score={faithfulness} />
          <ScoreRow label="Answer Relevance" score={answer_relevance} />
          <ScoreRow label="Context Recall" score={context_recall} />
          <div className="flex items-center justify-between pt-1 mt-1 border-t border-border font-medium">
            <span>Overall</span>
            <span className={scoreColor(overall_score)}>{overall_score.toFixed(2)}</span>
          </div>

          {hallucinated_sentences.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border">
              <p className="text-text-muted mb-1">Unsupported statements:</p>
              <ul className="space-y-1">
                {hallucinated_sentences.map((item, i) => (
                  <li key={i} className="text-red-300/80">
                    <s>{item.sentence}</s>{' '}
                    <span className="text-text-muted">({item.score.toFixed(2)})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
