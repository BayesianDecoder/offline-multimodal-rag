import { useQuery } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, BarChart3, FileStack } from 'lucide-react';
import FileUploader from './FileUploader';
import { api, getStoredSessionId } from '../lib/api';

export default function Sidebar() {
  const location = useLocation();
  const currentSessionId = getStoredSessionId();

  const documentsQuery = useQuery({
    queryKey: ['documents'],
    queryFn: async () => (await api.get('/api/documents')).data
  });

  const sessionsQuery = useQuery({
    queryKey: ['sessions'],
    queryFn: async () => (await api.get('/api/sessions')).data
  });

  return (
    <aside className="w-72 shrink-0 border-r border-border h-full flex flex-col p-4 overflow-y-auto">
      <h1 className="text-lg font-semibold mb-4">⚖️ Legal RAG</h1>

      <nav className="flex gap-1 mb-5 text-sm">
        <Link
          to="/"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${
            location.pathname === '/' ? 'bg-accent text-white' : 'text-text-muted hover:bg-surface'
          }`}
        >
          <MessageSquare size={14} /> Chat
        </Link>
        <Link
          to="/analytics"
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${
            location.pathname === '/analytics' ? 'bg-accent text-white' : 'text-text-muted hover:bg-surface'
          }`}
        >
          <BarChart3 size={14} /> Analytics
        </Link>
      </nav>

      <FileUploader onIndexed={() => documentsQuery.refetch()} />

      <div className="mt-6">
        <p className="text-xs font-medium text-text-muted flex items-center gap-1.5 mb-2">
          <FileStack size={13} /> Indexed documents
        </p>
        <ul className="space-y-1 text-xs">
          {documentsQuery.data?.documents?.map((doc) => (
            <li key={doc._id} className="text-text-muted truncate" title={doc.filename}>
              {doc.filename}
            </li>
          ))}
          {!documentsQuery.data?.documents?.length && (
            <li className="text-text-muted/60">No documents indexed yet</li>
          )}
        </ul>
      </div>

      <div className="mt-6">
        <p className="text-xs font-medium text-text-muted mb-2">Recent sessions</p>
        <ul className="space-y-1 text-xs">
          {sessionsQuery.data?.sessions?.map((s) => (
            <li
              key={s.session_id}
              className={`truncate ${s.session_id === currentSessionId ? 'text-secondary' : 'text-text-muted'}`}
              title={s.session_id}
            >
              {s.session_id.slice(0, 8)}… · {s.turn_count} turns
            </li>
          ))}
          {!sessionsQuery.data?.sessions?.length && <li className="text-text-muted/60">No sessions yet</li>}
        </ul>
      </div>
    </aside>
  );
}
