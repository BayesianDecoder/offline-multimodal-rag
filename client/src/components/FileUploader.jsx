import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileText, Image as ImageIcon, AudioLines, UploadCloud, CheckCircle2 } from 'lucide-react';
import { api } from '../lib/api';

const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff'];
const AUDIO_EXT = ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus', 'wma', 'aac'];

function modalityOf(filename) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'pdf' || ext === 'docx' || ext === 'doc') return 'document';
  if (IMAGE_EXT.includes(ext)) return 'image';
  if (AUDIO_EXT.includes(ext)) return 'audio';
  return 'document';
}

const ICONS = {
  document: <FileText size={16} className="text-accent" />,
  image: <ImageIcon size={16} className="text-secondary" />,
  audio: <AudioLines size={16} className="text-emerald-400" />
};

export default function FileUploader({ onIndexed }) {
  const [files, setFiles] = useState([]);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((accepted) => {
    setFiles(accepted);
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    setProgress(0);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    try {
      const response = await api.post('/api/documents', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (evt) => {
          if (evt.total) setProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      });
      setResult(response.data);
      onIndexed?.(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div
        {...getRootProps()}
        className={`rounded-xl border-2 border-dashed p-6 text-center text-sm cursor-pointer transition-colors ${
          isDragActive ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/60'
        }`}
      >
        <input {...getInputProps()} />
        <UploadCloud size={22} className="mx-auto mb-2 text-text-muted" />
        <p className="text-text-muted">
          {isDragActive ? 'Drop files here' : 'Drag & drop legal documents, or click to browse'}
        </p>
        <p className="text-text-muted text-xs mt-1">PDF, DOCX, images, audio</p>
      </div>

      {files.length > 0 && (
        <ul className="mt-3 space-y-1">
          {files.map((file) => (
            <li key={file.name} className="flex items-center gap-2 text-xs text-text-muted">
              {ICONS[modalityOf(file.name)]}
              {file.name}
            </li>
          ))}
        </ul>
      )}

      {files.length > 0 && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="mt-3 w-full py-2 rounded-lg bg-accent hover:bg-accent-hover disabled:opacity-50 text-sm transition-colors"
        >
          {uploading ? `Uploading... ${progress}%` : 'Upload & index'}
        </button>
      )}

      {uploading && (
        <div className="mt-2 h-1.5 rounded-full bg-surface overflow-hidden">
          <div className="h-full bg-accent transition-all" style={{ width: `${progress}%` }} />
        </div>
      )}

      {result && (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-emerald-400">
          <CheckCircle2 size={14} /> Indexed — {result.total_chunks} chunks
        </p>
      )}
      {error && <p className="mt-2 text-xs text-red-400">❌ {error}</p>}
    </div>
  );
}
