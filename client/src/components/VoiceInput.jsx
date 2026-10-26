import { useRef, useState } from 'react';
import { Mic, MicOff } from 'lucide-react';

// Uses the browser's built-in SpeechRecognition (Chrome/Edge) for live
// transcription. There's no server-side speech-to-text-at-query-time
// endpoint in this stack (audio is only transcribed during document
// ingestion), so this stays client-side rather than faking an upload.
const SpeechRecognitionAPI =
  typeof window !== 'undefined' ? window.SpeechRecognition || window.webkitSpeechRecognition : null;

export default function VoiceInput({ onTranscript }) {
  const [recording, setRecording] = useState(false);
  const recognitionRef = useRef(null);

  if (!SpeechRecognitionAPI) {
    return (
      <button
        type="button"
        disabled
        title="Voice input isn't supported in this browser"
        className="p-2 rounded-full text-text-muted opacity-40 cursor-not-allowed"
      >
        <MicOff size={16} />
      </button>
    );
  }

  const startRecording = () => {
    const recognition = new SpeechRecognitionAPI();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onTranscript(transcript);
    };
    recognition.onerror = () => setRecording(false);
    recognition.onend = () => setRecording(false);

    recognitionRef.current = recognition;
    recognition.start();
    setRecording(true);
  };

  const stopRecording = () => {
    recognitionRef.current?.stop();
    setRecording(false);
  };

  return (
    <button
      type="button"
      onClick={recording ? stopRecording : startRecording}
      title={recording ? 'Stop recording' : 'Ask by voice'}
      className={`p-2 rounded-full transition-colors ${
        recording ? 'bg-red-500/20 text-red-400 animate-pulse' : 'text-text-muted hover:text-text'
      }`}
    >
      <Mic size={16} />
    </button>
  );
}
