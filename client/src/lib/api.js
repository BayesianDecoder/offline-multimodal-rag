import axios from 'axios';

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
  timeout: 150000  // 2.5 min: LLM inference on M3 8GB can take 60-90s on first load
});

export const SESSION_STORAGE_KEY = 'legal_rag_session_id';

export function getStoredSessionId() {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

export function setStoredSessionId(sessionId) {
  if (sessionId) localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

export function clearStoredSessionId() {
  localStorage.removeItem(SESSION_STORAGE_KEY);
}
