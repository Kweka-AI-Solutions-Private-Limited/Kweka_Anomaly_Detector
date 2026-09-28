import axios from 'axios';

// Production: VITE_API_BASE_URL is set at Docker build time (ARG VITE_API_BASE_URL=https://your-app.run.app)
// Local dev: leave empty — Vite proxy in vite.config.ts forwards /api /storage /data to localhost:8000
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
export const DB_NAME = import.meta.env.VITE_DB_NAME || 'Anomaly_Detector';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function getStorageUrl(path?: string | null): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return `${BASE_URL}/${cleanPath}`;
}
