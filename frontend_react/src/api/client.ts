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
  timeout: 45000,
});

// Automatic retry interceptor for handling Cloud Run cold starts and transient 502/503/504 errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const config = error.config;
    if (!config) return Promise.reject(error);

    config.__retryCount = config.__retryCount || 0;
    const MAX_RETRIES = 3;

    const isRetryable =
      !error.response ||
      error.response.status === 503 ||
      error.response.status === 502 ||
      error.response.status === 504;

    if (isRetryable && config.__retryCount < MAX_RETRIES) {
      config.__retryCount += 1;
      const delayMs = config.__retryCount * 1500;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      return apiClient(config);
    }

    return Promise.reject(error);
  }
);

export function getStorageUrl(path?: string | null): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  return `${BASE_URL}/${cleanPath}`;
}
