import axios from 'axios';

// Production Backend URL fallback for hosted production deployments (Firebase, Cloud Run, Vercel, etc.)
const DEFAULT_PROD_API_URL = 'https://kweka-anomaly-detector-238644809220.asia-south1.run.app';

const getBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/$/, '');
  }
  // If building for production without explicit VITE_API_BASE_URL, default to Cloud Run backend
  if (import.meta.env.PROD) {
    return DEFAULT_PROD_API_URL;
  }
  // Local dev: leave empty so Vite proxy forwards /api to local uvicorn
  return '';
};

const BASE_URL = getBaseUrl();
export const DB_NAME = import.meta.env.VITE_DB_NAME || 'Anomaly_Detector';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 45000,
});

// Automatic User Isolation Context Interceptor
apiClient.interceptors.request.use((config) => {
  const userId = localStorage.getItem('user_id') || 'usr_default';
  const token = localStorage.getItem('auth_token');
  config.headers['X-User-ID'] = userId;
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
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
