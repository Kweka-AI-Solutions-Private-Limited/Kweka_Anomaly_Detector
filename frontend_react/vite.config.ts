import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  // Proxy target for local development server:
  // Can be set via VITE_PROXY_TARGET in .env to point to Cloud Run or local uvicorn
  const proxyTarget =
    env.VITE_PROXY_TARGET ||
    env.VITE_API_TARGET ||
    'https://kweka-anomaly-detector-238644809220.asia-south1.run.app';

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 5173,
      host: true,
      hmr: {
        host: 'localhost',
        port: 5173,
      },
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          secure: false,
        },
        '/storage': {
          target: proxyTarget,
          changeOrigin: true,
          secure: false,
        },
        '/data': {
          target: proxyTarget,
          changeOrigin: true,
          secure: false,
        },
      },
    },
  };
});
