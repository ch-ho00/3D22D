// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 3004,
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        secure: false,
        // Optional: Rewrite the URL if needed
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
