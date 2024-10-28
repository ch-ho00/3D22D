// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3003,
    proxy: {
      '/api': 'http://localhost:3001',
    },
  },
});
