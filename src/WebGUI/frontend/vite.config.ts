import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Proxy API and media to Django so the browser sees one origin in dev.
    proxy: {
      '/api': { target: process.env.BACKEND_URL || 'http://localhost:8000', changeOrigin: true },
      '/media': { target: process.env.BACKEND_URL || 'http://localhost:8000', changeOrigin: true },
    },
  },
})
