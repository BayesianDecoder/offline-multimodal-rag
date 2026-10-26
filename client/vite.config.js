import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': { target: process.env.VITE_API_URL || 'http://localhost:5000', changeOrigin: true }
    }
  },
  preview: {
    port: 3000
  }
})
