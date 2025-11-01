import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // --- THIS IS THE FIX ---
  // We are telling Vite to target modern browsers
  // that support 'import.meta.env'
  build: {
    target: 'esnext' 
  },
  server: {
    // This ensures the dev server also supports it
    target: 'esnext'
  }
  // --- END FIX ---
})