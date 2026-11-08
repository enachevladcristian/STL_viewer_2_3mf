import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { allowedHosts: true },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],
          icons: ['lucide-react'],
        },
      },
    },
  },
})
