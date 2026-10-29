import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// We serve the built bundle from the LangGraph agent server at `/app/`.
// Use a relative base so all asset URLs are sub-path safe.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
    proxy: {
      // In `npm run dev`, forward agent-server endpoints to `langgraph dev`
      // on :2024 so the frontend can talk to the local agent.
      "/threads": "http://localhost:2024",
      "/runs": "http://localhost:2024",
      "/assistants": "http://localhost:2024",
      "/info": "http://localhost:2024",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
