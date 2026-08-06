import { defineConfig } from "vite";

export default defineConfig({
  server: {
    // getUserMedia requires a secure context. localhost counts as secure, so
    // plain http is fine for local dev; use `--host` + https for LAN testing.
    port: 5173,
    host: true,
  },
  build: {
    target: "es2022",
    sourcemap: true,
  },
});
