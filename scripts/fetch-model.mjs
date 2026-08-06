/**
 * Vendors the MediaPipe face landmarker model into public/models/ so the app
 * can run without reaching a CDN at start-up. Optional — the app falls back to
 * the public URL when the local copy is absent.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const target = resolve(root, "public/models/face_landmarker.task");

await mkdir(dirname(target), { recursive: true });

process.stdout.write(`Fetching ${MODEL_URL}\n`);
const response = await fetch(MODEL_URL);
if (!response.ok) {
  process.stderr.write(`Failed: ${response.status} ${response.statusText}\n`);
  process.exit(1);
}

const buffer = Buffer.from(await response.arrayBuffer());
await writeFile(target, buffer);
process.stdout.write(`Wrote ${target} (${(buffer.length / 1e6).toFixed(1)} MB)\n`);
