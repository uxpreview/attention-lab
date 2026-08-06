import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from "@mediapipe/tasks-vision";

/**
 * Thin wrapper around MediaPipe's FaceLandmarker: owns the camera stream and
 * emits landmark results on every animation frame.
 *
 * We need the refined (478-point) mesh because the last ten points are the iris
 * contours, and iris position relative to the eye corners is the entire basis
 * of the gaze signal. The 468-point mesh cannot do this.
 */

const WASM_BASE =
  import.meta.env.VITE_MEDIAPIPE_WASM_BASE ??
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const MODEL_URL =
  import.meta.env.VITE_FACE_MODEL_URL ??
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

/** Local copy written by `npm run fetch-model`, preferred when present. */
const LOCAL_MODEL_URL = "/models/face_landmarker.task";

export interface TrackerFrame {
  result: FaceLandmarkerResult;
  timestamp: number;
  videoWidth: number;
  videoHeight: number;
}

export type FrameListener = (frame: TrackerFrame) => void;

export class FaceTracker {
  readonly video: HTMLVideoElement;

  private landmarker: FaceLandmarker | null = null;
  private stream: MediaStream | null = null;
  private rafId: number | null = null;
  private lastVideoTime = -1;
  private listeners = new Set<FrameListener>();
  private running = false;

  /** Rolling estimate of the delivered frame rate, for the quality readout. */
  private frameTimes: number[] = [];

  constructor() {
    this.video = document.createElement("video");
    this.video.autoplay = true;
    this.video.playsInline = true;
    this.video.muted = true;
    // The preview is mirrored for the participant's benefit, but the landmark
    // coordinates are always in unmirrored video space.
    this.video.style.transform = "scaleX(-1)";
  }

  get isRunning(): boolean {
    return this.running;
  }

  get fps(): number {
    if (this.frameTimes.length < 2) return 0;
    const span = this.frameTimes[this.frameTimes.length - 1] - this.frameTimes[0];
    if (span <= 0) return 0;
    return ((this.frameTimes.length - 1) * 1000) / span;
  }

  async init(onProgress?: (message: string) => void): Promise<void> {
    if (this.landmarker) return;

    onProgress?.("Loading vision runtime…");
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

    onProgress?.("Loading face model…");
    const modelAssetPath = await resolveModelUrl();

    // GPU delegate is dramatically faster, but it is unavailable on some Linux
    // and older Safari configurations, so fall back rather than failing.
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false,
      });
    } catch {
      onProgress?.("GPU unavailable, falling back to CPU…");
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath, delegate: "CPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false,
      });
    }
  }

  async start(onProgress?: (message: string) => void): Promise<void> {
    if (this.running) return;
    await this.init(onProgress);

    onProgress?.("Requesting camera…");
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        facingMode: "user",
      },
      audio: false,
    });

    this.video.srcObject = this.stream;
    await this.video.play();
    await waitForVideoDimensions(this.video);

    this.running = true;
    this.loop();
  }

  stop(): void {
    this.running = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    this.video.srcObject = null;
    this.lastVideoTime = -1;
    this.frameTimes = [];
  }

  onFrame(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private loop = (): void => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this.loop);

    const landmarker = this.landmarker;
    if (!landmarker) return;

    const video = this.video;
    if (video.readyState < 2 || video.videoWidth === 0) return;

    // MediaPipe requires strictly increasing timestamps, and re-detecting on a
    // frame the camera has not advanced past wastes a lot of CPU.
    if (video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;

    const timestamp = performance.now();
    let result: FaceLandmarkerResult;
    try {
      result = landmarker.detectForVideo(video, timestamp);
    } catch {
      return;
    }

    this.frameTimes.push(timestamp);
    if (this.frameTimes.length > 30) this.frameTimes.shift();

    const frame: TrackerFrame = {
      result,
      timestamp,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    };
    for (const listener of this.listeners) listener(frame);
  };
}

async function resolveModelUrl(): Promise<string> {
  // A locally vendored model avoids a cold-start CDN fetch and lets the app run
  // fully offline; fall back to the public model when it has not been fetched.
  try {
    const head = await fetch(LOCAL_MODEL_URL, { method: "HEAD" });
    if (head.ok && (head.headers.get("content-type") ?? "").indexOf("html") === -1) {
      return LOCAL_MODEL_URL;
    }
  } catch {
    /* fall through to the CDN */
  }
  return MODEL_URL;
}

function waitForVideoDimensions(video: HTMLVideoElement): Promise<void> {
  if (video.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve) => {
    const handler = () => {
      video.removeEventListener("loadeddata", handler);
      resolve();
    };
    video.addEventListener("loadeddata", handler);
  });
}
