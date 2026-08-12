import type { GazeEngine } from "../tracker/gaze";
import { el } from "./dom";

/**
 * Pre-calibration setup check: live feedback on the things that actually sink
 * a calibration — distance, framing, and lighting — before the participant
 * invests thirteen dots' worth of effort in one.
 *
 * The assessment is pure and testable; the panel is a thin loop around it that
 * polls the tracker a few times a second and re-renders when anything changes.
 */

export interface SetupReading {
  faceVisible: boolean;
  /** Outer-corner interocular distance as a fraction of the frame width. */
  interocular: number | null;
  headX: number | null;
  headY: number | null;
  fps: number;
  /** Mean luminance (0-255) of the face region, or null before first sample. */
  faceLuma: number | null;
  /** Mean luminance (0-255) of the whole frame. */
  frameLuma: number | null;
}

export type CheckState = "ok" | "warn" | "fail" | "unknown";

export interface SetupCheck {
  id: "face" | "distance" | "centering" | "lighting" | "fps";
  label: string;
  state: CheckState;
  hint: string;
}

/**
 * Distance thresholds are in interocular fraction of frame width. With a
 * typical ~65° horizontal webcam FOV, outer eye corners (~9cm apart) subtend
 * roughly 0.08 of the frame at 90cm and 0.19 at 40cm — the useful range
 * brackets "about an arm's length".
 */
const TOO_FAR = 0.075;
const TOO_CLOSE = 0.19;

export function assessSetup(r: SetupReading): SetupCheck[] {
  const checks: SetupCheck[] = [];

  checks.push(
    r.faceVisible
      ? { id: "face", label: "Face detected", state: "ok", hint: "" }
      : {
          id: "face",
          label: "No face detected",
          state: "fail",
          hint: "Face the camera straight on, with light coming from in front of you.",
        }
  );

  if (!r.faceVisible || r.interocular === null) {
    checks.push({ id: "distance", label: "Distance", state: "unknown", hint: "" });
  } else if (r.interocular < TOO_FAR) {
    checks.push({
      id: "distance",
      label: "Too far away",
      state: "warn",
      hint: "Move closer — about an arm's length from the screen.",
    });
  } else if (r.interocular > TOO_CLOSE) {
    checks.push({
      id: "distance",
      label: "Too close",
      state: "warn",
      hint: "Back up a little — about an arm's length works best.",
    });
  } else {
    checks.push({ id: "distance", label: "Good distance", state: "ok", hint: "" });
  }

  if (!r.faceVisible || r.headX === null || r.headY === null) {
    checks.push({ id: "centering", label: "Framing", state: "unknown", hint: "" });
  } else if (Math.abs(r.headX - 0.5) > 0.22 || Math.abs(r.headY - 0.5) > 0.25) {
    checks.push({
      id: "centering",
      label: "Off-centre",
      state: "warn",
      hint: "Shift so your face sits in the middle of the preview.",
    });
  } else {
    checks.push({ id: "centering", label: "Centred in frame", state: "ok", hint: "" });
  }

  if (r.faceLuma === null || r.frameLuma === null) {
    checks.push({ id: "lighting", label: "Lighting", state: "unknown", hint: "" });
  } else if (r.faceLuma < 60) {
    checks.push({
      id: "lighting",
      label: "Face too dark",
      state: "fail",
      hint: "Add light in front of you — a lamp behind the screen works well.",
    });
  } else if (r.frameLuma - r.faceLuma > 45) {
    checks.push({
      id: "lighting",
      label: "Backlit",
      state: "warn",
      hint: "The background is brighter than your face — avoid windows behind you.",
    });
  } else if (r.faceLuma > 235) {
    checks.push({
      id: "lighting",
      label: "Overexposed",
      state: "warn",
      hint: "Your face is washed out — dim the light or move it further away.",
    });
  } else {
    checks.push({ id: "lighting", label: "Good lighting", state: "ok", hint: "" });
  }

  if (r.fps <= 0) {
    checks.push({ id: "fps", label: "Frame rate", state: "unknown", hint: "" });
  } else if (r.fps < 15) {
    checks.push({
      id: "fps",
      label: `Low frame rate (${Math.round(r.fps)} fps)`,
      state: "warn",
      hint: "Close other tabs or apps using the camera or CPU.",
    });
  } else {
    checks.push({ id: "fps", label: `Tracking at ${Math.round(r.fps)} fps`, state: "ok", hint: "" });
  }

  return checks;
}

export function isReady(checks: SetupCheck[]): boolean {
  return checks.every((c) => c.state === "ok");
}

/**
 * Samples face and whole-frame luminance from the live video at a small
 * resolution. Runs on demand rather than per frame — lighting changes on the
 * order of seconds, not milliseconds.
 */
class LumaSampler {
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d", { willReadFrequently: true });

  sample(
    video: HTMLVideoElement,
    face: { headX: number; headY: number; interocular: number } | null
  ): { faceLuma: number | null; frameLuma: number | null } {
    if (!this.ctx || video.videoWidth === 0) return { faceLuma: null, frameLuma: null };

    const w = 96;
    const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx.drawImage(video, 0, 0, w, h);
    const data = this.ctx.getImageData(0, 0, w, h).data;

    let frameSum = 0;
    let faceSum = 0;
    let faceCount = 0;

    // Face box around the nose, sized from the interocular distance. Landmark
    // coordinates are unmirrored video space, matching drawImage output.
    const box = face
      ? {
          x0: Math.max(0, Math.floor((face.headX - face.interocular * 1.1) * w)),
          x1: Math.min(w, Math.ceil((face.headX + face.interocular * 1.1) * w)),
          y0: Math.max(0, Math.floor((face.headY - face.interocular * 1.5) * h)),
          y1: Math.min(h, Math.ceil((face.headY + face.interocular * 1.5) * h)),
        }
      : null;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        frameSum += luma;
        if (box && x >= box.x0 && x < box.x1 && y >= box.y0 && y < box.y1) {
          faceSum += luma;
          faceCount++;
        }
      }
    }

    return {
      faceLuma: faceCount > 0 ? faceSum / faceCount : null,
      frameLuma: frameSum / (w * h),
    };
  }
}

/**
 * Mounts the live checklist into `host` and reports readiness changes.
 * Returns a dispose function; also self-disposes if the host leaves the DOM.
 */
export function mountSetupPanel(
  engine: GazeEngine,
  host: HTMLElement,
  onReady: (ready: boolean) => void
): () => void {
  const list = el("ul", { class: "setup-checks" });
  host.append(list);

  const sampler = new LumaSampler();
  let faceLuma: number | null = null;
  let frameLuma: number | null = null;
  let lastLumaAt = 0;
  let lastRendered = "";
  let lastReady: boolean | null = null;

  const tick = (): void => {
    if (!host.isConnected) {
      dispose();
      return;
    }

    const face = engine.currentFace;
    const now = performance.now();
    if (now - lastLumaAt > 700) {
      lastLumaAt = now;
      const luma = sampler.sample(
        engine.tracker.video,
        face ? { headX: face.headX, headY: face.headY, interocular: face.interocular } : null
      );
      faceLuma = luma.faceLuma;
      frameLuma = luma.frameLuma;
    }

    const checks = assessSetup({
      faceVisible: face !== null,
      interocular: face?.interocular ?? null,
      headX: face?.headX ?? null,
      headY: face?.headY ?? null,
      fps: engine.tracker.fps,
      faceLuma,
      frameLuma,
    });

    const key = checks.map((c) => `${c.state}:${c.label}`).join("|");
    if (key !== lastRendered) {
      lastRendered = key;
      render(checks);
    }

    const ready = isReady(checks);
    if (ready !== lastReady) {
      lastReady = ready;
      onReady(ready);
    }
  };

  const render = (checks: SetupCheck[]): void => {
    list.replaceChildren(
      ...checks.map((c) =>
        el(
          "li",
          { class: `setup-check is-${c.state}` },
          el("span", { class: "setup-dot" }),
          el(
            "div",
            {},
            el("span", { class: "setup-label" }, c.label),
            c.hint ? el("span", { class: "setup-hint" }, c.hint) : ""
          )
        )
      )
    );
  };

  const interval = window.setInterval(tick, 250);
  const dispose = (): void => {
    window.clearInterval(interval);
    list.remove();
  };
  tick();
  return dispose;
}
