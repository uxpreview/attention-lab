import type { Fixation } from "./fixations";

/**
 * Scanpath rendering: numbered fixation circles joined by saccade lines, sized
 * by dwell. Answers "in what order did they read this", which a heatmap cannot.
 */

export interface ScanpathOptions {
  /** Radius in pixels for the shortest fixation. */
  minRadius?: number;
  /** Radius in pixels for the longest fixation. */
  maxRadius?: number;
  showNumbers?: boolean;
  /** Draw a colour gradient over time (early = cool, late = warm). */
  colourByTime?: boolean;
}

export function renderScanpath(
  canvas: HTMLCanvasElement,
  fixations: Fixation[],
  options: ScanpathOptions = {}
): void {
  const minRadius = options.minRadius ?? 10;
  const maxRadius = options.maxRadius ?? 46;
  const showNumbers = options.showNumbers ?? true;
  const colourByTime = options.colourByTime ?? true;

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);
  if (fixations.length === 0) return;

  let maxDuration = 0;
  for (const f of fixations) maxDuration = Math.max(maxDuration, f.duration);
  if (maxDuration <= 0) maxDuration = 1;

  const px = (f: Fixation) => ({ x: f.x * width, y: f.y * height });

  // Saccades first, so the fixation circles sit on top of the connecting lines.
  // Drawn twice: a light halo underneath a dark line, so the path stays legible
  // over both pale wireframes and dark screenshots.
  const path = new Path2D();
  for (let i = 0; i < fixations.length; i++) {
    const p = px(fixations[i]);
    if (i === 0) path.moveTo(p.x, p.y);
    else path.lineTo(p.x, p.y);
  }
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.stroke(path);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(20,26,38,0.8)";
  ctx.stroke(path);

  for (let i = 0; i < fixations.length; i++) {
    const f = fixations[i];
    const p = px(f);
    // Area, not radius, scales with duration — otherwise long fixations read as
    // far more dominant than they are.
    const t = Math.sqrt(f.duration / maxDuration);
    const radius = minRadius + (maxRadius - minRadius) * t;

    const hue = colourByTime ? 210 - (i / Math.max(1, fixations.length - 1)) * 210 : 200;
    ctx.fillStyle = `hsla(${hue}, 85%, 55%, 0.55)`;
    ctx.strokeStyle = `hsla(${hue}, 85%, 40%, 0.95)`;
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (showNumbers) {
      const fontSize = Math.max(11, Math.min(20, radius * 0.85));
      ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.strokeText(String(i + 1), p.x, p.y);
      ctx.fillStyle = "#fff";
      ctx.fillText(String(i + 1), p.x, p.y);
    }
  }
}
