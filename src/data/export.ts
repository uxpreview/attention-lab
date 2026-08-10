import type { AoiAggregate, AoiResult } from "../analysis/aoi";
import type { Fixation } from "../analysis/fixations";
import { legendFor, OVERLAY_LABELS, type LegendSpec, type OverlayMode } from "../analysis/legend";
import { CANVAS_FONT_FAMILY } from "../analysis/scanpath";
import type { Recording, Study } from "./types";

/** Export helpers. Everything is generated client-side and downloaded directly. */

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "study";
}

function csvCell(value: string | number | null): string {
  if (value === null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function toCsv(rows: Array<Array<string | number | null>>): string {
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

export function exportRawCsv(study: Study, recordings: Recording[]): void {
  const rows: Array<Array<string | number | null>> = [
    ["participant", "t_ms", "x_norm", "y_norm"],
  ];
  for (const recording of recordings) {
    for (const p of recording.points) {
      rows.push([recording.participant, Math.round(p.t), p.x.toFixed(5), p.y.toFixed(5)]);
    }
  }
  download(new Blob([toCsv(rows)], { type: "text/csv" }), `${slug(study.name)}-raw-gaze.csv`);
}

export function exportFixationsCsv(
  study: Study,
  fixationsByParticipant: Array<{ participant: string; fixations: Fixation[] }>
): void {
  const rows: Array<Array<string | number | null>> = [
    ["participant", "index", "start_ms", "duration_ms", "x_norm", "y_norm", "samples"],
  ];
  for (const { participant, fixations } of fixationsByParticipant) {
    fixations.forEach((f, i) => {
      rows.push([
        participant,
        i + 1,
        Math.round(f.start),
        Math.round(f.duration),
        f.x.toFixed(5),
        f.y.toFixed(5),
        f.samples,
      ]);
    });
  }
  download(
    new Blob([toCsv(rows)], { type: "text/csv" }),
    `${slug(study.name)}-fixations.csv`
  );
}

export function exportAoiCsv(
  study: Study,
  aggregates: AoiAggregate[],
  perParticipant: Array<{ participant: string; results: AoiResult[] }>
): void {
  const rows: Array<Array<string | number | null>> = [
    ["scope", "participant", "aoi", "hit_rate", "dwell_ms", "fixations", "ttff_ms", "dwell_share"],
  ];

  for (const a of aggregates) {
    rows.push([
      "study",
      `n=${a.participants}`,
      a.label,
      a.hitRate.toFixed(3),
      Math.round(a.meanDwell),
      a.meanFixationCount.toFixed(2),
      a.meanTimeToFirstFixation === null ? null : Math.round(a.meanTimeToFirstFixation),
      "",
    ]);
  }

  for (const { participant, results } of perParticipant) {
    for (const r of results) {
      rows.push([
        "participant",
        participant,
        r.label,
        r.fixationCount > 0 ? 1 : 0,
        Math.round(r.dwell),
        r.fixationCount,
        r.timeToFirstFixation === null ? null : Math.round(r.timeToFirstFixation),
        r.dwellShare.toFixed(4),
      ]);
    }
  }

  download(new Blob([toCsv(rows)], { type: "text/csv" }), `${slug(study.name)}-aoi.csv`);
}

/**
 * Ground and ink for the caption band, mirroring tokens.css. Canvas cannot
 * read a custom property, so these are the one place in the app where the
 * palette is repeated — kept to four values, all from :root.
 */
const CAPTION_BG = "#fef6e9";
const CAPTION_LINE = "#e5dac6";
const CAPTION_STRONG = "#182528";
const CAPTION_MUTED = "#5f6e73";

export interface OverlayExportContext {
  mode: OverlayMode;
  /** Participant labels in the current selection, in draw order. */
  participants: string[];
  /** "All participants" or the one participant the stage is showing. */
  scope: string;
}

/**
 * Flattens a stimulus and its overlay into a single PNG at the stimulus's
 * native resolution, so the export is not capped by the screen it was viewed on,
 * and captions it.
 *
 * The caption is not decoration. A heatmap pasted into a deck with no study
 * name, no task, no participant count and no colour scale is unciteable — the
 * reader cannot tell what was asked, of how many people, or what red means.
 * Every commercial tool's export is self-describing for exactly this reason,
 * and all of it is available at the call site already.
 */
export async function exportOverlayPng(
  study: Study,
  stimulusImage: HTMLImageElement | null,
  overlay: HTMLCanvasElement,
  context: OverlayExportContext
): Promise<void> {
  const width = stimulusImage?.naturalWidth || overlay.width;
  const height = stimulusImage?.naturalHeight || overlay.height;

  // The caption is sized off the image so it stays legible whether the
  // stimulus is a 900px wireframe or a 3000px retina screenshot.
  const unit = Math.max(11, Math.min(22, width / 62));
  const pad = unit * 1.6;
  const captionHeight = Math.round(unit * 11.5);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height + captionHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  if (stimulusImage) {
    ctx.drawImage(stimulusImage, 0, 0, width, height);
  } else {
    ctx.fillStyle = "#0f1115";
    ctx.fillRect(0, 0, width, height);
  }
  ctx.drawImage(overlay, 0, 0, width, height);

  drawCaption(ctx, study, context, {
    x: 0,
    y: height,
    width,
    height: captionHeight,
    unit,
    pad,
  });

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (blob) download(blob, `${slug(study.name)}-${slug(OVERLAY_LABELS[context.mode])}.png`);
}

interface CaptionBox {
  x: number;
  y: number;
  width: number;
  height: number;
  unit: number;
  pad: number;
}

function drawCaption(
  ctx: CanvasRenderingContext2D,
  study: Study,
  context: OverlayExportContext,
  box: CaptionBox
): void {
  const { unit, pad } = box;
  const spec = legendFor(context.mode, context.participants);

  ctx.fillStyle = CAPTION_BG;
  ctx.fillRect(box.x, box.y, box.width, box.height);
  ctx.fillStyle = CAPTION_LINE;
  ctx.fillRect(box.x, box.y, box.width, Math.max(1, unit * 0.09));

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";

  // Left column: what this is a picture of.
  const left = box.x + pad;
  const columnWidth = box.width * 0.56 - pad;
  let y = box.y + pad + unit;

  ctx.fillStyle = CAPTION_STRONG;
  ctx.font = `700 ${unit * 1.35}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(ellipsise(ctx, study.name, columnWidth), left, y);

  y += unit * 1.7;
  ctx.fillStyle = CAPTION_MUTED;
  ctx.font = `400 ${unit}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(
    ellipsise(ctx, study.task ? `Task: ${study.task}` : "No task set", columnWidth),
    left,
    y
  );

  y += unit * 1.6;
  const n = context.participants.length;
  const meta = [
    OVERLAY_LABELS[context.mode],
    `n=${n} participant${n === 1 ? "" : "s"}`,
    context.scope,
    new Date().toISOString().slice(0, 10),
  ].join("  ·  ");
  ctx.font = `500 ${unit * 0.86}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(ellipsise(ctx, meta.toUpperCase(), columnWidth), left, y);

  // Right column: the same legend the results screen shows.
  drawLegend(ctx, spec, {
    x: box.x + box.width * 0.58,
    y: box.y + pad,
    width: box.width * 0.42 - pad,
    height: box.height - pad * 2,
    unit,
  });
}

function drawLegend(
  ctx: CanvasRenderingContext2D,
  spec: LegendSpec,
  box: { x: number; y: number; width: number; height: number; unit: number }
): void {
  const { unit } = box;
  let y = box.y + unit;

  ctx.fillStyle = CAPTION_MUTED;
  ctx.font = `500 ${unit * 0.86}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(spec.title.toUpperCase(), box.x, y);
  y += unit * 0.8;

  if (spec.stops) {
    const stripHeight = unit * 1.1;
    if (spec.banded) {
      const step = box.width / spec.stops.length;
      spec.stops.forEach((colour, i) => {
        ctx.fillStyle = colour;
        ctx.fillRect(box.x + i * step, y, Math.ceil(step), stripHeight);
      });
    } else {
      const gradient = ctx.createLinearGradient(box.x, 0, box.x + box.width, 0);
      spec.stops.forEach((colour, i) => {
        gradient.addColorStop(i / Math.max(1, spec.stops!.length - 1), colour);
      });
      ctx.fillStyle = gradient;
      ctx.fillRect(box.x, y, box.width, stripHeight);
    }
    ctx.strokeStyle = CAPTION_LINE;
    ctx.lineWidth = 1;
    ctx.strokeRect(box.x, y, box.width, stripHeight);

    y += stripHeight + unit * 1.05;
    ctx.fillStyle = CAPTION_MUTED;
    ctx.font = `400 ${unit * 0.8}px ${CANVAS_FONT_FAMILY}`;
    ctx.fillText(spec.minLabel, box.x, y);
    ctx.textAlign = "right";
    ctx.fillText(spec.maxLabel, box.x + box.width, y);
    ctx.textAlign = "left";
    y += unit * 1.3;
  }

  if (spec.swatches) {
    const chip = unit * 0.72;
    for (const swatch of spec.swatches.slice(0, 6)) {
      ctx.fillStyle = swatch.colour;
      ctx.fillRect(box.x, y - chip, chip, chip);
      ctx.fillStyle = CAPTION_MUTED;
      ctx.font = `400 ${unit * 0.8}px ${CANVAS_FONT_FAMILY}`;
      ctx.fillText(swatch.label, box.x + chip * 1.6, y);
      y += chip * 1.6;
    }
    y += unit * 0.3;
  }

  ctx.fillStyle = CAPTION_MUTED;
  ctx.font = `400 ${unit * 0.78}px ${CANVAS_FONT_FAMILY}`;
  for (const line of wrap(ctx, spec.note, box.width).slice(0, 3)) {
    if (y > box.y + box.height) return;
    ctx.fillText(line, box.x, y);
    y += unit * 1.15;
  }
}

/** Trims a single line to fit, with an ellipsis rather than an overflow. */
function ellipsise(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) {
    cut = cut.slice(0, -1);
  }
  return `${cut.trimEnd()}…`;
}

function wrap(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/)) {
    const candidate = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

export function exportStudyJson(study: Study, recordings: Recording[]): void {
  const payload = {
    study: {
      id: study.id,
      name: study.name,
      task: study.task,
      duration: study.duration,
      createdAt: study.createdAt,
      aois: study.aois,
      stimulus:
        study.stimulus.kind === "image"
          ? { kind: "image", name: study.stimulus.name, width: study.stimulus.width, height: study.stimulus.height }
          : study.stimulus,
    },
    recordings: recordings.map((r) => ({
      id: r.id,
      participant: r.participant,
      createdAt: r.createdAt,
      quality: r.quality,
      points: r.points.map((p) => ({
        t: Math.round(p.t),
        x: Number(p.x.toFixed(5)),
        y: Number(p.y.toFixed(5)),
      })),
    })),
  };

  download(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `${slug(study.name)}-session.json`
  );
}
