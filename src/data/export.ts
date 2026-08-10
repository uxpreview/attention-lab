import type { Aoi, AoiAggregate, AoiResult } from "../analysis/aoi";
import type { Fixation } from "../analysis/fixations";
import {
  legendFor,
  OVERLAY_LABELS,
  type LegendScale,
  type LegendSpec,
  type OverlayMode,
} from "../analysis/legend";
import { isLowSignal } from "../analysis/quality";
import { CANVAS_FONT_FAMILY } from "../analysis/scanpath";
import type { Recording, Study } from "./types";

/** Export helpers. Everything is generated client-side and downloaded directly. */

/**
 * Which recordings a file covers, and why.
 *
 * An exported CSV outlives the screen it came from, and the results screen goes
 * to real trouble to exclude low-signal sessions — so a file generated from
 * that screen has to carry the exclusion with it. Without this the same study
 * reported 75% on screen and 60% in the file, and nothing in the file hinted
 * that the two were counting different sets of people.
 */
export interface ExportScope {
  /** One sentence, written into the file above the column header row. */
  note: string;
}

/** A recording and what was measured from it, as the CSV writers want it. */
export interface ExportRow<T> {
  recording: Recording;
  data: T;
}

/** The scope line plus a blank row, ahead of the real header. Spreadsheets
 * read it as a two-cell first row; a human reads it as provenance. */
function scopeRows(scope: ExportScope): Array<Array<string | number | null>> {
  return [["scope_note", scope.note], []];
}

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

export function exportRawCsv(study: Study, recordings: Recording[], scope: ExportScope): void {
  const rows: Array<Array<string | number | null>> = [
    ...scopeRows(scope),
    ["participant", "low_signal", "t_ms", "x_norm", "y_norm"],
  ];
  for (const recording of recordings) {
    const low = isLowSignal(recording.quality) ? "true" : "false";
    for (const p of recording.points) {
      rows.push([recording.participant, low, Math.round(p.t), p.x.toFixed(5), p.y.toFixed(5)]);
    }
  }
  download(new Blob([toCsv(rows)], { type: "text/csv" }), `${slug(study.name)}-raw-gaze.csv`);
}

export function exportFixationsCsv(
  study: Study,
  fixationsByParticipant: Array<ExportRow<Fixation[]>>,
  scope: ExportScope
): void {
  const rows: Array<Array<string | number | null>> = [
    ...scopeRows(scope),
    [
      "participant",
      "low_signal",
      "index",
      "start_ms",
      "duration_ms",
      "x_norm",
      "y_norm",
      "samples",
    ],
  ];
  for (const { recording, data: fixations } of fixationsByParticipant) {
    const low = isLowSignal(recording.quality) ? "true" : "false";
    fixations.forEach((f, i) => {
      rows.push([
        recording.participant,
        low,
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
  perParticipant: Array<ExportRow<AoiResult[]>>,
  scope: ExportScope
): void {
  const rows: Array<Array<string | number | null>> = [
    ...scopeRows(scope),
    [
      "scope",
      "participant",
      "low_signal",
      "aoi",
      "hit_rate",
      "dwell_ms",
      "fixations",
      "ttff_ms",
      "dwell_share",
    ],
  ];

  for (const a of aggregates) {
    rows.push([
      "study",
      `n=${a.participants}`,
      // The aggregate row is a roll-up of whatever the scope line describes,
      // so it is not itself one recording's grade.
      "",
      a.label,
      a.hitRate.toFixed(3),
      Math.round(a.meanDwell),
      a.meanFixationCount.toFixed(2),
      a.meanTimeToFirstFixation === null ? null : Math.round(a.meanTimeToFirstFixation),
      "",
    ]);
  }

  for (const { recording, data: results } of perParticipant) {
    const low = isLowSignal(recording.quality) ? "true" : "false";
    for (const r of results) {
      rows.push([
        "participant",
        recording.participant,
        low,
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
/** --accent. Region ink, on the image rather than in the caption. */
const CAPTION_ACCENT = "#003f48";

export interface OverlayExportContext {
  mode: OverlayMode;
  /** Participant labels in the current selection, in draw order. */
  participants: string[];
  /** "All participants" or the one participant the stage is showing. */
  scope: string;
  /**
   * The study's regions, and whether the screen was showing them.
   *
   * The region layer on screen is DOM, so it never reached this canvas: a
   * heatmap exported from a study with five named regions came out as a clean
   * heat wash with no boxes and no names, which cannot support the one sentence
   * it is pasted into a deck to support. Drawn here from the same normalised
   * rects, gated on the screen's own "Show regions" switch.
   */
  aois: Aoi[];
  showAois: boolean;
  /**
   * What the hot end of the colour scale was worth on screen, when the overlay
   * has a unit. The exported caption carries the same numbered axis the results
   * screen shows — a figure pasted into a deck is the one place the reader
   * cannot ask what dark red means.
   */
  scale?: LegendScale | null;
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
  // One line taller than it was, for the numbered axis under the scale strip.
  // The note below it is capped at MAX_NOTE_LINES and would otherwise be the
  // thing that got squeezed — and the note is the caveat that stops the figure
  // being cited wrongly.
  const captionHeight = Math.round(unit * 12.6);

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

  if (context.showAois) drawAois(ctx, context.aois, width, height, unit);

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
  const spec = legendFor(context.mode, context.participants, context.scale ?? null);

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
  // The date gets its own line rather than the tail of a joined string. On a
  // 1440px stimulus the uppercased meta line overflowed its column and the
  // ellipsis ate exactly the provenance a research artifact needs — the export
  // read "… ALL PARTICIPANTS, LOW-SIGNAL EXCLUDED · 20…". The band has the
  // vertical room; the line did not have the horizontal room.
  const meta = [
    OVERLAY_LABELS[context.mode],
    `n=${n} participant${n === 1 ? "" : "s"}`,
    context.scope,
  ].join("  ·  ");
  ctx.font = `500 ${unit * 0.86}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(ellipsise(ctx, meta.toUpperCase(), columnWidth), left, y);

  y += unit * 1.25;
  ctx.font = `400 ${unit * 0.8}px ${CANVAS_FONT_FAMILY}`;
  ctx.fillText(`Exported ${new Date().toISOString().slice(0, 10)}`, left, y);

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

    y += stripHeight;

    // The numbered axis, when the overlay has a unit — the same ticks the
    // results screen draws under the same strip. Ends are pulled inside the
    // strip; anything in between is centred on the fraction it names.
    if (spec.ticks) {
      y += unit * 0.95;
      ctx.fillStyle = CAPTION_MUTED;
      ctx.font = `500 ${unit * 0.78}px ${CANVAS_FONT_FAMILY}`;
      for (const tick of spec.ticks) {
        ctx.textAlign = tick.at <= 0 ? "left" : tick.at >= 1 ? "right" : "center";
        ctx.fillText(tick.label, box.x + box.width * tick.at, y);
      }
      ctx.textAlign = "left";
    }

    y += unit * 1.05;
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
  // A caption that stops mid-sentence ("…are left clear rather than") reads as
  // a broken file rather than a trimmed one. Whatever the last drawn line is,
  // it says so.
  const lines = wrap(ctx, spec.note, box.width);
  const shown = lines.slice(0, MAX_NOTE_LINES);
  const truncated = lines.length > shown.length;
  shown.forEach((line, i) => {
    if (y > box.y + box.height) return;
    const last = i === shown.length - 1;
    ctx.fillText(truncated && last ? ellipsise(ctx, `${line}…`, box.width) : line, box.x, y);
    y += unit * 1.15;
  });
}

/** How many lines of the legend note fit beside the scale. */
const MAX_NOTE_LINES = 3;

/**
 * The region layer, stroked onto the exported image.
 *
 * Same recipe as `.aoi-box` on screen: a white outer halo under a 1.5px accent
 * stroke, so the box survives a pale wireframe and a dark screenshot alike, and
 * the same numbered badge that keys the Areas of interest table — a reader with
 * the PNG and the CSV can match "3" in one to "3" in the other.
 */
function drawAois(
  ctx: CanvasRenderingContext2D,
  aois: Aoi[],
  width: number,
  height: number,
  unit: number
): void {
  if (aois.length === 0) return;
  const stroke = Math.max(1.5, unit * 0.13);
  const font = Math.max(11, unit * 0.85);
  const badge = font * 1.62;

  ctx.save();
  ctx.textBaseline = "middle";

  aois.forEach((aoi, i) => {
    const x = aoi.x * width;
    const y = aoi.y * height;
    const w = aoi.width * width;
    const h = aoi.height * height;

    ctx.lineWidth = stroke * 2.4;
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = stroke;
    ctx.strokeStyle = CAPTION_ACCENT;
    ctx.strokeRect(x, y, w, h);

    // Name chip, above the box where there is room and inside its top edge
    // where there is not.
    ctx.font = `500 ${font}px ${CANVAS_FONT_FAMILY}`;
    const chipHeight = font * 1.7;
    const chipWidth = Math.min(width, ctx.measureText(aoi.label).width + font * 1.1);
    const above = y - badge * 0.5 - chipHeight - font * 0.2;
    const chipY = above >= 0 ? above : y + badge * 0.5 + font * 0.2;
    // Pulled back onto the canvas for a region against the right edge, rather
    // than run off it.
    const chipX = Math.min(Math.max(0, x + badge * 0.7), Math.max(0, width - chipWidth));
    ctx.fillStyle = CAPTION_ACCENT;
    roundedRect(ctx, chipX, chipY, chipWidth, chipHeight, font * 0.3);
    ctx.fill();
    ctx.fillStyle = CAPTION_BG;
    ctx.textAlign = "left";
    ctx.fillText(aoi.label, chipX + font * 0.55, chipY + chipHeight / 2);

    // Ordinal badge, straddling the top-left corner so it never covers the
    // heat it is annotating.
    const ordinal = String(i + 1);
    ctx.font = `700 ${font * 0.86}px ${CANVAS_FONT_FAMILY}`;
    const badgeWidth = Math.max(badge, ctx.measureText(ordinal).width + font * 0.8);
    const bx = Math.max(0, x - badge * 0.5);
    const by = Math.max(0, y - badge * 0.5);
    ctx.fillStyle = CAPTION_BG;
    roundedRect(ctx, bx - stroke, by - stroke, badgeWidth + stroke * 2, badge + stroke * 2, badge * 0.5);
    ctx.fill();
    ctx.fillStyle = CAPTION_ACCENT;
    roundedRect(ctx, bx, by, badgeWidth, badge, badge * 0.5);
    ctx.fill();
    ctx.fillStyle = CAPTION_BG;
    ctx.textAlign = "center";
    ctx.fillText(ordinal, bx + badgeWidth / 2, by + badge / 2);
  });

  ctx.restore();
}

/** A rounded rectangle path. `roundRect` is not universal enough to lean on
 * for the one artifact that leaves the tool. */
function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
): void {
  const radius = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
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
