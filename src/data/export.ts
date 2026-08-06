import type { AoiAggregate, AoiResult } from "../analysis/aoi";
import type { Fixation } from "../analysis/fixations";
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
 * Flattens a stimulus and its overlay into a single PNG at the stimulus's
 * native resolution, so the export is not capped by the screen it was viewed on.
 */
export async function exportOverlayPng(
  study: Study,
  stimulusImage: HTMLImageElement | null,
  overlay: HTMLCanvasElement,
  label: string
): Promise<void> {
  const width = stimulusImage?.naturalWidth || overlay.width;
  const height = stimulusImage?.naturalHeight || overlay.height;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  if (stimulusImage) {
    ctx.drawImage(stimulusImage, 0, 0, width, height);
  } else {
    ctx.fillStyle = "#0f1115";
    ctx.fillRect(0, 0, width, height);
  }
  ctx.drawImage(overlay, 0, 0, width, height);

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (blob) download(blob, `${slug(study.name)}-${slug(label)}.png`);
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
