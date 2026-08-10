import { aggregateAois, analyseAois, type Aoi, type AoiResult } from "../analysis/aoi";
import { detectFixations, summarise, type Fixation } from "../analysis/fixations";
import { renderHeatmap, type HeatmapStyle, type HeatPoint } from "../analysis/heatmap";
import { renderScanpath } from "../analysis/scanpath";
import {
  exportAoiCsv,
  exportFixationsCsv,
  exportOverlayPng,
  exportRawCsv,
  exportStudyJson,
} from "../data/export";
import { deleteRecording, listRecordings, newId, saveStudy } from "../data/store";
import type { Recording, Study } from "../data/types";
import { clear, confirmButton, el, formatMs, formatPercent, nextFrame } from "./dom";

type ViewMode = HeatmapStyle | "scanpath" | "raw";

const VIEW_MODES: Array<[ViewMode, string]> = [
  ["heat", "Heatmap"],
  ["spotlight", "Spotlight"],
  ["contour", "Contours"],
  ["scanpath", "Scanpath"],
  ["raw", "Raw gaze"],
];

interface AnalysedRecording {
  recording: Recording;
  fixations: Fixation[];
  aoiResults: AoiResult[];
}

/**
 * Results view: stimulus with an analysis overlay, plus the numbers behind it.
 *
 * Fixation detection runs per recording using that recording's own on-screen
 * stimulus size. Thresholds like "45 pixels of dispersion" only mean anything
 * relative to the display the participant actually looked at, so normalising
 * first and thresholding second would silently change the definition of a
 * fixation depending on the reviewer's monitor.
 */
export async function renderResults(
  host: HTMLElement,
  study: Study,
  onBack: () => void
): Promise<void> {
  clear(host);

  const recordings = await listRecordings(study.id);
  let aois: Aoi[] = [...study.aois];
  let mode: ViewMode = "heat";
  let selected: string | "all" = "all";
  let drawingAoi = false;

  const stage = el("div", { class: "results-stage" });
  const overlay = el("canvas", { class: "results-overlay", role: "img" });
  const aoiLayer = el("div", { class: "aoi-layer" });
  let stimulusImage: HTMLImageElement | null = null;
  let objectUrl: string | null = null;

  if (study.stimulus.kind === "image") {
    objectUrl = URL.createObjectURL(study.stimulus.blob);
    stimulusImage = el("img", { class: "results-image", src: objectUrl, alt: "" });
    stage.append(stimulusImage);
  } else {
    stage.append(
      el(
        "div",
        { class: "results-placeholder" },
        el("p", {}, "Live page stimulus"),
        el("code", {}, study.stimulus.url),
        el(
          "p",
          { class: "muted" },
          "Overlays are drawn against the recorded viewport, not a re-render of the page."
        )
      )
    );
  }
  stage.append(overlay, aoiLayer);

  const sidebar = el("aside", { class: "results-sidebar" });
  const layout = el(
    "div",
    { class: "results-layout" },
    el("div", { class: "results-main" }, stage),
    sidebar
  );

  const header = el(
    "header",
    { class: "results-header" },
    el("button", { class: "btn btn-ghost", type: "button", onclick: onBack }, "← Studies"),
    el("h1", {}, study.name),
    el("span", { class: "pill" }, `${recordings.length} recording${recordings.length === 1 ? "" : "s"}`)
  );

  // The results screen sits on the same measure as the experiment page around
  // it; .container is the site's shell.
  host.append(el("div", { class: "container screen" }, header, layout));

  if (recordings.length === 0) {
    sidebar.append(
      el(
        "div",
        { class: "empty" },
        el("p", {}, "No recordings yet."),
        el("p", { class: "muted" }, "Run a session from the study list to collect data.")
      )
    );
    // The stage is the primary surface, so it explains its own emptiness
    // rather than leaving that to the sidebar.
    stage.append(
      el(
        "div",
        { class: "stage-empty" },
        el("p", {}, "No recordings yet — run a session to see attention here.")
      )
    );
    return;
  }

  if (stimulusImage && !stimulusImage.complete) {
    const img = stimulusImage;
    await new Promise<void>((resolve) => img.addEventListener("load", () => resolve(), { once: true }));
  }

  const fixationsOf = (recording: Recording): Fixation[] => {
    const rect = recording.quality.stimulusRect;
    const width = rect.width || recording.quality.viewportWidth || 1280;
    const height = rect.height || recording.quality.viewportHeight || 720;

    // Detect in the pixel space the participant actually saw, then normalise.
    const pxPoints = recording.points.map((p) => ({ x: p.x * width, y: p.y * height, t: p.t }));
    return detectFixations(pxPoints, {
      dispersion: Math.max(30, Math.min(width, height) * 0.045),
      minDuration: 100,
    }).map((f) => ({ ...f, x: f.x / width, y: f.y / height }));
  };

  // Fixation detection is the expensive pass and depends only on the
  // recording, so it runs once per recording and is cached. AOI edits — a new
  // box, a rename, a delete — re-run only the cheap analyseAois pass over
  // these cached fixations.
  const fixationsById = new Map(recordings.map((r) => [r.id, fixationsOf(r)] as const));

  const analyse = (): AnalysedRecording[] =>
    recordings.map((recording) => {
      const fixations = fixationsById.get(recording.id) ?? [];
      return {
        recording,
        fixations,
        aoiResults: analyseAois(aois, fixations, 0),
      };
    });

  let analysed = analyse();

  const activeSet = (): AnalysedRecording[] =>
    selected === "all" ? analysed : analysed.filter((a) => a.recording.id === selected);

  const draw = async (): Promise<void> => {
    await nextFrame();
    const rect = (stimulusImage ?? stage).getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    overlay.style.left = `${rect.left - stageRect.left}px`;
    overlay.style.top = `${rect.top - stageRect.top}px`;
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));

    aoiLayer.style.left = overlay.style.left;
    aoiLayer.style.top = overlay.style.top;
    aoiLayer.style.width = overlay.style.width;
    aoiLayer.style.height = overlay.style.height;

    const modeLabel = VIEW_MODES.find(([value]) => value === mode)?.[1] ?? mode;
    overlay.setAttribute("aria-label", `${modeLabel} overlay`);

    const set = activeSet();

    if (mode === "scanpath") {
      // A combined scanpath across participants would be meaningless, so show
      // the first selected participant's path and say so.
      renderScanpath(overlay, set[0]?.fixations ?? [], { minRadius: 10 * dpr, maxRadius: 46 * dpr });
    } else if (mode === "raw") {
      renderRawPoints(overlay, set.map((a) => a.recording));
    } else {
      const points: HeatPoint[] = [];
      for (const a of set) {
        for (const f of a.fixations) points.push({ x: f.x, y: f.y, weight: f.duration });
      }
      renderHeatmap(overlay, points, { style: mode, radiusRatio: 0.055 });
    }

    renderAoiBoxes(aoiLayer, aois, onAoiChange);
  };

  const persistAois = (): void => {
    study.aois = aois;
    void saveStudy(study);
  };

  const onAoiChange = (): void => {
    persistAois();
    analysed = analyse();
    void draw();
    renderSidebar();
  };

  // --- AOI drawing -------------------------------------------------------

  aoiLayer.addEventListener("pointerdown", (event: PointerEvent) => {
    if (!drawingAoi) return;
    event.preventDefault();

    // Capture the pointer, or a finger that slides off the stage mid-drag
    // stops sending moves and the region is never finished.
    aoiLayer.setPointerCapture(event.pointerId);

    const bounds = aoiLayer.getBoundingClientRect();
    const startX = (event.clientX - bounds.left) / bounds.width;
    const startY = (event.clientY - bounds.top) / bounds.height;
    const ghost = el("div", { class: "aoi-box is-drawing" });
    aoiLayer.append(ghost);

    const onMove = (e: PointerEvent) => {
      const x = (e.clientX - bounds.left) / bounds.width;
      const y = (e.clientY - bounds.top) / bounds.height;
      ghost.style.left = `${Math.min(startX, x) * 100}%`;
      ghost.style.top = `${Math.min(startY, y) * 100}%`;
      ghost.style.width = `${Math.abs(x - startX) * 100}%`;
      ghost.style.height = `${Math.abs(y - startY) * 100}%`;
    };

    const onUp = (e: PointerEvent) => {
      aoiLayer.removeEventListener("pointermove", onMove);
      aoiLayer.removeEventListener("pointerup", onUp);
      aoiLayer.removeEventListener("pointercancel", onUp);
      ghost.remove();

      const x = (e.clientX - bounds.left) / bounds.width;
      const y = (e.clientY - bounds.top) / bounds.height;
      const width = Math.abs(x - startX);
      const height = Math.abs(y - startY);
      if (width < 0.01 || height < 0.01) return;

      aois.push({
        id: newId("aoi"),
        label: `Region ${aois.length + 1}`,
        x: Math.min(startX, x),
        y: Math.min(startY, y),
        width,
        height,
      });
      drawingAoi = false;
      aoiLayer.classList.remove("is-drawing");
      onAoiChange();
    };

    aoiLayer.addEventListener("pointermove", onMove);
    aoiLayer.addEventListener("pointerup", onUp);
    // A phone call, a notification, or the system taking over the gesture all
    // cancel the pointer. Treat that as finishing the region rather than
    // leaving a ghost box on the stage forever.
    aoiLayer.addEventListener("pointercancel", onUp);
  });

  // --- Sidebar -----------------------------------------------------------

  function renderSidebar(): void {
    // Rebuilding destroys whatever control held keyboard focus, which would
    // drop focus to <body> and force a Tab journey from the top after every
    // interaction. Controls carry a data-key so focus can be handed back to
    // the rebuilt equivalent.
    const active = document.activeElement as HTMLElement | null;
    const focusKey = active && sidebar.contains(active) ? active.dataset.key : undefined;

    clear(sidebar);

    sidebar.append(
      el("h3", {}, "View"),
      el(
        "div",
        { class: "segmented", role: "group", "aria-label": "View mode" },
        ...VIEW_MODES.map(([value, label]) =>
          el(
            "button",
            {
              class: `seg ${mode === value ? "is-active" : ""}`,
              type: "button",
              "aria-pressed": mode === value ? "true" : "false",
              "data-key": `view-${value}`,
              onclick: () => {
                mode = value;
                void draw();
                renderSidebar();
              },
            },
            label
          )
        )
      )
    );

    const participantSelect = el(
      "select",
      {
        class: "input",
        id: "participant-filter",
        "data-key": "participants",
        onchange: (event: Event) => {
          selected = (event.target as HTMLSelectElement).value;
          void draw();
          renderSidebar();
        },
      },
      el("option", { value: "all" }, `All participants (${recordings.length})`),
      ...analysed.map((a) =>
        el(
          "option",
          { value: a.recording.id, ...(selected === a.recording.id ? { selected: true } : {}) },
          a.recording.participant
        )
      )
    );
    participantSelect.value = selected;

    sidebar.append(
      el("h3", {}, el("label", { for: "participant-filter" }, "Participants")),
      participantSelect
    );

    if (mode === "scanpath" && selected === "all" && analysed.length > 1) {
      sidebar.append(
        el(
          "p",
          { class: "note" },
          `Scanpaths are per-person. Showing ${analysed[0].recording.participant}. Pick a participant to see others.`
        )
      );
    }

    // Quality + summary stats
    const set = activeSet();
    const allFixations = set.flatMap((a) => a.fixations);
    const stats = summarise(allFixations, 0);
    const meanValidation = averageOf(
      set.map((a) => a.recording.quality.validationError).filter((v): v is number => v !== null)
    );

    sidebar.append(
      el("h3", {}, "Summary"),
      statRow("Fixations", String(stats.fixationCount)),
      statRow("Mean fixation", formatMs(stats.meanFixationDuration)),
      statRow("Time to first fixation", formatMs(stats.timeToFirstFixation)),
      statRow(
        "Tracking ratio",
        formatPercent(averageOf(set.map((a) => a.recording.quality.trackingRatio)) ?? 0)
      ),
      statRow(
        "Calibration error",
        meanValidation === null ? "—" : `${Math.round(meanValidation)}px`
      )
    );

    // AOIs
    const aoiHeader = el(
      "div",
      { class: "row-between" },
      el("h3", {}, "Areas of interest"),
      el(
        "button",
        {
          class: `btn btn-small ${drawingAoi ? "is-active" : ""}`,
          type: "button",
          "aria-pressed": drawingAoi ? "true" : "false",
          "data-key": "aoi-draw",
          onclick: () => {
            drawingAoi = !drawingAoi;
            aoiLayer.classList.toggle("is-drawing", drawingAoi);
            renderSidebar();
          },
        },
        drawingAoi ? "Cancel" : "+ Draw"
      )
    );
    sidebar.append(aoiHeader);

    if (aois.length === 0) {
      sidebar.append(
        el("p", { class: "muted" }, "Draw a box over the stimulus to measure attention on it.")
      );
    } else {
      const aggregates = aggregateAois(
        aois,
        set.map((a) => a.aoiResults)
      );
      const table = el(
        "table",
        { class: "data-table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "Region"),
            el("th", {}, "Seen"),
            el("th", {}, "Dwell"),
            el("th", {}, "TTFF")
          )
        ),
        el(
          "tbody",
          {},
          ...aggregates.map((agg) =>
            el(
              "tr",
              {},
              el(
                "td",
                {},
                el("input", {
                  class: "inline-input",
                  value: agg.label,
                  "aria-label": "Region name",
                  "data-key": `aoi-name-${agg.aoiId}`,
                  onchange: (event: Event) => {
                    const aoi = aois.find((a) => a.id === agg.aoiId);
                    if (aoi) {
                      aoi.label = (event.target as HTMLInputElement).value;
                      persistAois();
                      analysed = analyse();
                      // A rename moves no gaze: refresh the on-stage box
                      // labels and skip the heatmap repaint entirely.
                      renderAoiBoxes(aoiLayer, aois, onAoiChange);
                    }
                  },
                })
              ),
              el("td", {}, formatPercent(agg.hitRate)),
              el("td", {}, formatMs(agg.meanDwell)),
              el(
                "td",
                {},
                agg.meanTimeToFirstFixation === null ? "—" : formatMs(agg.meanTimeToFirstFixation)
              )
            )
          )
        )
      );
      sidebar.append(table);
    }

    // Exports
    sidebar.append(
      el("h3", {}, "Export"),
      el(
        "div",
        { class: "button-grid" },
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
            "data-key": "export-png",
            onclick: () => void exportOverlayPng(study, stimulusImage, overlay, mode),
          },
          "PNG overlay"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
            "data-key": "export-fixations",
            onclick: () =>
              exportFixationsCsv(
                study,
                analysed.map((a) => ({
                  participant: a.recording.participant,
                  fixations: a.fixations,
                }))
              ),
          },
          "Fixations CSV"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
            "data-key": "export-raw",
            onclick: () => exportRawCsv(study, recordings),
          },
          "Raw CSV"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
            "data-key": "export-aoi",
            onclick: () =>
              exportAoiCsv(
                study,
                aggregateAois(
                  aois,
                  analysed.map((a) => a.aoiResults)
                ),
                analysed.map((a) => ({
                  participant: a.recording.participant,
                  results: a.aoiResults,
                }))
              ),
          },
          "AOI CSV"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
            "data-key": "export-json",
            onclick: () => exportStudyJson(study, recordings),
          },
          "Session JSON"
        )
      )
    );

    // Per-recording management
    sidebar.append(
      el("h3", {}, "Recordings"),
      el(
        "ul",
        { class: "recording-list" },
        ...analysed.map((a) =>
          el(
            "li",
            {},
            el(
              "div",
              {},
              el("strong", {}, a.recording.participant),
              el(
                "span",
                { class: "muted" },
                ` ${a.fixations.length} fixations · ${formatPercent(a.recording.quality.trackingRatio)} tracked`
              )
            ),
            recordingDeleteButton(a.recording)
          )
        )
      )
    );

    if (focusKey) {
      sidebar.querySelector<HTMLElement>(`[data-key="${focusKey}"]`)?.focus();
    }
  }

  function recordingDeleteButton(recording: Recording): HTMLButtonElement {
    // A recorded session cannot be re-recorded, so this is the one place a
    // single mis-click could destroy a participant's time.
    const btn = confirmButton("Delete", "Really delete?", async () => {
      await deleteRecording(recording.id);
      const index = recordings.findIndex((r) => r.id === recording.id);
      if (index >= 0) recordings.splice(index, 1);
      fixationsById.delete(recording.id);
      analysed = analyse();
      if (selected === recording.id) selected = "all";
      void draw();
      renderSidebar();
    });
    btn.setAttribute("aria-label", `Delete ${recording.participant}'s recording`);
    btn.dataset.key = `delete-${recording.id}`;
    return btn;
  }

  renderSidebar();
  await draw();

  // Live resizes fire many events per second, and each full redraw re-splats
  // the entire heatmap. A trailing-edge debounce redraws once when the drag
  // settles; in the interim the CSS-sized canvas simply stretches.
  let resizeTimer = 0;
  const onResize = () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => void draw(), 150);
  };
  window.addEventListener("resize", onResize);

  // The caller replaces the host contents on navigation; clean up after it.
  const observer = new MutationObserver(() => {
    if (!host.isConnected || !stage.isConnected) {
      window.removeEventListener("resize", onResize);
      window.clearTimeout(resizeTimer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      observer.disconnect();
    }
  });
  observer.observe(host, { childList: true });
}

function statRow(label: string, value: string): HTMLElement {
  return el("div", { class: "stat-row" }, el("span", {}, label), el("strong", {}, value));
}

function averageOf(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function renderRawPoints(canvas: HTMLCanvasElement, recordings: Recording[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // fillRect rather than one beginPath/arc/fill per sample: a minute of gaze
  // is ~1800 dots per participant, and at this size square dots are
  // indistinguishable from round ones while keeping the per-dot alpha
  // build-up that makes dense clusters read darker.
  const size = 5;
  recordings.forEach((recording, index) => {
    const hue = (index * 67) % 360;
    ctx.fillStyle = `hsla(${hue}, 90%, 60%, 0.35)`;
    for (const p of recording.points) {
      ctx.fillRect(p.x * canvas.width - size / 2, p.y * canvas.height - size / 2, size, size);
    }
  });
}

function renderAoiBoxes(layer: HTMLElement, aois: Aoi[], onChange: () => void): void {
  for (const node of Array.from(layer.querySelectorAll(".aoi-box:not(.is-drawing)"))) {
    node.remove();
  }

  for (const aoi of aois) {
    const box = el(
      "div",
      { class: "aoi-box" },
      el("span", { class: "aoi-label" }, aoi.label),
      el(
        "button",
        {
          class: "aoi-remove",
          type: "button",
          title: "Remove region",
          "aria-label": `Remove region ${aoi.label}`,
          onclick: (event: Event) => {
            event.stopPropagation();
            const index = aois.findIndex((a) => a.id === aoi.id);
            if (index >= 0) aois.splice(index, 1);
            onChange();
          },
        },
        "×"
      )
    );
    box.style.left = `${aoi.x * 100}%`;
    box.style.top = `${aoi.y * 100}%`;
    box.style.width = `${aoi.width * 100}%`;
    box.style.height = `${aoi.height * 100}%`;
    layer.append(box);
  }
}
