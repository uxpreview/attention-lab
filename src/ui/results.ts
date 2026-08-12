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
import { clear, el, formatMs, formatPercent, nextFrame } from "./dom";
import { isTallStimulus } from "./record";

type ViewMode = HeatmapStyle | "scanpath" | "raw";

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
  // Stimulus, overlay, and AOI layer share this wrapper so that scrolling a
  // tall stimulus moves all three together and alignment never drifts.
  const wrap = el("div", { class: "results-wrap" });
  const overlay = el("canvas", { class: "results-overlay" });
  const aoiLayer = el("div", { class: "aoi-layer" });
  let stimulusImage: HTMLImageElement | null = null;
  let objectUrl: string | null = null;
  let stimulusFrame: HTMLElement | null = null;
  let frameIframe: HTMLIFrameElement | null = null;
  let frameNote: HTMLElement | null = null;
  // Recorded viewport of the first recording: the page is re-rendered at that
  // exact size and scaled down, so its layout matches what participants saw.
  let recordedW = 1280;
  let recordedH = 720;

  if (study.stimulus.kind === "image") {
    objectUrl = URL.createObjectURL(study.stimulus.blob);
    stimulusImage = el("img", { class: "results-image", src: objectUrl, alt: "" });
    if (isTallStimulus(study.stimulus.width, study.stimulus.height)) {
      stimulusImage.classList.add("is-full");
      wrap.classList.add("is-full");
    }
    wrap.append(stimulusImage);
  } else if (recordings.length > 0) {
    const rect = recordings[0].quality.stimulusRect;
    recordedW = Math.round(rect.width || recordings[0].quality.viewportWidth || 1280);
    recordedH = Math.round(rect.height || recordings[0].quality.viewportHeight || 720);
    stimulusFrame = el("div", { class: "results-frame-wrap" });
    stimulusFrame.style.aspectRatio = `${recordedW} / ${recordedH}`;
    frameIframe = el("iframe", {
      class: "results-frame",
      src: study.stimulus.url,
      referrerpolicy: "no-referrer",
      tabindex: "-1",
      title: "Recorded page",
    });
    stimulusFrame.append(frameIframe);
    wrap.classList.add("is-full");
    wrap.append(stimulusFrame);
    frameNote = el(
      "p",
      { class: "muted results-frame-note" },
      `Live re-render of ${study.stimulus.url} at the recorded ${recordedW}×${recordedH} viewport — dynamic content may differ from what participants saw.`
    );
  } else {
    stage.append(
      el(
        "div",
        { class: "results-placeholder" },
        el("p", {}, "Live page stimulus"),
        el("code", {}, study.stimulus.url)
      )
    );
  }
  wrap.append(overlay, aoiLayer);
  stage.append(wrap);

  const sidebar = el("aside", { class: "results-sidebar" });
  const layout = el(
    "div",
    { class: "results-layout" },
    el("div", { class: "results-main" }, stage, frameNote),
    sidebar
  );

  const header = el(
    "header",
    { class: "results-header" },
    el("button", { class: "btn btn-ghost", type: "button", onclick: onBack }, "← Studies"),
    el("h1", {}, study.name),
    el("span", { class: "pill" }, `${recordings.length} recording${recordings.length === 1 ? "" : "s"}`)
  );

  host.append(header, layout);

  if (recordings.length === 0) {
    sidebar.append(
      el(
        "div",
        { class: "empty" },
        el("p", {}, "No recordings yet."),
        el("p", { class: "muted" }, "Run a session from the study list to collect data.")
      )
    );
    return;
  }

  if (stimulusImage && !stimulusImage.complete) {
    await new Promise<void>((resolve) =>
      stimulusImage!.addEventListener("load", () => resolve(), { once: true })
    );
  }

  const analyse = (): AnalysedRecording[] =>
    recordings.map((recording) => {
      const rect = recording.quality.stimulusRect;
      const width = rect.width || recording.quality.viewportWidth || 1280;
      const height = rect.height || recording.quality.viewportHeight || 720;

      // Detect in the pixel space the participant actually saw, then normalise.
      const pxPoints = recording.points.map((p) => ({ x: p.x * width, y: p.y * height, t: p.t }));
      const fixations = detectFixations(pxPoints, {
        dispersion: Math.max(30, Math.min(width, height) * 0.045),
        minDuration: 100,
      }).map((f) => ({ ...f, x: f.x / width, y: f.y / height }));

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

    // The iframe renders at the recorded viewport size and is scaled to fit
    // the stage, so element positions line up with the recorded gaze.
    if (frameIframe && stimulusFrame) {
      const scale = stimulusFrame.clientWidth / recordedW;
      frameIframe.style.width = `${recordedW}px`;
      frameIframe.style.height = `${recordedH}px`;
      frameIframe.style.transform = `scale(${scale})`;
    }

    const rect = (stimulusImage ?? stimulusFrame ?? stage).getBoundingClientRect();
    // Page-length canvases at 2x device pixels cost hundreds of megabytes of
    // ImageData in the heatmap pass; drop to 1x where nobody can see the
    // difference anyway.
    const dpr = rect.height > 2600 ? 1 : Math.min(2, window.devicePixelRatio || 1);

    // The stimulus is the wrapper's first child at its origin, so the overlay
    // pins to 0,0 and scrolls with it.
    overlay.style.left = "0px";
    overlay.style.top = "0px";
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));

    aoiLayer.style.left = overlay.style.left;
    aoiLayer.style.top = overlay.style.top;
    aoiLayer.style.width = overlay.style.width;
    aoiLayer.style.height = overlay.style.height;

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

    renderAoiBoxes(aoiLayer, aois, () => {
      persistAois();
      analysed = analyse();
      void draw();
      renderSidebar();
    });
  };

  const persistAois = (): void => {
    study.aois = aois;
    void saveStudy(study);
  };

  // --- AOI drawing -------------------------------------------------------

  aoiLayer.addEventListener("pointerdown", (event: PointerEvent) => {
    if (!drawingAoi) return;
    event.preventDefault();

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
      persistAois();
      analysed = analyse();
      drawingAoi = false;
      aoiLayer.classList.remove("is-drawing");
      void draw();
      renderSidebar();
    };

    aoiLayer.addEventListener("pointermove", onMove);
    aoiLayer.addEventListener("pointerup", onUp);
  });

  // --- Sidebar -----------------------------------------------------------

  function renderSidebar(): void {
    clear(sidebar);

    const modes: Array<[ViewMode, string]> = [
      ["heat", "Heatmap"],
      ["spotlight", "Spotlight"],
      ["contour", "Contours"],
      ["scanpath", "Scanpath"],
      ["raw", "Raw gaze"],
    ];

    sidebar.append(
      el("h3", {}, "View"),
      el(
        "div",
        { class: "segmented" },
        ...modes.map(([value, label]) =>
          el(
            "button",
            {
              class: `seg ${mode === value ? "is-active" : ""}`,
              type: "button",
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

    sidebar.append(el("h3", {}, "Participants"), participantSelect);

    if (mode === "scanpath" && selected === "all" && analysed.length > 1) {
      sidebar.append(
        el(
          "p",
          { class: "note" },
          `Scanpaths are per-person. Showing ${analysed[0].recording.participant} — pick a participant to see others.`
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
                  onchange: (event: Event) => {
                    const aoi = aois.find((a) => a.id === agg.aoiId);
                    if (aoi) {
                      aoi.label = (event.target as HTMLInputElement).value;
                      persistAois();
                      analysed = analyse();
                      void draw();
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
            onclick: () => void exportOverlayPng(study, stimulusImage, overlay, mode),
          },
          "PNG overlay"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
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
            onclick: () => exportRawCsv(study, recordings),
          },
          "Raw CSV"
        ),
        el(
          "button",
          {
            class: "btn btn-small",
            type: "button",
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
            el(
              "button",
              {
                class: "btn btn-ghost btn-small",
                type: "button",
                onclick: async () => {
                  await deleteRecording(a.recording.id);
                  const index = recordings.findIndex((r) => r.id === a.recording.id);
                  if (index >= 0) recordings.splice(index, 1);
                  analysed = analyse();
                  if (selected === a.recording.id) selected = "all";
                  void draw();
                  renderSidebar();
                },
              },
              "Delete"
            )
          )
        )
      )
    );
  }

  renderSidebar();
  await draw();

  const onResize = () => void draw();
  window.addEventListener("resize", onResize);

  // The caller replaces the host contents on navigation; clean up after it.
  const observer = new MutationObserver(() => {
    if (!host.isConnected || !stage.isConnected) {
      window.removeEventListener("resize", onResize);
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

  recordings.forEach((recording, index) => {
    const hue = (index * 67) % 360;
    ctx.fillStyle = `hsla(${hue}, 90%, 60%, 0.35)`;
    for (const p of recording.points) {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 2.5, 0, Math.PI * 2);
      ctx.fill();
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
