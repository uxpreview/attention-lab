import { aggregateAois, analyseAois, type Aoi, type AoiResult } from "../analysis/aoi";
import { detectFixations, summarise, type Fixation } from "../analysis/fixations";
import { renderHeatmap, type HeatPoint } from "../analysis/heatmap";
import { OVERLAY_LABELS, participantColour, type OverlayMode } from "../analysis/legend";
import {
  ERROR_BAD_DEG,
  gradeError,
  gradeRecording,
  gradeTracking,
  isLowSignal,
  lowSignalReason,
  pxToDegrees,
  TRACKING_BAD,
  type QualityGrade,
} from "../analysis/quality";
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
import { appBar } from "./chrome";
import { clear, confirmButton, el, formatMs, formatPercent, nextFrame } from "./dom";
import { legendElement } from "./legend";

type ViewMode = OverlayMode;

const VIEW_MODES: ViewMode[] = ["heat", "spotlight", "contour", "scanpath", "raw"];

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

  // A recording below the quality threshold is not evidence, and the app's own
  // bench notes say why: a bad calibration "looks exactly like data". Folding a
  // 44%-tracked session into the same heatmap as a 91% one produces a blended
  // number that describes neither. They are excluded from the aggregate by
  // default, said out loud in the Summary, and one click away from being
  // included — unless every recording is flagged, in which case excluding them
  // all would leave an empty stage with no explanation.
  let includeLowSignal = false;
  const flaggedRecordings = (): Recording[] => recordings.filter((r) => isLowSignal(r.quality));
  /** True only when there is something to exclude *and* something left after
   * excluding it. Recomputed rather than captured, because deleting the one
   * clean recording would otherwise leave the screen hiding everything it has
   * and reporting on nothing. */
  const excludingLowSignal = (): boolean => {
    const flagged = flaggedRecordings().length;
    return !includeLowSignal && flagged > 0 && flagged < recordings.length;
  };

  const stage = el("div", { class: "results-stage" });
  const overlay = el("canvas", { class: "results-overlay", role: "img" });
  const aoiLayer = el("div", { class: "aoi-layer" });
  let stimulusImage: HTMLImageElement | null = null;
  let objectUrl: string | null = null;

  // The stage is the primary surface, so it explains its own emptiness rather
  // than leaving that to the sidebar. There is exactly one such message: a URL
  // study used to print this line and the placeholder's on the same baselines,
  // one translucent layer over the other, and the two texts interleaved
  // character for character.
  const emptyLine = "No recordings yet — run a session to see attention here.";

  if (study.stimulus.kind === "image") {
    objectUrl = URL.createObjectURL(study.stimulus.blob);
    stimulusImage = el("img", { class: "results-image", src: objectUrl, alt: "" });
    stage.append(stimulusImage);
    if (recordings.length === 0) {
      stage.append(el("div", { class: "stage-empty" }, el("p", {}, emptyLine)));
    }
  } else {
    stage.append(
      el(
        "div",
        { class: "results-placeholder" },
        el("p", { class: "label" }, "Live page stimulus"),
        el("code", {}, study.stimulus.url),
        el(
          "p",
          { class: "muted" },
          recordings.length === 0
            ? `${emptyLine} Overlays are drawn against the recorded viewport, not a re-render of the page.`
            : "Overlays are drawn against the recorded viewport, not a re-render of the page."
        )
      )
    );
  }
  stage.append(overlay, aoiLayer);

  const legend = el("figure", { class: "legend-slot" });
  const sidebar = el("aside", { class: "results-sidebar" });
  const layout = el(
    "div",
    { class: "results-layout" },
    el("div", { class: "results-main" }, stage, legend),
    sidebar
  );

  // Export is the whole point of a research tool, so it is a persistent
  // toolbar action beside the recordings count rather than the last block of a
  // sidebar — where it sat below the fold of an invisible nested scroller.
  const exportBar = el("div", { class: "results-actions" });
  const header = el(
    "header",
    { class: "results-header" },
    el(
      "div",
      { class: "results-title" },
      el("button", { class: "btn btn-ghost btn-small", type: "button", onclick: onBack }, "← Studies"),
      el("h1", {}, study.name),
      el(
        "span",
        { class: "pill" },
        `${recordings.length} recording${recordings.length === 1 ? "" : "s"}`
      )
    ),
    exportBar
  );

  // The results screen sits on the same measure as the experiment page around
  // it; .container is the site's shell. The bar above it keeps the wordmark and
  // the route back to the site on the deepest screen in the app.
  host.append(appBar(), el("div", { class: "container screen screen-fill" }, header, layout));

  if (recordings.length === 0) {
    sidebar.append(
      el(
        "div",
        { class: "empty" },
        el("h3", {}, "No recordings yet"),
        el("p", { class: "muted" }, "Run a session from the study list to collect data.")
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

  const activeSet = (): AnalysedRecording[] => {
    if (selected !== "all") return analysed.filter((a) => a.recording.id === selected);
    if (!excludingLowSignal()) return analysed;
    return analysed.filter((a) => !isLowSignal(a.recording.quality));
  };

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

    overlay.setAttribute("aria-label", `${OVERLAY_LABELS[mode]} overlay`);

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

    // Spotlight dims the stage to near-black, which a deep-teal region box and
    // its dark label chip simply disappear into. The class flips both to the
    // cream side of the palette for as long as the mask is up.
    stage.classList.toggle("stage--spotlight", mode === "spotlight");

    // Only the scanpath's own participant is keyed, because that is the only
    // one on the stage.
    const keyed = mode === "scanpath" ? set.slice(0, 1) : set;
    clear(legend);
    legend.append(legendElement(mode, keyed.map((a) => a.recording.participant)));

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
        ...VIEW_MODES.map((value) =>
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
            OVERLAY_LABELS[value]
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
      el("option", { value: "all" }, `All participants (${activeSet().length})`),
      ...analysed.map((a) =>
        el(
          "option",
          { value: a.recording.id, ...(selected === a.recording.id ? { selected: true } : {}) },
          isLowSignal(a.recording.quality)
            ? `${a.recording.participant} — low signal`
            : a.recording.participant
        )
      )
    );
    participantSelect.value = selected;

    sidebar.append(
      el("h3", {}, el("label", { for: "participant-filter" }, "Participants")),
      participantSelect
    );

    const flagged = flaggedRecordings();
    const excluding = excludingLowSignal();

    if (flagged.length > 0 && flagged.length < recordings.length) {
      const toggle = el("input", {
        type: "checkbox",
        "data-key": "include-low",
        ...(includeLowSignal ? { checked: true } : {}),
        onchange: (event: Event) => {
          includeLowSignal = (event.target as HTMLInputElement).checked;
          void draw();
          renderSidebar();
        },
      });
      sidebar.append(
        el(
          "label",
          { class: "checkbox checkbox-quiet" },
          toggle,
          el(
            "span",
            {},
            `Include ${flagged.length} low-signal recording${flagged.length === 1 ? "" : "s"} in the aggregate`
          )
        )
      );
    }

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

    const meanTracking = averageOf(set.map((a) => a.recording.quality.trackingRatio)) ?? 0;

    sidebar.append(
      el("h3", {}, "Summary"),
      statRow("Fixations", String(stats.fixationCount)),
      statRow("Mean fixation", formatMs(stats.meanFixationDuration)),
      statRow("Time to first fixation", formatMs(stats.timeToFirstFixation)),
      // The two quality numbers wear their grade. Reporting a blended 69%
      // tracked in the same grey as a clean 91% is what let a broken session
      // pass for a finding.
      statRow("Tracking ratio", formatPercent(meanTracking), gradeTracking(meanTracking)),
      statRow(
        "Calibration error",
        meanValidation === null
          ? "—"
          : `${Math.round(meanValidation)}px · ~${pxToDegrees(meanValidation).toFixed(1)}°`,
        meanValidation === null ? undefined : gradeError(meanValidation)
      )
    );

    if (flagged.length > 0 && selected === "all") {
      const isAre = flagged.length === 1 ? "is" : "are";
      const threshold = `under ${Math.round(TRACKING_BAD * 100)}% tracked, or over ${ERROR_BAD_DEG}° of calibration error`;
      sidebar.append(
        el(
          "p",
          { class: `note ${excluding ? "" : "note-warn"}` },
          excluding
            ? `${flagged.length} of ${recordings.length} recordings ${isAre} below the quality threshold (${threshold}) and ${isAre} excluded from these numbers and the overlay.`
            : `${flagged.length} of ${recordings.length} recordings ${isAre} below the quality threshold (${threshold}) and ${isAre} included in these numbers. Read them as indicative, not as findings.`
        )
      );
    }

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
              {
                // With five or six overlapping regions there is otherwise no
                // way to tell which row is which box on the stimulus. Focus
                // counts as well as hover, or the link is mouse-only.
                "data-aoi-row": agg.aoiId,
                onmouseenter: () => highlightAoi(agg.aoiId, true),
                onmouseleave: () => highlightAoi(agg.aoiId, false),
                onfocusin: () => highlightAoi(agg.aoiId, true),
                onfocusout: () => highlightAoi(agg.aoiId, false),
              },
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

    // Per-recording management
    sidebar.append(
      el("h3", {}, "Recordings"),
      el(
        "ul",
        { class: "recording-list" },
        ...analysed.map((a) => {
          const grade = gradeRecording(a.recording.quality);
          const low = grade === "bad";
          const excluded = low && excluding && selected === "all";
          return el(
            "li",
            { class: low ? "is-low-signal" : "" },
            el(
              "div",
              { class: "recording-meta" },
              el(
                "div",
                {},
                el("strong", {}, a.recording.participant),
                low
                  ? el(
                      "span",
                      {
                        class: "quality-badge",
                        title: `Low signal: ${lowSignalReason(a.recording.quality)}.${excluded ? " Excluded from the aggregate." : ""}`,
                      },
                      excluded ? "Low signal · excluded" : "Low signal"
                    )
                  : null
              ),
              el(
                "span",
                { class: `recording-stats signal-${grade}` },
                `${a.fixations.length} fixations · ${formatPercent(a.recording.quality.trackingRatio)} tracked`
              )
            ),
            recordingDeleteButton(a.recording)
          );
        })
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

  /** Lights the box on the stimulus that a table row describes. */
  function highlightAoi(aoiId: string, on: boolean): void {
    aoiLayer
      .querySelector<HTMLElement>(`.aoi-box[data-aoi="${aoiId}"]`)
      ?.classList.toggle("is-linked", on);
  }

  // The toolbar is built once: every handler reads `mode`, `selected`,
  // `analysed` and `aois` when it fires, so it never goes stale.
  exportBar.append(
    el("span", { class: "label results-actions-label" }, "Export"),
    exportButton("PNG overlay", "export-png", () => {
      const set = activeSet();
      const keyed = mode === "scanpath" ? set.slice(0, 1) : set;
      void exportOverlayPng(study, stimulusImage, overlay, {
        mode,
        participants: keyed.map((a) => a.recording.participant),
        scope:
          selected === "all"
            ? excludingLowSignal()
              ? "All participants, low-signal excluded"
              : "All participants"
            : (keyed[0]?.recording.participant ?? "All participants"),
      });
    }),
    exportButton("Fixations CSV", "export-fixations", () =>
      exportFixationsCsv(
        study,
        analysed.map((a) => ({ participant: a.recording.participant, fixations: a.fixations }))
      )
    ),
    exportButton("Raw CSV", "export-raw", () => exportRawCsv(study, recordings)),
    exportButton("AOI CSV", "export-aoi", () =>
      exportAoiCsv(
        study,
        aggregateAois(
          aois,
          analysed.map((a) => a.aoiResults)
        ),
        analysed.map((a) => ({ participant: a.recording.participant, results: a.aoiResults }))
      )
    ),
    exportButton("Session JSON", "export-json", () => exportStudyJson(study, recordings))
  );

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

function exportButton(label: string, key: string, onclick: () => void): HTMLButtonElement {
  return el("button", { class: "btn btn-small", type: "button", "data-key": key, onclick }, label);
}

function statRow(label: string, value: string, grade?: QualityGrade): HTMLElement {
  return el(
    "div",
    { class: "stat-row" },
    el("span", {}, label),
    el("strong", { class: grade ? `signal-${grade}` : "" }, value)
  );
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
  //
  // Colours come from the shared categorical palette, which is also what the
  // legend keys — three unattributable dot clouds are worse than one.
  const size = 5;
  recordings.forEach((recording, index) => {
    ctx.fillStyle = participantColour(index, 0.35);
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
      { class: "aoi-box", "data-aoi": aoi.id },
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
