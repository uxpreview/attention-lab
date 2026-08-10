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
import {
  clear,
  confirmButton,
  el,
  formatMs,
  formatPercent,
  nextFrame,
  relativeDay,
} from "./dom";
import { legendElement } from "./legend";

type ViewMode = OverlayMode;

const VIEW_MODES: ViewMode[] = ["heat", "spotlight", "contour", "scanpath", "raw"];

interface AnalysedRecording {
  recording: Recording;
  fixations: Fixation[];
  aoiResults: AoiResult[];
}

export interface ResultsOptions {
  /** Starts a session for this study, when this machine can run one. The
   * results screen is a study's home, and at zero recordings it was a dead end
   * whose only advice was to navigate somewhere else. */
  onRunSession?: (() => void) | null;
  /** Why a session cannot be started from here, if it cannot. */
  runBlockedReason?: string | null;
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
  onBack: () => void,
  options: ResultsOptions = {}
): Promise<void> {
  clear(host);

  const recordings = await listRecordings(study.id);
  /** Everything downstream of a recording — overlays, summary numbers, the
   * per-region table, exports — is skipped when there are none. Everything
   * upstream of one, above all the region tools, is not: a study is worth
   * marking up before the first participant sits down, which is exactly what
   * the study card's Results button already promises. */
  const hasRecordings = recordings.length > 0;
  let aois: Aoi[] = [...study.aois];
  let mode: ViewMode = "heat";
  let selected: string | "all" = "all";
  let drawingAoi = false;
  /** Whether the region layer is drawn over the stimulus at all. Regions are a
   * separate layer of annotation from the attention data, and the overlay
   * people screenshot into a deck is usually the clean one — Tobii keeps AOI
   * visibility on its own switch for the same reason. Defaults on, because a
   * region you cannot see is a region you forget you drew. */
  let showAois = true;

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

  /** The way out of an empty results screen. Prose telling someone to navigate
   * back is not a way out; the button that does the thing is. */
  const runAction = (): HTMLElement => {
    if (options.onRunSession) {
      return el(
        "button",
        { class: "btn btn-primary", type: "button", onclick: options.onRunSession },
        "Run session"
      );
    }
    return el(
      "button",
      {
        class: "btn btn-primary",
        type: "button",
        disabled: true,
        title: options.runBlockedReason ?? "Sessions need a laptop or a desktop",
      },
      "Run session"
    );
  };

  /** The "nothing recorded yet" bar. It lives *under* the stage, not on it.
   * Floated inside the stage it was a card with a border and no shadow sitting
   * over the middle of a short wireframe — a modal that had lost its backdrop,
   * covering the very stimulus the regions are meant to be drawn on. Below the
   * stage it reads as what it is: a footer bar belonging to the stage, with the
   * action in it. */
  let emptyStrip: HTMLElement | null = null;

  if (study.stimulus.kind === "image") {
    objectUrl = URL.createObjectURL(study.stimulus.blob);
    stimulusImage = el("img", { class: "results-image", src: objectUrl, alt: "" });
    stage.append(stimulusImage);
    if (!hasRecordings) {
      emptyStrip = el("div", { class: "stage-empty" }, el("p", {}, emptyLine), runAction());
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
          hasRecordings
            ? "Overlays are drawn against the recorded viewport, not a re-render of the page."
            : `${emptyLine} Overlays are drawn against the recorded viewport, not a re-render of the page.`
        ),
        hasRecordings ? null : el("div", { class: "placeholder-actions" }, runAction())
      )
    );
  }
  stage.append(overlay, aoiLayer);

  const legend = el("figure", { class: "legend-slot" });
  // Regions and recordings live under the stage, at the width of the stage.
  // In the 320px rail they were four ~50px columns of numbers beside a full
  // screen-height of empty cream — the tool presenting its own findings worse
  // than it presents the stimulus. What stays in the rail is what governs the
  // stage: the view, who is in it, and the headline numbers for that selection.
  const dataBlock = el("div", { class: "results-data" });
  const sidebar = el("aside", { class: "results-sidebar" });
  // The data block is a child of the *layout*, not of the main column, and
  // spans both grid tracks. As a child of the main column it left the rail
  // stranded: the sidebar ended level with the stage while the tables ran on
  // for another 437px beside a column of empty cream, and every recording added
  // made the gap taller. Spanning, the tables get the full measure and the rail
  // only has to be as tall as the thing it governs.
  const layout = el(
    "div",
    { class: "results-layout" },
    el("div", { class: "results-main" }, stage, emptyStrip, legend),
    sidebar,
    dataBlock
  );

  // Export is the whole point of a research tool, so it is a persistent
  // toolbar action rather than the last block of a sidebar — where it sat below
  // the fold of an invisible nested scroller. One button, not five: the header
  // read as six equivalent pills, of which the first was not a control at all.
  const exportMenu = el("div", { class: "menu", role: "menu", hidden: true });
  const exportToggle = el(
    "button",
    {
      class: "btn btn-small",
      type: "button",
      "aria-haspopup": "true",
      "aria-expanded": "false",
      onclick: () => setExportOpen(exportMenu.hidden),
    },
    "Export ",
    el("span", { class: "menu-caret", "aria-hidden": "true" }, "▾")
  );
  const exportBar = el("div", { class: "results-actions" }, exportToggle, exportMenu);

  /**
   * The count in the header, which has to agree with the rail three inches to
   * its right.
   *
   * It read `recordings.length` while the sidebar's participant list read the
   * active set, so a study with one auto-excluded low-signal session opened on
   * "4 recordings" beside "All participants (3)": two contradictory totals, both
   * correct, with the reconciliation buried in a note further down. Saying "3 of
   * 4" here makes that note a confirmation instead of a correction.
   */
  const countPill = el("span", { class: "pill pill-count" });
  function paintCountPill(): void {
    const total = recordings.length;
    const kept = aggregateSet().length;
    countPill.textContent =
      kept === total
        ? `${total} recording${total === 1 ? "" : "s"}`
        : `${kept} of ${total} recordings`;
    countPill.title =
      kept === total ? "" : "Low-signal recordings are excluded from the aggregate";
  }

  function exportItem(label: string, key: string, run: () => void): HTMLButtonElement {
    return el(
      "button",
      {
        class: "menu-item",
        type: "button",
        role: "menuitem",
        "data-key": key,
        onclick: () => {
          setExportOpen(false);
          run();
        },
      },
      label
    );
  }

  function setExportOpen(open: boolean): void {
    exportMenu.hidden = !open;
    exportToggle.setAttribute("aria-expanded", open ? "true" : "false");
    exportToggle.classList.toggle("is-active", open);
  }

  // A menu that cannot be dismissed by clicking away from it is a trap.
  const onDocumentPointer = (event: Event) => {
    if (!exportBar.contains(event.target as Node)) setExportOpen(false);
  };
  const onDocumentKey = (event: KeyboardEvent) => {
    if (event.key !== "Escape" || exportMenu.hidden) return;
    setExportOpen(false);
    exportToggle.focus();
  };
  document.addEventListener("pointerdown", onDocumentPointer);
  document.addEventListener("keydown", onDocumentKey);

  const header = el(
    "header",
    { class: "results-header" },
    el(
      "div",
      { class: "results-title" },
      el("button", { class: "btn btn-ghost btn-small", type: "button", onclick: onBack }, "← Studies"),
      el("h1", {}, study.name),
      countPill
    ),
    hasRecordings ? exportBar : null
  );

  // The results screen sits on the same measure as the experiment page around
  // it; .container is the site's shell. The bar above it keeps the wordmark and
  // the route back to the site on the deepest screen in the app.
  host.append(appBar(), el("div", { class: "container screen screen-fill" }, header, layout));

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

  /** Everything the "All participants" scope covers, whatever is selected right
   * now. The header count and the scope option's own count are facts about the
   * study, not about the current filter — reading them off activeSet() made the
   * option label say "All participants (1)" the moment one participant was
   * picked. */
  const aggregateSet = (): AnalysedRecording[] =>
    excludingLowSignal() ? analysed.filter((a) => !isLowSignal(a.recording.quality)) : analysed;

  const activeSet = (): AnalysedRecording[] =>
    selected === "all"
      ? aggregateSet()
      : analysed.filter((a) => a.recording.id === selected);

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
      renderScanpath(overlay, set[0]?.fixations ?? [], {
        minRadius: 10 * dpr,
        maxRadius: 46 * dpr,
        // The canvas is in device pixels; the ordinals have to be told, or a
        // retina display renders them at half the size they were specified at.
        scale: dpr,
      });
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
    // one on the stage. With nothing on the stage there is nothing to key: a
    // legend for an overlay that is not drawn is a caption for a missing
    // figure.
    const keyed = mode === "scanpath" ? set.slice(0, 1) : set;
    clear(legend);
    if (hasRecordings) {
      legend.append(legendElement(mode, keyed.map((a) => a.recording.participant)));
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
    renderData();
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

  // --- Sidebar and data blocks -------------------------------------------

  /**
   * Rebuilds a container's contents, handing keyboard focus back afterwards.
   * Rebuilding destroys whatever control held focus, which would drop it to
   * <body> and force a Tab journey from the top after every interaction.
   * Controls carry a data-key so focus can find its rebuilt equivalent.
   */
  function rebuild(container: HTMLElement, build: () => void): void {
    const active = document.activeElement as HTMLElement | null;
    const focusKey = active && container.contains(active) ? active.dataset.key : undefined;
    clear(container);
    build();
    if (focusKey) container.querySelector<HTMLElement>(`[data-key="${focusKey}"]`)?.focus();
  }

  function renderSidebar(): void {
    // The header count is a function of the same state the rail is, so it is
    // repainted from the same place rather than left to drift.
    paintCountPill();
    rebuild(sidebar, buildSidebar);
  }

  function renderData(): void {
    rebuild(dataBlock, () => {
      dataBlock.append(aoiSection());
      if (hasRecordings) dataBlock.append(recordingsSection());
    });
  }

  /** Shows or hides the whole region layer. Drawing is switched off with it:
   * a crosshair over a layer nothing appears on is a control that lies. */
  function setAoisVisible(visible: boolean): void {
    showAois = visible;
    aoiLayer.classList.toggle("is-hidden", !visible);
    if (!visible && drawingAoi) {
      drawingAoi = false;
      aoiLayer.classList.remove("is-drawing");
    }
  }

  function regionToggle(): HTMLElement {
    return el(
      "label",
      { class: "checkbox checkbox-quiet" },
      el("input", {
        type: "checkbox",
        "data-key": "show-aois",
        ...(showAois ? { checked: true } : {}),
        onchange: (event: Event) => {
          setAoisVisible((event.target as HTMLInputElement).checked);
          renderSidebar();
          renderData();
        },
      }),
      el("span", {}, "Show regions")
    );
  }

  function buildSidebar(): void {
    if (!hasRecordings) {
      sidebar.append(
        el("h3", {}, "View"),
        el(
          "p",
          { class: "muted" },
          "Heatmap, spotlight, contour and scanpath views arrive with the first recording. Regions drawn now apply to every session run afterwards."
        ),
        regionToggle()
      );
      return;
    }

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
                renderData();
              },
            },
            OVERLAY_LABELS[value]
          )
        )
      ),
      regionToggle()
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
          renderData();
        },
      },
      el("option", { value: "all" }, `All participants (${aggregateSet().length})`),
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
          renderData();
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

  }

  /** Regions and their numbers, under the stage where there is room for them
   * to be a table rather than four 50px columns squeezed into a rail. */
  function aoiSection(): HTMLElement {
    const section = el(
      "section",
      { class: "results-block" },
      el(
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
              // Drawing implies looking: reaching for + Draw with the layer
              // hidden turns it back on rather than dropping the new box into
              // a layer the operator cannot see.
              if (drawingAoi && !showAois) {
                setAoisVisible(true);
                renderSidebar();
              }
              aoiLayer.classList.toggle("is-drawing", drawingAoi);
              renderData();
            },
          },
          drawingAoi ? "Cancel" : "+ Draw"
        )
      )
    );

    if (aois.length === 0) {
      section.append(
        el("p", { class: "muted" }, "Draw a box over the stimulus to measure attention on it.")
      );
      return section;
    }

    const aggregates = aggregateAois(
      aois,
      activeSet().map((a) => a.aoiResults)
    );

    section.append(
      el(
        "table",
        { class: "data-table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "Region"),
            // Without recordings these columns would be four columns of zeroes
            // presented as measurements.
            hasRecordings ? el("th", {}, "Seen") : null,
            hasRecordings ? el("th", {}, "Dwell") : null,
            // An acronym nobody outside the field reads on sight, in a tool
            // whose whole claim is that it explains its own numbers.
            hasRecordings
              ? el("th", {}, el("abbr", { title: "Time to first fixation" }, "TTFF"))
              : null
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
                { class: "cell-name" },
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
              hasRecordings ? el("td", {}, formatPercent(agg.hitRate)) : null,
              hasRecordings ? el("td", {}, formatMs(agg.meanDwell)) : null,
              hasRecordings
                ? el(
                    "td",
                    {},
                    agg.meanTimeToFirstFixation === null
                      ? "—"
                      : formatMs(agg.meanTimeToFirstFixation)
                  )
                : null
            )
          )
        )
      )
    );
    return section;
  }

  function recordingsSection(): HTMLElement {
    const excluding = excludingLowSignal();
    return el(
      "section",
      { class: "results-block" },
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
              // Only the tracked percentage is a judgement. A fixation count is
              // a neutral descriptor, and painting it success-green spent the
              // palette's one "this is fine" signal on it five rows running —
              // which is what drained the meaning from the green on "Tracking
              // ratio 90%" in the summary above.
              // When a recording was made, and how long it ran. Two sessions
              // two days apart were indistinguishable here, and a researcher
              // iterating on the same wireframe has to know which round a
              // recording came from before trusting an aggregate. Coarse in the
              // line, exact in the tooltip.
              el(
                "span",
                {
                  class: "recording-stats",
                  title: new Date(a.recording.createdAt).toLocaleString(),
                },
                `${a.fixations.length} fixations · `,
                el(
                  "span",
                  { class: `signal-${grade}` },
                  `${formatPercent(a.recording.quality.trackingRatio)} tracked`
                ),
                ` · ${formatMs(recordingLength(a.recording))} · ${relativeDay(a.recording.createdAt)}`
              )
            ),
            recordingDeleteButton(a.recording)
          );
        })
      )
    );
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
      renderData();
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

  // The menu is built once: every handler reads `mode`, `selected`,
  // `analysed` and `aois` when it fires, so it never goes stale.
  exportMenu.append(
    exportItem("PNG overlay", "export-png", () => {
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
    exportItem("Fixations CSV", "export-fixations", () =>
      exportFixationsCsv(
        study,
        analysed.map((a) => ({ participant: a.recording.participant, fixations: a.fixations }))
      )
    ),
    exportItem("Raw CSV", "export-raw", () => exportRawCsv(study, recordings)),
    exportItem("AOI CSV", "export-aoi", () =>
      exportAoiCsv(
        study,
        aggregateAois(
          aois,
          analysed.map((a) => a.aoiResults)
        ),
        analysed.map((a) => ({ participant: a.recording.participant, results: a.aoiResults }))
      )
    ),
    exportItem("Session JSON", "export-json", () => exportStudyJson(study, recordings))
  );

  renderSidebar();
  renderData();
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
      document.removeEventListener("pointerdown", onDocumentPointer);
      document.removeEventListener("keydown", onDocumentKey);
      window.clearTimeout(resizeTimer);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      observer.disconnect();
    }
  });
  observer.observe(host, { childList: true });
}

function statRow(label: string, value: string, grade?: QualityGrade): HTMLElement {
  return el(
    "div",
    { class: "stat-row" },
    el("span", {}, label),
    el("strong", { class: grade ? `signal-${grade}` : "" }, value)
  );
}

/** How long the recording ran. Sample times are stored relative to the first
 * sample, so the last one is the length of the pass over the screen. */
function recordingLength(recording: Recording): number {
  return recording.points.length > 0 ? recording.points[recording.points.length - 1].t : 0;
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
