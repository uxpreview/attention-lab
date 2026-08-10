import { aggregateAois, analyseAois, type Aoi, type AoiResult } from "../analysis/aoi";
import { detectFixations, summarise, type Fixation } from "../analysis/fixations";
import { renderHeatmap, type HeatPoint } from "../analysis/heatmap";
import {
  OVERLAY_LABELS,
  participantColour,
  type LegendScale,
  type OverlayMode,
} from "../analysis/legend";
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
  type ExportScope,
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
  /** What the hot end of the heat ramp was worth on the last draw, for the
   * legend and the PNG caption to print. Only the renderer knows it — the
   * ceiling is a percentile of this selection's own blob peaks — so it is
   * captured on the way out of the draw rather than recomputed. Null for the
   * overlays that have no unit. */
  let heatScale: LegendScale | null = null;
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
  /** The box the measurements are drawn on, which is the stimulus itself.
   *
   * The overlay and the region layer used to be positioned from a measurement
   * of the image taken at draw time — right until something resized the image
   * without firing a resize event. Print does precisely that, and produced a
   * figure whose heat blobs and region boxes sat up to 30% of the figure's
   * width away from the components they were measuring. Wrapping the stimulus
   * and its two annotation layers in one shrink-wrapped box makes the tracking
   * a CSS fact instead of a cached number. */
  const figure = el("div", { class: "results-figure" });
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
    // The figure takes the stimulus's ratio so the image fills it with no
    // letterbox — a letterboxed strip is figure that is not stimulus, and the
    // overlay inset to the figure would be painting attention onto it.
    figure.style.aspectRatio = `${study.stimulus.width} / ${study.stimulus.height}`;
    figure.append(stimulusImage);
    if (!hasRecordings) {
      emptyStrip = el("div", { class: "stage-empty" }, el("p", {}, emptyLine), runAction());
    }
  } else {
    // Nothing to shrink-wrap: the figure takes the stage, which is where a URL
    // study's overlay has always been drawn.
    figure.classList.add("results-figure--fill");
    figure.append(
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
  figure.append(overlay, aoiLayer);
  stage.append(figure);

  /**
   * The stage, full screen, for the width where it stops working.
   *
   * At 390px the results screen degrades well in every respect except the one
   * that matters: the stimulus renders around 250px wide, at which point the
   * heat blobs are smudges and the numbered region badges overlap each other —
   * while the AOI table under it stays perfectly readable. So the numbers
   * survive a phone and the picture does not, and reading results on a phone
   * between meetings is a real and likely use.
   *
   * The *same* figure element is moved into the dialog and back out again,
   * rather than a copy of it being rendered there. The overlay canvas, the
   * region layer and their registration to the stimulus are all properties of
   * this element (see .results-figure in styles.css), so moving it means there
   * is no second render path that can disagree with the first — and no chance
   * of a full-screen figure whose boxes sit somewhere else, which is exactly
   * the failure the print stylesheet used to have.
   */
  const lightboxBody = el("div", { class: "lightbox-body" });
  const lightbox = el(
    "dialog",
    { class: "figure-lightbox", "aria-label": `${study.name}, full screen` },
    lightboxBody,
    el(
      "div",
      { class: "lightbox-bar" },
      // A wireframe is wider than it is tall and a phone is not; there is no
      // honest way to fix that in CSS without rotating the figure, which would
      // put every pointer coordinate in the region layer out of register with
      // the picture. Saying so costs one line.
      el("p", { class: "lightbox-hint" }, "Rotate for a wider view"),
      el(
        "button",
        { class: "btn btn-small", type: "button", onclick: () => lightbox.close() },
        "Close"
      )
    )
  );
  const zoomButton = el(
    "button",
    {
      class: "btn btn-small stage-zoom",
      type: "button",
      onclick: () => {
        lightboxBody.append(figure);
        lightbox.showModal();
        // The figure's box has just changed by a factor of four; the canvas
        // backing store has to follow it or the overlay is an upscaled blur.
        void draw();
      },
    },
    "Expand"
  );
  // Both are absolutely positioned or in the top layer, so neither takes part
  // in the stage's centring of the figure. The `close` event covers Esc and the
  // backdrop as well as the button.
  lightbox.addEventListener("close", () => {
    stage.append(figure);
    void draw();
  });
  stage.append(zoomButton, lightbox);

  const legend = el("figure", { class: "legend-slot" });
  // Regions and recordings live under the stage, at the width of the stage.
  // In the 320px rail they were four ~50px columns of numbers beside a full
  // screen-height of empty cream — the tool presenting its own findings worse
  // than it presents the stimulus. What stays in the rail is what governs the
  // stage: the view, who is in it, and the headline numbers for that selection.
  const dataBlock = el("div", { class: "results-data" });
  const sidebar = el("aside", { class: "results-sidebar" });
  /** The rail's own grid cell, and the reason the rail cannot wander.
   *
   * A sticky box is constrained by its containing block, and Blink resolves
   * that for a grid item to the grid *container's* content box — not to the
   * item's grid area, which is only special-cased for absolutely positioned
   * children. So a sticky rail placed directly in the layout treats the whole
   * page as its runway: scrolled to the foot of a populated study it sat on top
   * of the full-width Recordings card below the stage, taking a 320×296px bite
   * out of it and putting itself between the cursor and four Delete buttons.
   * Wrapping it means the containing block is this element, which is one grid
   * cell in the stage's row and stops where the stage does. */
  const rail = el("div", { class: "results-rail" }, sidebar);
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
    rail,
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
    // Position is CSS's job now (see .results-figure): both layers are inset to
    // the stimulus, so they cannot drift out of register with it. The only
    // thing left to compute is the canvas's backing-store resolution, and even
    // that degrades gracefully — a canvas whose CSS box has since grown scales
    // its bitmap, so a stale backing store is softer but never misplaced.
    const rect = figure.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    overlay.width = Math.max(1, Math.round(rect.width * dpr));
    overlay.height = Math.max(1, Math.round(rect.height * dpr));

    overlay.setAttribute("aria-label", `${OVERLAY_LABELS[mode]} overlay`);

    const set = activeSet();

    heatScale = null;

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
      const ceiling = renderHeatmap(overlay, points, { style: mode, radiusRatio: 0.055 });
      // Spotlight is a mask: the same field drives it, but what it encodes is
      // "revealed or not", so a millisecond axis under it would name a quantity
      // the picture does not carry.
      if (mode !== "spotlight" && ceiling > 0) heatScale = { ceiling };
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
      legend.append(
        legendElement(mode, keyed.map((a) => a.recording.participant), heatScale)
      );
    }

    renderAoiBoxes(aoiLayer, aois);
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

  /**
   * "Time to first fixation", defined so that it is not zero by construction.
   *
   * The row used to be `summarise(everyFixation, 0).timeToFirstFixation`, i.e.
   * the start of the first fixation of the first recording measured from that
   * recording's own t=0 — which is a few tens of milliseconds by definition,
   * because the participant is already looking at the screen when capture
   * begins. It rendered "0ms" for every study and every participant, directly
   * above two numbers that are correct and graded.
   *
   * What a researcher means by TTFF is time to first fixation *on something*.
   * With regions drawn, that is computable: per recording, the earliest
   * fixation landing inside any region, averaged over the recordings that found
   * one. With no regions there is no "something", so the row is not shown at
   * all rather than shown as a zero.
   */
  function timeToFirstRegionRow(set: AnalysedRecording[]): HTMLElement[] {
    if (aois.length === 0) return [];

    const firsts = set
      .map((a) => {
        const times = a.aoiResults
          .map((r) => r.timeToFirstFixation)
          .filter((t): t is number => t !== null);
        return times.length > 0 ? Math.min(...times) : null;
      })
      .filter((t): t is number => t !== null);

    const found = firsts.length;
    const mean = averageOf(firsts);
    return [
      statRow(
        "Time to first region",
        mean === null ? "—" : formatMs(mean),
        undefined,
        mean === null
          ? "No participant in this selection fixated any region."
          : `Mean over the ${found} of ${set.length} recording${set.length === 1 ? "" : "s"} whose gaze reached a region.`
      ),
    ];
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
      // Print drops the chips — a live control on paper is noise — and used to
      // leave the VIEW heading standing over nothing. This says which view the
      // figure above it actually is, and is invisible on screen.
      el("p", { class: "print-only" }, OVERLAY_LABELS[mode]),
      regionToggle()
    );

    const participantSelect = el(
      "select",
      {
        // .select is .input plus `appearance: none` and a drawn chevron: this
        // was the one control in the app the browser, rather than this design
        // system, decided the look of — an OS dropdown sitting directly under a
        // set of hand-built pills.
        class: "input select",
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

    const scopeName =
      selected === "all"
        ? `All participants (${aggregateSet().length})`
        : (analysed.find((a) => a.recording.id === selected)?.recording.participant ??
          "All participants");

    sidebar.append(
      el("h3", {}, el("label", { for: "participant-filter" }, "Participants")),
      participantSelect,
      // The select is a form control and prints as one; what a reader of the
      // printout needs is the scope the figure was drawn at, in text.
      el("p", { class: "print-only" }, scopeName)
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

    // What the stage is actually drawing, which is not always the whole
    // selection: a scanpath is one person's path by construction. Both the note
    // and the numbers below are scoped to it — the note used to name
    // `analysed[0]` while the draw path used `activeSet()[0]`, so with the
    // first recording auto-excluded as low signal it captioned the picture with
    // somebody else's name, and the rail reported the 4-participant aggregate
    // ("Fixations 110") beside a picture of 36.
    const set = activeSet();
    const soloScanpath = mode === "scanpath" && selected === "all" && set.length > 1;
    const statSet = soloScanpath ? set.slice(0, 1) : set;

    if (soloScanpath) {
      sidebar.append(
        el(
          "p",
          { class: "note" },
          `Scanpaths are per-person. Showing ${statSet[0].recording.participant}. Pick a participant to see others.`
        )
      );
    }
    const stats = summarise(
      statSet.flatMap((a) => a.fixations),
      0
    );
    const meanValidation = averageOf(
      statSet.map((a) => a.recording.quality.validationError).filter((v): v is number => v !== null)
    );

    const meanTracking = averageOf(statSet.map((a) => a.recording.quality.trackingRatio)) ?? 0;

    sidebar.append(
      el(
        "h3",
        {},
        soloScanpath ? `Summary — ${statSet[0].recording.participant}` : "Summary"
      ),
      statRow("Fixations", String(stats.fixationCount)),
      statRow("Mean fixation", formatMs(stats.meanFixationDuration)),
      ...timeToFirstRegionRow(statSet),
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
    /**
     * A URL study with no recordings has nothing to draw on.
     *
     * The stage is a cream placeholder printing the address; arming the
     * crosshair over it let regions be committed whose coordinates are
     * normalised against that placeholder rather than against any page a
     * participant will ever see. This screen already refuses to do that with
     * the region layer's own visibility switch — "a crosshair over a layer
     * nothing appears on is a control that lies" — and the same is true of a
     * crosshair over a layer with no stimulus behind it.
     */
    const noStimulusYet = study.stimulus.kind === "url" && !hasRecordings;
    const drawBlockedReason =
      "Run a session first — regions on a live page are anchored to the recorded viewport.";

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
            disabled: noStimulusYet,
            title: noStimulusYet ? drawBlockedReason : null,
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
        el(
          "p",
          { class: "muted" },
          noStimulusYet
            ? drawBlockedReason
            : "Draw a box over the stimulus to measure attention on it."
        )
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
            // The ordinal is the whole link between this table and the stage:
            // five hand-drawn boxes are five identical rectangles until each
            // one wears the number of its row.
            el("th", { class: "col-ordinal" }, el("span", { class: "sr-only" }, "Number")),
            el("th", {}, "Region"),
            // Without recordings these columns would be four columns of zeroes
            // presented as measurements.
            hasRecordings ? el("th", {}, "Seen") : null,
            hasRecordings ? el("th", {}, "Dwell") : null,
            // An acronym nobody outside the field reads on sight, in a tool
            // whose whole claim is that it explains its own numbers.
            hasRecordings
              ? el("th", {}, el("abbr", { title: "Time to first fixation" }, "TTFF"))
              : null,
            el("th", { class: "col-action" }, el("span", { class: "sr-only" }, "Remove"))
          )
        ),
        el(
          "tbody",
          {},
          ...aggregates.map((agg, index) =>
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
              el("td", { class: "cell-ordinal" }, el("span", { class: "ordinal-chip" }, String(index + 1))),
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
                      renderAoiBoxes(aoiLayer, aois);
                    }
                  },
                })
              ),
              // Each measure carries its own header name. Below 640px the row
              // becomes two lines — name above, measures below — and the thead
              // stops lining up with anything, so the label travels with the
              // number instead of sitting in a column header that is no longer
              // over it. See .cell-measure in styles.css.
              hasRecordings
                ? el("td", { class: "cell-measure", "data-label": "Seen" }, formatPercent(agg.hitRate))
                : null,
              hasRecordings
                ? el("td", { class: "cell-measure", "data-label": "Dwell" }, formatMs(agg.meanDwell))
                : null,
              hasRecordings
                ? el(
                    "td",
                    { class: "cell-measure", "data-label": "TTFF" },
                    agg.meanTimeToFirstFixation === null
                      ? "—"
                      : formatMs(agg.meanTimeToFirstFixation)
                  )
                : null,
              el("td", { class: "cell-action" }, aoiDeleteButton(agg.aoiId, agg.label))
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

  /**
   * Removing a region, as a labelled two-step control in its own table row.
   *
   * It used to be a vermillion × that faded in under passive hover anywhere on
   * the box, wired straight to a splice: the stimulus is most of the screen, so
   * a mouse merely crossing it parked a one-click, no-confirm delete under the
   * cursor. Regions are hand-drawn work that persists to storage, and every
   * other destructive action in this app — deleting a study, a recording, a
   * session mid-capture — is a labelled confirm. This one is now too.
   */
  function aoiDeleteButton(aoiId: string, label: string): HTMLButtonElement {
    const btn = confirmButton("Remove", "Really remove?", () => {
      const index = aois.findIndex((a) => a.id === aoiId);
      if (index >= 0) aois.splice(index, 1);
      onAoiChange();
    });
    btn.setAttribute("aria-label", `Remove region ${label}`);
    btn.dataset.key = `aoi-remove-${aoiId}`;
    return btn;
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

  /**
   * What the file covers, in the words the screen uses.
   *
   * Exports used to be handed `analysed` — every recording — while the screen
   * they were triggered from was reporting on `activeSet()`. A study whose
   * header pill read "4 of 5 recordings" and whose table read 75% produced a
   * file saying 0.600, and nothing in the file hinted that the two were
   * counting different people. Every CSV now covers exactly what the stage and
   * the tables cover, and says which set that was.
   */
  function exportScope(): ExportScope {
    const total = recordings.length;
    if (selected !== "all") {
      const one = activeSet()[0]?.recording.participant ?? "—";
      return { note: `Single participant: ${one} — 1 of ${total} recordings` };
    }
    const flagged = flaggedRecordings().length;
    const kept = aggregateSet().length;
    if (excludingLowSignal()) {
      return {
        note: `All participants, low-signal excluded — ${kept} of ${total} recordings (${flagged} below the quality threshold)`,
      };
    }
    if (flagged > 0) {
      return {
        note: `All participants, low-signal included — ${total} of ${total} recordings (${flagged} below the quality threshold)`,
      };
    }
    return { note: `All participants — ${total} of ${total} recordings` };
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
        aois,
        showAois,
        // The exported caption gets the same numbered axis the screen has, so
        // a figure in a deck can be read without the app open beside it.
        scale: heatScale,
      });
    }),
    exportItem("Fixations CSV", "export-fixations", () =>
      exportFixationsCsv(
        study,
        activeSet().map((a) => ({ recording: a.recording, data: a.fixations })),
        exportScope()
      )
    ),
    exportItem("Raw CSV", "export-raw", () =>
      exportRawCsv(
        study,
        activeSet().map((a) => a.recording),
        exportScope()
      )
    ),
    exportItem("AOI CSV", "export-aoi", () =>
      exportAoiCsv(
        study,
        aggregateAois(
          aois,
          activeSet().map((a) => a.aoiResults)
        ),
        activeSet().map((a) => ({ recording: a.recording, data: a.aoiResults })),
        exportScope()
      )
    ),
    // The one export that is an archive rather than a view of the screen, so it
    // stays whole: every recording, low-signal ones included and graded.
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

function statRow(
  label: string,
  value: string,
  grade?: QualityGrade,
  hint?: string
): HTMLElement {
  return el(
    "div",
    { class: "stat-row", title: hint ?? null },
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

  // Round dots, one fill each.
  //
  // These were 5px fillRects, on the argument that a square is indistinguishable
  // from a circle at this size. It is not: at 1× the cloud reads as pixel debris
  // rather than as samples, which is the wrong impression of the one view that
  // shows the tracker's raw output.
  //
  // One path for the whole cloud with a single fill at the end would be cheaper
  // still, but it would also flatten the thing this view is for: within a single
  // path, overlapping subpaths are painted once, so a dense cluster would come
  // out the same 0.35 alpha as a lone stray. The legend promises "density reads
  // as saturation", so each sample is filled on its own and the alpha stacks.
  // A minute of gaze is ~1800 samples per participant; this is a few
  // milliseconds on a debounced redraw, not a per-frame cost.
  //
  // Colours come from the shared categorical palette, which is also what the
  // legend keys — three unattributable dot clouds are worse than one.
  const radius = 2.5;
  const TAU = Math.PI * 2;
  recordings.forEach((recording, index) => {
    ctx.fillStyle = participantColour(index, 0.35);
    for (const p of recording.points) {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, radius, 0, TAU);
      ctx.fill();
    }
  });
}

/**
 * The region layer over the stimulus.
 *
 * Each box carries a permanent numbered badge and its name on hover. The name
 * alone was hover-only, which left the primary annotation layer with no
 * identity at all: five named regions rendered as five interchangeable hairline
 * rectangles, and matching a table row to a box meant hovering each one in
 * turn. The permanent chip that was removed for occluding the heat blob
 * underneath is not the answer either — so the identity that is always on is a
 * small numbered badge straddling the corner, outside the box, keyed to the
 * ordinal in the Areas of interest table; the full name still arrives on hover.
 */
function renderAoiBoxes(layer: HTMLElement, aois: Aoi[]): void {
  for (const node of Array.from(layer.querySelectorAll(".aoi-box:not(.is-drawing)"))) {
    node.remove();
  }

  aois.forEach((aoi, index) => {
    const box = el(
      "div",
      { class: "aoi-box", "data-aoi": aoi.id },
      el("span", { class: "aoi-ordinal", "aria-hidden": "true" }, String(index + 1)),
      el("span", { class: "aoi-label" }, aoi.label)
    );
    box.style.left = `${aoi.x * 100}%`;
    box.style.top = `${aoi.y * 100}%`;
    box.style.width = `${aoi.width * 100}%`;
    box.style.height = `${aoi.height * 100}%`;
    layer.append(box);
  });
}
