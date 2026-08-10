import { aggregateAois, analyseAois, type Aoi, type AoiResult } from "../analysis/aoi";
import { detectFixations, summarise, type Fixation } from "../analysis/fixations";
import { kernelRatio, renderHeatmap, type HeatPoint } from "../analysis/heatmap";
import {
  OVERLAY_LABELS,
  participantColour,
  type LegendBlur,
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
import { controlBandHeight, FIT_SCALE_FLOOR, fitStimulus } from "./record";
import {
  clear,
  confirmButton,
  el,
  formatMs,
  formatOnset,
  formatPercent,
  nextFrame,
  relativeDay,
} from "./dom";
import { legendElement } from "./legend";

type ViewMode = OverlayMode;

/** Which measured column the region table is ranked by. */
type AoiSortKey = "seen" | "dwell" | "ttff";

const VIEW_MODES: ViewMode[] = ["heat", "spotlight", "contour", "scanpath", "raw"];

/** The stage's own `min-height` in `.results-stage`. Below it the stylesheet
 * governs and a smaller `--stage-cap` would only be a number CSS ignores. */
export const STAGE_MIN_HEIGHT = 320;

/** The three measurements the stage's ceiling is arithmetic over. */
export interface StageChrome {
  /** Viewport height, in CSS pixels. */
  viewport: number;
  /** Document-space distance from the top of the page to the top of the stage
   * — everything the column puts above the artboard. */
  above: number;
  /** Everything below the stage inside its column: the legend that decodes the
   * overlay, the empty-state strip, and the margins that join them. */
  under: number;
}

/**
 * How tall the artboard is allowed to be, so that it and the legend that
 * decodes it both end at the fold.
 *
 * Pure and exported because this is a contract, not a preference, and because
 * the only other way to check it is to open a browser at a particular size.
 *
 * It used to be the larger of two terms. The first is the one below. The second
 * was `rail height − stage bar − under`, bounded by `viewport − above`, and it
 * existed to stop the row's slack being stranded as cream when the rail is
 * taller than the column beside it. That bound omitted `under`, and the
 * omission was the whole bug: at 1440×900 it returned 614 where the honest
 * number is 551, and the 63px difference was the legend — title strip visible,
 * the 0ms / 540ms / ≥1.1s ticks that make the picture readable pushed off the
 * screen.
 *
 * Only above 1180px, where the column is a column. Below that `.results-main` is
 * `display: contents`, so this term never applied there and never did — the
 * legend sits far below the fold at those widths for a different reason
 * entirely: it is ordered *after* the rail (see the max-width: 1180px block in
 * styles.css), which puts ~780px of sidebar between the picture and its scale.
 * That is a deliberate trade made to keep the view switcher off the bottom of a
 * 1024×800 screen, and no arithmetic here can undo it.
 *
 * Adding `under` back to that bound does not merely correct it, it retires it:
 * `max(fold, min(rail, fold))` is `fold` for every possible rail height. The two
 * goals are in direct conflict — the stage cannot both stop at the fold and grow
 * past it to match a taller rail — and the fold wins, because the legend is what
 * makes the overlay a measurement rather than a picture. What that costs is
 * visible: on a study whose rail runs taller than its column, the column ends at
 * the fold and the rail runs on below it, leaving that much ground beside the
 * rail's tail. It is ground the reader only reaches by scrolling, which is
 * exactly where it is cheapest.
 */
export function stageCap(chrome: StageChrome): number {
  return Math.max(
    STAGE_MIN_HEIGHT,
    Math.round(chrome.viewport - chrome.above - chrome.under)
  );
}

/**
 * The share of the artboard a contained figure has to fall below before the
 * screen offers fit-width on its own.
 *
 * A stimulus that fills three quarters of the stage's width is fine contained;
 * below that it is being shown well under its designed size with the rest of the
 * artboard as ground, which is the state the affordance exists for.
 */
export const AUTO_FIT_RATIO = 0.75;

/** Whether a figure of this width is starved inside a stage of that width. Pure
 * so the threshold can be checked against measured geometry without a browser —
 * the reason it never fired for so long is that it was only ever *called* with
 * geometry that was not final. */
export function shouldFitWidth(figureWidth: number, stageWidth: number): boolean {
  if (stageWidth <= 0 || figureWidth <= 0) return false;
  return figureWidth / stageWidth < AUTO_FIT_RATIO;
}

/**
 * Who the results screen is reporting on — computed once, read by everything
 * that states a scope.
 *
 * This exists because the screen contradicted itself. In Scanpath view the
 * header pill said "4 of 5 recordings", the rail said "Summary — P01", the
 * region table under them printed the four-participant aggregate, all four
 * scoped export rows said "4 of 5 recordings", and the PNG they produced held
 * one person's path over a caption reading "All participants". Four statements,
 * three different denominators, one screen. Each was independently correct
 * about a different set, because each computed its own — `paintCountPill` read
 * `aggregateSet()`, `scopeSummary` read `selected`, the PNG read a literal
 * string, and only the rail knew a scanpath is one person's path.
 *
 * So the set is decided in exactly one place ({@link reportedScope}) and the
 * four sentences below are the only wordings of it. They differ in register —
 * a pill is not a file's provenance line — but never in who they are counting,
 * and the tests assert that.
 */
export interface ReportedScope {
  /** The participants the stage, the rail, the region table and the exports
   * all cover, in draw order. */
  participants: string[];
  /** Recordings in the study, whatever is being reported on. */
  total: number;
  /** True when the set narrowed to one person because the *view* is per-person,
   * rather than because a participant was picked. Only the file note prints the
   * reason, but it decides the wording there. */
  perView: boolean;
  /** Recordings in the study below the quality threshold. */
  flagged: number;
  /** Whether those flagged recordings are being held out of the aggregate. */
  excludingFlagged: boolean;
}

/** One person out of several: the case every statement here used to get
 * differently. A one-recording study is not "solo" — there is nobody to be
 * distinguished from. */
function isSolo(scope: ReportedScope): boolean {
  return scope.participants.length === 1 && scope.total > 1;
}

function recordingWord(total: number): string {
  return total === 1 ? "recording" : "recordings";
}

/**
 * The header pill: the count, and whose it is.
 *
 * It read `aggregateSet().length` — a fact about the study rather than about
 * the screen — so it kept printing the aggregate's denominator while the rail
 * beside it, the table below it and the figure between them were describing one
 * person.
 */
export function scopePill(scope: ReportedScope): { text: string; title: string } {
  if (isSolo(scope)) {
    return {
      text: `${scope.participants[0]} — 1 of ${scope.total} ${recordingWord(scope.total)}`,
      title: scope.perView
        ? "A scanpath is one person's path, so this screen is reporting on one recording"
        : "One participant is selected, so this screen is reporting on one recording",
    };
  }
  if (scope.participants.length < scope.total) {
    return {
      text: `${scope.participants.length} of ${scope.total} recordings`,
      title: "Low-signal recordings are excluded from the aggregate",
    };
  }
  return { text: `${scope.total} ${recordingWord(scope.total)}`, title: "" };
}

/** The scope as one phrase, for the export menu's rows, the region table's
 * caption and the printed page. */
export function scopeSentence(scope: ReportedScope): string {
  if (isSolo(scope)) {
    return `${scope.participants[0]} only — 1 of ${scope.total} ${recordingWord(scope.total)}`;
  }
  if (scope.participants.length === scope.total) {
    return `all ${scope.total} ${recordingWord(scope.total)}`;
  }
  return `${scope.participants.length} of ${scope.total} recordings`;
}

/**
 * The provenance line written into every exported CSV above the header row.
 *
 * A file outlives the screen it came from, so this is the one wording that
 * carries *why* the set is what it is — including the fact that a scanpath is
 * per-person, which is otherwise invisible once the file is open in a
 * spreadsheet.
 */
export function scopeNote(scope: ReportedScope): string {
  if (isSolo(scope)) {
    const why = scope.perView ? " (scanpath view is per-person)" : "";
    return `Single participant: ${scope.participants[0]} — 1 of ${scope.total} ${recordingWord(scope.total)}${why}`;
  }
  if (scope.excludingFlagged) {
    return `All participants, low-signal excluded — ${scope.participants.length} of ${scope.total} recordings (${scope.flagged} below the quality threshold)`;
  }
  if (scope.flagged > 0) {
    return `All participants, low-signal included — ${scope.total} of ${scope.total} recordings (${scope.flagged} below the quality threshold)`;
  }
  return `All participants — ${scope.total} of ${scope.total} ${recordingWord(scope.total)}`;
}

/**
 * The scope stamped into the exported PNG's caption band.
 *
 * It was the literal string "All participants" on a file whose pixels held one
 * person's scanpath — the single worst statement on the screen, because the PNG
 * is the artifact that leaves the tool and gets pasted into a deck with nothing
 * beside it to check against.
 */
export function scopeCaption(scope: ReportedScope): string {
  if (isSolo(scope)) return scope.participants[0];
  if (scope.excludingFlagged) return "All participants, low-signal excluded";
  return "All participants";
}

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
  /**
   * The participant whose calibration is cached for this sitting, if any.
   *
   * Passed in rather than read here: the cache is session storage owned by the
   * app shell (see CALIBRATION_KEY in main.ts), and a second module reaching for
   * the same key by name is how two readers of one store drift apart. It is a
   * fact worth showing on an empty results screen, because it changes what the
   * next session costs.
   */
  reusableCalibration?: string | null;
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
  /** How the region table is ordered, or null for the order the regions were
   * drawn in — which is the order their badges are numbered in. */
  let aoiSort: { key: AoiSortKey; dir: 1 | -1 } | null = null;

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
   * stage it reads as what it is: a footer bar belonging to the stage.
   *
   * The action that used to sit in it has moved into the rail beside it, which
   * now holds the empty state's "what happens next" (see buildSidebar). The
   * rail is where the primary control of this screen lives once there are
   * recordings, so putting it there before there are any is what makes the two
   * skeletons the same screen — and one Run session button per screen beats
   * two. */
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
      emptyStrip = el("div", { class: "stage-empty" }, el("p", {}, emptyLine));
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
        )
      )
    );
  }
  figure.append(overlay, aoiLayer);
  /** The scrolling box, so the stage's own controls do not scroll with it.
   *
   * Fit-width makes the figure taller than the stage, and an absolutely
   * positioned control inside a scroll container scrolls with its content — so
   * with the scroll on the stage itself, the Fit toggle that put the stage into
   * this state travelled off the top of it and could not be used to leave.
   * Scrolling one layer in gives the tools a fixed frame to sit in. */
  const scroller = el("div", { class: "stage-scroll" }, figure);
  stage.append(scroller);

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
  // A wireframe is wider than it is tall and a phone is not; there is no honest
  // way to fix that in CSS without rotating the figure, which would put every
  // pointer coordinate in the region layer out of register with the picture.
  // Saying so costs one line — but only where the advice can be taken. It
  // rendered unconditionally, so a 1440px desktop window was told to rotate a
  // monitor.
  const lightboxHint = el("p", { class: "lightbox-hint", hidden: true }, "Rotate for a wider view");
  /**
   * What the expanded figure is a picture of.
   *
   * Expand is the view that gets screenshotted for a deck, and it carried no
   * study name, no participant count and — because the legend stayed behind in
   * the column — no colour scale: a 1064×831 heatmap a reader cannot ask "is
   * dark red 300ms or 3s, and is this one person or four" of. Both facts follow
   * the figure in now, in the same words the rest of the screen uses.
   */
  const lightboxScope = el("p", { class: "lightbox-scope" });
  const canRotate = (): boolean =>
    typeof window.matchMedia === "function" &&
    window.matchMedia("(orientation: portrait) and (max-width: 700px)").matches;
  const lightbox = el(
    "dialog",
    { class: "figure-lightbox", "aria-label": `${study.name}, full screen` },
    // The bar is a row of the dialog, not a scrim over the picture. It used to
    // be absolutely positioned across the top with a cream gradient behind it,
    // which put 44px of wash over the top band of the very stimulus "Expand"
    // exists to show closely — on a wireframe that band is the nav, and usually
    // region 1.
    el(
      "div",
      { class: "lightbox-bar" },
      lightboxScope,
      lightboxHint,
      el(
        "button",
        { class: "btn btn-small", type: "button", onclick: () => lightbox.close() },
        "Close"
      )
    ),
    lightboxBody
  );
  const zoomButton = el(
    "button",
    {
      class: "btn btn-small stage-zoom",
      type: "button",
      onclick: () => {
        lightboxBody.append(figure);
        // The legend travels with the figure it decodes, as the last row of the
        // dialog — the same element, so there is no second legend that could
        // key a different scale from the one on the stage. It goes back to the
        // column in the `close` handler below.
        if (hasRecordings) lightbox.append(legend);
        lightboxScope.textContent = hasRecordings
          ? `${study.name} — ${scopeSummary()}`
          : study.name;
        lightboxHint.hidden = !canRotate();
        lightbox.showModal();
        // The figure's box has just changed by a factor of four; the canvas
        // backing store has to follow it or the overlay is an upscaled blur.
        void draw();
      },
    },
    "Expand"
  );

  /**
   * Fit-height or fit-width, because one fixed fit is not enough.
   *
   * The figure is capped at 78vh, so a portrait stimulus — a full-page
   * screenshot, which is most of what gets uploaded — becomes height-bound and
   * gives its width away: measured at 1440×900, a 976px stage holding a 562px
   * figure, 42.6% of the primary analysis surface empty ground. Fit-width
   * releases the height cap, gives the figure the stage's full width and lets
   * the stage scroll on Y, which is what every annotation tool offers and what
   * makes a full-page screenshot readable at all. The overlay and the region
   * layer are inset to the figure in CSS, so both fits are exact — nothing is
   * measured and cached, and the two modes cannot disagree about where a blob
   * belongs.
   */
  let fitWidth = false;
  /** Set once the operator picks a fit, so the automatic choice below stops
   * second-guessing them on the next resize. */
  let fitChosen = false;
  const fitButton = el(
    "button",
    {
      class: "btn btn-small stage-fit",
      type: "button",
      "aria-pressed": "false",
      title: "Show the stimulus at the stage's full width and scroll it",
    },
    "Fit width"
  );
  /**
   * A toggle labelled by its state, not by its destination.
   *
   * The label used to swap to "Fit height" the moment fit-width came on, while
   * `aria-pressed` — correctly — went to "true". So the visible word and the
   * announced state were inverses: the screen read "Fit height" over a stimulus
   * that was width-fitted and scrolling, and a first-time operator reads that as
   * a statement about the stage rather than an offer. The text is now constant
   * and the on/off rides on `.is-active` and `aria-pressed`, which is how
   * `+ Draw` and `With recordings` already work on this screen — and it makes
   * the announced state and the printed one the same claim.
   */
  const setFit = (next: boolean): void => {
    fitWidth = next;
    figure.classList.toggle("is-fit-width", next);
    stage.classList.toggle("is-scrolling", next);
    fitButton.classList.toggle("is-active", next);
    fitButton.title = next
      ? "Showing the stimulus at the stage's full width. Turn off to fit the whole stimulus in the stage."
      : "Show the stimulus at the stage's full width and scroll it";
    fitButton.setAttribute("aria-pressed", next ? "true" : "false");
    markStageClip();
    void draw();
  };
  fitButton.addEventListener("click", () => {
    fitChosen = true;
    setFit(!fitWidth);
  });

  /**
   * Picks the fit the stimulus needs, once, before anyone has expressed a
   * preference. See {@link shouldFitWidth} for the threshold.
   *
   * The threshold was right and the measurement was not. This ran once, on the
   * line after the first `draw()`, and at that moment the legend under the stage
   * had only just been filled in: `--stage-cap` still held the value computed
   * against an empty legend, which is 128px too generous, so the height-bound
   * figure was measured 128px too wide. Sampled at 80ms intervals on a fresh
   * load, `aria-pressed` was "false" in every sample at 1180, 1100 and 1024
   * while the settled ratios were 0.626, 0.675 and 0.731 — all below the
   * threshold, none of them ever tested. At 1180 the pre-cap ratio was 0.757:
   * it missed by seven thousandths of the number it was comparing against.
   *
   * So the caller measures a settled stage (see the end of this function), and
   * this asserts it rather than trusting the call site: with no legend measured
   * yet there is nothing to fit against, and re-running later would fight an
   * operator who has since chosen.
   */
  let autoFitDone = false;
  const autoFit = (): void => {
    if (autoFitDone || fitChosen || !stimulusImage) return;
    autoFitDone = true;
    if (shouldFitWidth(figure.getBoundingClientRect().width, scroller.clientWidth)) {
      setFit(true);
    }
  };

  // Both are absolutely positioned or in the top layer, so neither takes part
  // in the stage's centring of the figure. The `close` event covers Esc and the
  // backdrop as well as the button.
  lightbox.addEventListener("close", () => {
    scroller.append(figure);
    // Back to the foot of the column, which is where `chromeUnderStage`
    // measures it and where the stage's ceiling is bought from.
    mainColumn.append(legend);
    void draw();
  });
  // The tools group holds only what applies. A URL study's figure *is* the
  // stage, so there is nothing to fit — and before its first recording there is
  // nothing to expand either: the stage is a placeholder printing an address,
  // and a full-screen view of it promises a picture there is no picture for.
  const stageTools =
    stimulusImage || hasRecordings
      ? el("div", { class: "stage-tools" }, stimulusImage ? fitButton : null, zoomButton)
      : null;
  /**
   * The tools sit in a bar above the stage, not on it.
   *
   * They were pinned to the stage's top-right corner, and measurement at twelve
   * widths from 1440 down to 390 found them intersecting the figure in all
   * twelve — 1,079 to 5,313 px² of the stimulus underneath a pair of buttons,
   * and at every width one of those buttons was over the first region box. The
   * spotlight view re-tinted them to survive the mask, which concedes the
   * problem rather than fixing it. This app's one claim is "here is where they
   * looked at this region"; furniture parked on the region is the one kind of
   * chrome it cannot afford. Every annotation tool puts stage controls in a
   * toolbar above the canvas for exactly this reason.
   *
   * The bar's left half is not filler: it says what the measured rect is, which
   * is the number every figure on this screen is normalised against.
   */
  const stageBar = stageTools
    ? el(
        "div",
        { class: "stage-bar" },
        stimulusImage
          ? el(
              "p",
              { class: "stage-meta" },
              study.stimulus.kind === "image"
                ? `${study.stimulus.width} × ${study.stimulus.height} px`
                : ""
            )
          : null,
        stageTools
      )
    : null;
  stage.append(lightbox);

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
  const mainColumn = el("div", { class: "results-main" }, stageBar, stage, emptyStrip, legend);
  const layout = el("div", { class: "results-layout" }, mainColumn, rail, dataBlock);

  /**
   * The stage's ceiling, measured off the page instead of asserted about it.
   *
   * `--stage-cap` used to be `100vh - 450px` — a constant standing in for "the
   * chrome above the stage, the legend below it, and the head of the region
   * table". The chrome does not cost 450px, and the difference came out of the
   * stimulus: at 1440×900 the cap held the stage to 450px while the figure was
   * 631, so a third of the wireframe was inside a scroller with no scrollbar,
   * and the region table the reserve was bought for still opened below the fold.
   *
   * What replaces it is read from the live layout — see {@link stageCap} for
   * the arithmetic and for why the second term this function used to carry, the
   * one that grew the stage into a taller rail's slack, is gone.
   *
   * Nothing here depends on the stage's own height, so this settles in one pass
   * rather than converging: the stage's distance from the top of the document is
   * set by the bar above it, and the chrome below it is the legend, whose width
   * is the column's and not the stage's.
   */
  const chromeUnderStage = (): number =>
    [emptyStrip, legend].reduce((total, node) => {
      // `isConnected` is not the whole question — the legend travels into the
      // lightbox (see zoomButton) and is still connected there, in the top
      // layer, where it costs the column nothing. Nor is "is it inside the main
      // column", because the dialog is a child of the stage and so still inside
      // it. What disqualifies a node is being in the dialog; without that test,
      // expanding the figure charged the stage 114px of chrome that had left it.
      if (!node || !node.isConnected || lightbox.contains(node)) return total;
      const box = node.getBoundingClientRect();
      if (box.height === 0) return total;
      return total + box.height + (parseFloat(getComputedStyle(node).marginTop) || 0);
    }, 0);

  let appliedCap = 0;
  const fitStageCap = (): void => {
    if (!stage.isConnected) return;
    const stageBox = stage.getBoundingClientRect();
    const next = stageCap({
      viewport: window.innerHeight,
      // Document-space, so a page scrolled down does not report a shorter
      // reserve than the one the operator actually landed on.
      above: stageBox.top + window.scrollY,
      under: chromeUnderStage(),
    });
    if (next === appliedCap) return;
    appliedCap = next;
    layout.style.setProperty("--stage-cap", `${next}px`);
  };

  /**
   * Whether the artboard is holding anything back, which is what the fade on its
   * bottom edge is drawn from (see `.results-stage.is-clipped` in styles.css).
   * Cleared on arrival at the end so the cue never marks an edge that is not one.
   */
  const markStageClip = (): void => {
    const clipped =
      stage.classList.contains("is-scrolling") &&
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 1;
    stage.classList.toggle("is-clipped", clipped);
  };
  scroller.addEventListener("scroll", markStageClip, { passive: true });

  // Export is the whole point of a research tool, so it is a persistent
  // toolbar action rather than the last block of a sidebar — where it sat below
  // the fold of an invisible nested scroller. One button, not five: the header
  // read as six equivalent pills, of which the first was not a control at all.
  /**
   * A disclosure, and it now says so.
   *
   * It carried `role="menu"`, `role="menuitem"` and `aria-haspopup="true"` while
   * implementing none of the menu pattern: no arrow keys, no roving focus, no
   * focus moved into the panel on open, and Tab walking straight past it into
   * the page behind. A promised keyboard contract that does not exist is worse
   * than an honest simpler one — a screen-reader user told "menu" presses Down
   * and nothing happens. So the roles are gone: this is a button that shows a
   * group of buttons, which is exactly what `aria-expanded` on a plain button
   * describes, and Tab through them works because they are ordinary buttons in
   * document order. Escape closes and returns focus (see onDocumentKey); moving
   * focus out of the group closes it too, so Tab past the last item cannot leave
   * an open panel floating behind the cursor.
   */
  const exportMenu = el("div", { class: "menu", hidden: true });
  const exportToggle = el(
    "button",
    {
      class: "btn btn-small",
      type: "button",
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
   * 4" here makes that note a confirmation instead of a correction — and it now
   * takes that count from {@link reportedScope}, the same value the rail, the
   * region table and the export rows are worded from, so it also stops
   * disagreeing with them in the per-person views.
   */
  const countPill = el("span", { class: "pill pill-count" });
  function paintCountPill(): void {
    const pill = scopePill(reportedScope());
    countPill.textContent = pill.text;
    countPill.title = pill.title;
  }

  /**
   * One export, with what it will actually contain printed under its name.
   *
   * The menu was five identical rows, and one of them is a different kind of
   * file: the four view exports are scoped to whatever the screen is reporting
   * on, while Session JSON is deliberately the whole archive — low-signal
   * recordings included. A researcher exporting from a header that reads "4 of
   * 5 recordings" had no way to know that one of these five files contains
   * five. The scope is live, repainted every time the menu opens.
   */
  function exportItem(label: string, key: string, run: () => void): HTMLButtonElement {
    return el(
      "button",
      {
        class: "menu-item",
        type: "button",
        "data-key": key,
        onclick: () => {
          setExportOpen(false);
          run();
        },
      },
      el("span", { class: "menu-item-label" }, label),
      el("span", { class: "menu-item-note" })
    );
  }

  /** The scoped exports' coverage, in the same counting the header pill uses.
   * It read `selected` alone, so in the per-person views it promised four
   * participants' data under a button that produced one person's picture. */
  function scopeSummary(): string {
    return scopeSentence(reportedScope());
  }

  function paintExportScopes(): void {
    const total = recordings.length;
    const plural = total === 1 ? "" : "s";
    const scoped = scopeSummary();
    for (const item of Array.from(exportMenu.querySelectorAll<HTMLElement>(".menu-item"))) {
      const note = item.querySelector<HTMLElement>(".menu-item-note");
      if (!note) continue;
      note.textContent =
        item.dataset.key === "export-json"
          ? `archive — all ${total} recording${plural}, low signal included`
          : scoped;
    }
  }

  function setExportOpen(open: boolean): void {
    if (open) paintExportScopes();
    exportMenu.hidden = !open;
    exportToggle.setAttribute("aria-expanded", open ? "true" : "false");
    exportToggle.classList.toggle("is-active", open);
  }

  // A menu that cannot be dismissed by clicking away from it is a trap.
  const onDocumentPointer = (event: Event) => {
    if (!exportBar.contains(event.target as Node)) setExportOpen(false);
  };
  // The keyboard equivalent of clicking away: tabbing out of the group closes
  // it, so the panel cannot stay open behind a cursor that has left it.
  exportBar.addEventListener("focusout", (event: FocusEvent) => {
    const next = event.relatedTarget as Node | null;
    if (next && exportBar.contains(next)) return;
    setExportOpen(false);
  });
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

  /**
   * True when the *view* has narrowed the selection to one person.
   *
   * A combined scanpath across participants would be a path nobody took, so the
   * renderer has always drawn the first of the selected set and the rail has
   * always said so. What was missing is that nothing else on the screen knew:
   * the pill, the region table and all four export rows went on counting the
   * whole aggregate. This predicate is the one place that fact lives now.
   */
  const perViewSolo = (): boolean =>
    mode === "scanpath" && selected === "all" && aggregateSet().length > 1;

  /** The recordings every number, picture, table and file on this screen
   * covers. Identical to `activeSet()` except in the per-person views. */
  const reportedSet = (): AnalysedRecording[] => {
    const set = activeSet();
    return perViewSolo() ? set.slice(0, 1) : set;
  };

  const reportedScope = (): ReportedScope => ({
    participants: reportedSet().map((a) => a.recording.participant),
    total: recordings.length,
    perView: perViewSolo(),
    flagged: flaggedRecordings().length,
    excludingFlagged: excludingLowSignal(),
  });

  /**
   * How far to blur the attention field for the set on the stage.
   *
   * The ratio was the literal 0.055 for every study and every selection — see
   * {@link kernelRatio} for what that cost. Both terms are averaged over the
   * recordings actually being drawn, because that is the set whose uncertainty
   * the picture is claiming: a selection of one 48px-error recording and one
   * 184px one is genuinely less certain than either alone.
   *
   * `stimulusRect` is the stimulus as that participant saw it, in the same CSS
   * pixels the validation error was measured in, so the two divide cleanly. The
   * blur reported back is what was *drawn* — after the clamp — rather than what
   * was asked for, because the caption is a statement about the picture.
   */
  const heatKernel = (
    set: AnalysedRecording[]
  ): { ratio: number; blur: LegendBlur | null } => {
    const error = averageOf(
      set
        .map((a) => a.recording.quality.validationError)
        .filter((v): v is number => v !== null)
    );
    const minDim = averageOf(
      set
        .map((a) => {
          const q = a.recording.quality;
          const width = q.stimulusRect.width || q.viewportWidth;
          const height = q.stimulusRect.height || q.viewportHeight;
          return Math.min(width, height);
        })
        .filter((d) => d > 0)
    );
    const ratio = kernelRatio(error, minDim ?? 0);
    // Reported whenever there is a rect to have measured it against, including
    // when the error itself was never captured: the blur is a fact about the
    // picture either way, and a caption that goes quiet exactly when the number
    // is least certain is the wrong way round.
    if (minDim === null || minDim <= 0) return { ratio, blur: null };
    // σ is half the splat radius — see renderHeatmap — and the radius is the
    // ratio of the smaller dimension.
    const sigma = (ratio * minDim) / 2;
    return { ratio, blur: { degrees: pxToDegrees(sigma), pixels: sigma } };
  };

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

    // The same set the pill, the rail, the region table and the exports name:
    // in the per-person views that is one recording, everywhere else it is the
    // whole selection. Reading it from one place is what stops the picture and
    // the sentence under it describing different people.
    const set = reportedSet();

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
      renderRawPoints(
        overlay,
        set.map((a) => a.recording),
        dpr
      );
    } else {
      const points: HeatPoint[] = [];
      for (const a of set) {
        for (const f of a.fixations) points.push({ x: f.x, y: f.y, weight: f.duration });
      }
      const kernel = heatKernel(set);
      const ceiling = renderHeatmap(overlay, points, {
        style: mode,
        radiusRatio: kernel.ratio,
      });
      // Spotlight is a mask: the same field drives it, but what it encodes is
      // "revealed or not", so a millisecond axis under it would name a quantity
      // the picture does not carry. How far the field was blurred is a fact
      // about the mask all the same, so that part of the scale survives.
      heatScale =
        mode === "spotlight"
          ? kernel.blur && { ceiling: 0, blur: kernel.blur }
          : ceiling > 0
            ? { ceiling, blur: kernel.blur }
            : null;
    }

    // Spotlight dims the stage to near-black, which a deep-teal region box and
    // its dark label chip simply disappear into. The class flips both to the
    // cream side of the palette for as long as the mask is up.
    stage.classList.toggle("stage--spotlight", mode === "spotlight");

    // `set` is already only what is on the stage — in the per-person views that
    // is one recording — so the key names exactly what was drawn. With nothing
    // on the stage there is nothing to key: a legend for an overlay that is not
    // drawn is a caption for a missing figure.
    clear(legend);
    if (hasRecordings) {
      legend.append(
        legendElement(mode, set.map((a) => a.recording.participant), heatScale)
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
    // Below one gaze sample the honest reading is "sooner than this can be
    // timed", not "0ms" — see formatOnset. The fix that put a real quantity in
    // this row changed *what* was measured; it left the formatting floor, so a
    // participant already on a region at t=0 still produced a headline zero.
    const onset = mean === null ? null : formatOnset(mean);
    const scopeNote =
      mean === null
        ? "No participant in this selection fixated any region."
        : `Mean over the ${found} of ${set.length} recording${set.length === 1 ? "" : "s"} whose gaze reached a region.`;
    return [
      statRow(
        "Time to first region",
        onset === null ? "—" : onset.label,
        undefined,
        onset?.note ? `${onset.note} ${scopeNote}` : scopeNote
      ),
    ];
  }

  /**
   * The study's own facts, for the rail of a screen with nothing measured yet.
   *
   * The on-screen scale is quoted from `fitStimulus` — the same function the
   * recording stage decides with and the setup form already prints — rather than
   * recomputed here. A briefing that disagreed with what the participant is
   * about to see would be worse than no briefing.
   */
  function studyBriefing(): HTMLElement[] {
    const rows: HTMLElement[] = [el("h3", {}, "This study")];

    if (study.stimulus.kind === "image") {
      const { width, height } = study.stimulus;
      rows.push(statRow("Stimulus", `${width} × ${height}`));
      const fit = fitStimulus(
        { width, height },
        {
          width: window.innerWidth,
          height: Math.max(1, window.innerHeight - controlBandHeight(window.innerHeight)),
        }
      );
      const percent = Math.round(fit.scale * 100);
      rows.push(
        statRow(
          "On this screen",
          `${percent}%`,
          fit.scale < FIT_SCALE_FLOOR ? "warn" : undefined,
          fit.scale < FIT_SCALE_FLOOR
            ? "Small labels may be unreadable at this size, which puts the finding in doubt rather than only the look of it."
            : fit.mode === "width"
              ? "Taller than the recording stage, so it is shown at full width and scrolled."
              : "Fits the recording stage whole."
        )
      );
    } else {
      rows.push(statRow("Stimulus", "Live page"));
    }

    rows.push(
      statRow("Duration", study.duration > 0 ? `${study.duration}s` : "Manual stop")
    );

    rows.push(
      el(
        "p",
        { class: "brief-task" },
        study.task
          ? `“${study.task}”`
          : "No task set — a free-viewing heatmap mostly shows where the biggest image is."
      )
    );

    // Only when it is true, and named, because "reuse a calibration" is only
    // reassuring if you can see whose.
    if (options.reusableCalibration) {
      rows.push(
        el(
          "p",
          { class: "note" },
          `${options.reusableCalibration}'s calibration from this sitting can be reused, so the next session can skip straight to the task.`
        )
      );
    }

    return rows;
  }

  function buildSidebar(): void {
    /**
     * The empty state, in the rail rather than instead of it.
     *
     * The rail used to be collapsed away entirely before the first recording,
     * which cured a 320×517px column of empty cream by producing something
     * worse: three full-width strips — a 1320×320 stage, a 1320×126 "rail" and
     * a 1320×104 table — so the screen a first-time visitor meets is a
     * different skeleton from the one their first recording produces, and the
     * structure has to be re-learned the moment it lands. The rail is kept and
     * given something worth its width: what happens next, and the button that
     * does it. Same two-track grid, same rail, before and after.
     */
    if (!hasRecordings) {
      // A URL study has no stimulus to draw on until a session has recorded one
      // — the same condition aoiSection computes before disabling "+ Draw". A
      // visibility switch over a layer the app simultaneously forbids you to
      // put anything into is the control that lies, one level up.
      const noStimulusYet = study.stimulus.kind === "url";
      sidebar.append(
        el("h3", {}, "What happens next"),
        el(
          "ol",
          { class: "next-steps" },
          el("li", {}, "Run a session: calibrate, show the task, record the pass."),
          el(
            "li",
            {},
            noStimulusYet
              ? "Draw regions afterwards — on a live page they are anchored to the recorded viewport."
              : "Draw regions on the stimulus. Regions drawn now apply to every session run afterwards."
          ),
          el("li", {}, "Heatmap, spotlight, contour and scanpath views arrive with the first recording.")
        ),
        runAction(),
        ...(noStimulusYet ? [] : [regionToggle()]),
        // The rail's other half, before there is any data to put in it.
        //
        // Kept at zero recordings so the empty screen has the same skeleton as
        // the populated one — but three bullets and a button left a measured
        // 385px of empty cream under them, about 40% of the screen unused. What
        // belongs there is the briefing: what this study will actually show a
        // participant, at what size, with what task and for how long. Every one
        // of those is a decision someone can still change for free right now,
        // and impossible to change once recordings exist.
        ...studyBriefing()
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

    // Paper has no dropdown and no pill, so the printed page states the scope
    // in the same words the export menu does rather than in a fourth wording of
    // its own.
    const scopeName = capitalise(scopeSummary());

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
    const soloScanpath = perViewSolo();
    const statSet = reportedSet();

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

    /**
     * The regions, over the people the rest of the screen is reporting on.
     *
     * This read `activeSet()`, which in Scanpath view is the whole aggregate
     * while everything around it — the pill, the picture, the rail, the export
     * rows — is one person. It printed "Seen 50%" under a heading that said
     * P01: 50% is two people out of four, a reading a single participant cannot
     * produce, so the row was not merely differently scoped but arithmetically
     * impossible for the scope stated above it.
     *
     * The scope is also stated rather than left to be inferred: the table is the
     * block most likely to be pasted into a deck on its own, and it is the one
     * that used to carry no scope at all.
     */
    if (hasRecordings) {
      section.append(el("p", { class: "table-scope" }, `Covering ${scopeSummary()}.`));
    }

    const aggregates = aggregateAois(
      aois,
      reportedSet().map((a) => a.aoiResults)
    );

    // The ranking *is* the finding, and a column of unaligned decimals does not
    // carry it: "2.7s / 4.9s / 0ms / 17.3s / 1.8s" left the reader adding up
    // digits to notice that one region is six times everything else. Each
    // measure gets a bar scaled to the largest value in the current selection —
    // the same device Tobii, Hotjar and Maze all put in this cell — so the shape
    // of the result is legible before any number is read.
    const maxHitRate = Math.max(...aggregates.map((a) => a.hitRate), 0);
    const maxDwell = Math.max(
      ...aggregates.map((a) => (a.hitRate > 0 ? a.meanDwell : 0)),
      0
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
            hasRecordings ? sortableHeader("Seen", "seen") : null,
            hasRecordings ? sortableHeader("Dwell", "dwell") : null,
            // An acronym nobody outside the field reads on sight, in a tool
            // whose whole claim is that it explains its own numbers.
            hasRecordings
              ? sortableHeader(
                  el("abbr", { title: "Time to first fixation" }, "TTFF"),
                  "ttff"
                )
              : null,
            el("th", { class: "col-action" }, el("span", { class: "sr-only" }, "Remove"))
          )
        ),
        el(
          "tbody",
          {},
          ...sortRows(aggregates).map(({ agg, index }) =>
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
                  // The underline is the affordance, so it has to be the width
                  // of the name: at `width: 100%` six regions drew six ~270px
                  // hairlines floating out past the end of every word, the
                  // heaviest lines in a table of numbers. `field-sizing` does
                  // this natively where it exists; `size` is the fallback that
                  // has worked since forever, and both are kept in step as the
                  // name is typed.
                  size: String(Math.max(8, agg.label.length)),
                  "aria-label": "Region name",
                  "data-key": `aoi-name-${agg.aoiId}`,
                  oninput: (event: Event) => {
                    const input = event.target as HTMLInputElement;
                    input.size = Math.max(8, input.value.length);
                  },
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
                ? measureCell("Seen", formatPercent(agg.hitRate), agg.hitRate, maxHitRate)
                : null,
              // A region nobody fixated used to print three different nulls
              // across one row — "0%", "0ms", "—". Only the hit rate is a real
              // zero there; a mean dwell over no fixations is not a duration of
              // zero, it is an absence, and it is written like the TTFF beside
              // it.
              hasRecordings
                ? agg.hitRate > 0
                  ? measureCell("Dwell", formatMs(agg.meanDwell), agg.meanDwell, maxDwell)
                  : measureCell("Dwell", "—", 0, maxDwell)
                : null,
              hasRecordings ? ttffCell(agg.meanTimeToFirstFixation) : null,
              el("td", { class: "cell-action" }, aoiDeleteButton(agg.aoiId, agg.label))
            )
          )
        )
      )
    );
    return section;
  }

  /**
   * A sortable column head.
   *
   * The table had no sort at all, so "which region won" was a question you
   * answered by reading six rows of decimals. One click ranks by that measure —
   * biggest first for the two "more is more" columns, soonest first for TTFF,
   * because that is the direction each one is interesting in — and a second
   * click reverses it. A third returns the table to drawing order, which is the
   * order the badges on the stimulus are numbered in and therefore the only
   * order that is not an opinion.
   */
  function sortableHeader(label: string | HTMLElement, key: AoiSortKey): HTMLElement {
    const active = aoiSort?.key === key;
    return el(
      "th",
      {
        class: "col-measure",
        // Announced, not just drawn: a sorted table that says nothing about
        // being sorted is a table a screen-reader user reads in a mystery order.
        "aria-sort": active ? (aoiSort?.dir === 1 ? "ascending" : "descending") : "none",
      },
      el(
        "button",
        {
          class: `th-sort ${active ? "is-active" : ""}`,
          type: "button",
          "data-key": `sort-${key}`,
          onclick: () => {
            // Descending first for Seen and Dwell — the biggest number is the
            // finding. Ascending first for TTFF, where the smallest is.
            const first: 1 | -1 = key === "ttff" ? 1 : -1;
            if (!aoiSort || aoiSort.key !== key) aoiSort = { key, dir: first };
            else if (aoiSort.dir === first) aoiSort = { key, dir: first === 1 ? -1 : 1 };
            else aoiSort = null;
            renderData();
          },
        },
        label,
        el(
          "span",
          { class: "th-caret", "aria-hidden": "true" },
          active ? (aoiSort?.dir === 1 ? "↑" : "↓") : "↕"
        )
      )
    );
  }

  /** Rows in the order the table should print them, each keeping the ordinal of
   * the box it names — the badge on the stimulus is numbered by drawing order,
   * so sorting the rows must not renumber the regions. */
  function sortRows(
    aggregates: ReturnType<typeof aggregateAois>
  ): Array<{ agg: (typeof aggregates)[number]; index: number }> {
    const rows = aggregates.map((agg, index) => ({ agg, index }));
    if (!aoiSort) return rows;
    const sort = aoiSort;
    // A region with no data has no place in a ranking of the data, so it sinks
    // to the bottom whichever way the column is pointed.
    const valueOf = ({ agg }: (typeof rows)[number]): number | null => {
      if (sort.key === "seen") return agg.hitRate;
      if (sort.key === "dwell") return agg.hitRate > 0 ? agg.meanDwell : null;
      return agg.meanTimeToFirstFixation;
    };
    return rows.sort((a, b) => {
      const av = valueOf(a);
      const bv = valueOf(b);
      if (av === null || bv === null) {
        if (av === bv) return a.index - b.index;
        return av === null ? 1 : -1;
      }
      return av === bv ? a.index - b.index : (av - bv) * sort.dir;
    });
  }

  /**
   * One measured cell: the number, right-aligned on tabular figures, over a bar
   * scaled to the largest value in this table.
   *
   * The bar is drawn from the right edge of a track so it shares the numbers'
   * alignment — six bars ending on the same line is the comparison; six bars
   * starting on the same line with unaligned decimals above them is two
   * competing readings of one column. A zero draws no bar at all, because a
   * hairline at the right margin would read as a small value rather than none;
   * the track stays, so a zero reads as an empty measure rather than as a cell
   * that forgot to draw one.
   */
  function measureCell(
    label: string,
    text: string,
    value: number,
    max: number
  ): HTMLElement {
    const fraction = max > 0 && value > 0 ? Math.min(1, value / max) : 0;
    const bar = el("span", { class: "cell-bar" });
    bar.style.setProperty("--v", fraction.toFixed(3));
    return el(
      "td",
      { class: "cell-measure", "data-label": label },
      el("span", { class: "cell-value" }, text),
      // The track is what the bar is a fraction *of*: a 9px fill floating in a
      // 108px cell reads as a stray mark, and the same fill inside a drawn
      // track reads as a small quantity.
      el("span", { class: "cell-track", "aria-hidden": "true" }, fraction > 0 ? bar : null)
    );
  }

  /** One region's time to first fixation, with the same sub-sample floor the
   * summary row uses. A table that prints "0ms" beside a graded dwell figure is
   * reporting a broken counter, not a measurement. TTFF gets no bar: sooner is
   * better, so a long bar would encode "worse" in a column where every other bar
   * encodes "more". */
  function ttffCell(ms: number | null): HTMLElement {
    if (ms === null) return el("td", { class: "cell-measure", "data-label": "TTFF" }, "—");
    const onset = formatOnset(ms);
    return el(
      "td",
      { class: "cell-measure", "data-label": "TTFF", title: onset.note },
      onset.label
    );
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
                      { class: "quality-badge" },
                      excluded ? "Low signal · excluded" : "Low signal"
                    )
                  : null
              ),
              // Why this participant's data was dropped, on the screen.
              //
              // It lived only in a `title` on a non-interactive span:
              // unreachable on touch, never announced to a keyboard user who
              // has no reason to focus a span, and absent from any screenshot
              // of the finding. The reason a recording was excluded from an
              // aggregate is something an author has to defend in a write-up,
              // not a hover hint — so it is a line of its own under the name,
              // in the same --signal-bad the row's left edge already wears.
              low
                ? el(
                    "p",
                    { class: "recording-reason" },
                    `${capitalise(lowSignalReason(a.recording.quality))}.${excluded ? " Excluded from the aggregate." : ""}`
                  )
                : null,
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
    // Armed, not resting, is when the verb appears.
    //
    // Six regions meant six 126px cells all reading "Remove", beside the
    // Recordings card's five "Delete"s: eleven destructive verbs on one screen,
    // out-shouting the Seen/Dwell/TTFF numbers the table exists to report. The
    // control is a × at rest and says the word only once it is armed — the
    // aria-label carries the full verb throughout, so nothing is lost to a
    // screen reader, and confirmButton still stacks both labels in one cell so
    // arming cannot reflow the row.
    const btn = confirmButton("✕", "Remove?", () => {
      const index = aois.findIndex((a) => a.id === aoiId);
      if (index >= 0) aois.splice(index, 1);
      onAoiChange();
    }, "btn btn-ghost btn-small confirm-icon");
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
   *
   * "Exactly what the stage covers" includes the per-person views: in Scanpath
   * the stage is one participant, so these files are one participant's too, and
   * the note says which and why. A file whose scope line disagrees with the
   * rows under it is the same defect as a screen that disagrees with itself,
   * only harder to notice.
   */
  function exportScope(): ExportScope {
    return { note: scopeNote(reportedScope()) };
  }

  // The menu is built once: every handler reads `mode`, `selected`,
  // `analysed` and `aois` when it fires, so it never goes stale.
  exportMenu.append(
    exportItem("PNG overlay", "export-png", () => {
      // The pixels and the caption come from the same set. They did not: the
      // canvas was sliced to one participant for a scanpath while the caption
      // was handed the literal string "All participants".
      const scope = reportedScope();
      void exportOverlayPng(study, stimulusImage, overlay, {
        mode,
        participants: scope.participants,
        scope: scopeCaption(scope),
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
        reportedSet().map((a) => ({ recording: a.recording, data: a.fixations })),
        exportScope()
      )
    ),
    exportItem("Raw CSV", "export-raw", () =>
      exportRawCsv(
        study,
        reportedSet().map((a) => a.recording),
        exportScope()
      )
    ),
    exportItem("AOI CSV", "export-aoi", () =>
      exportAoiCsv(
        study,
        aggregateAois(
          aois,
          reportedSet().map((a) => a.aoiResults)
        ),
        reportedSet().map((a) => ({ recording: a.recording, data: a.aoiResults })),
        exportScope()
      )
    ),
    // The one export that is an archive rather than a view of the screen, so it
    // stays whole: every recording, low-signal ones included and graded. The
    // rule divides it from the four above it, because "different promise" is
    // not something five identical rows can say.
    el("div", { class: "menu-sep", "aria-hidden": "true" }),
    exportItem("Session JSON", "export-json", () => exportStudyJson(study, recordings))
  );

  renderSidebar();
  renderData();
  // Before the first draw, because the draw sizes the canvas off the figure and
  // the cap is what sizes the figure.
  fitStageCap();
  await draw();
  // And again after it, because the first call measured a legend that was still
  // empty — `draw()` is what fills it — and so reserved nothing for the strip
  // the stage has to end above. Only now is the stage the size it will settle
  // at, which is the size the fit has to be chosen against: reading
  // `getBoundingClientRect` forces the style recalc the property change needs,
  // so autoFit below sees the settled geometry rather than the pre-cap one.
  fitStageCap();
  await nextFrame();
  autoFit();
  markStageClip();

  // Live resizes fire many events per second, and each full redraw re-splats
  // the entire heatmap. A trailing-edge debounce redraws once when the drag
  // settles; in the interim the CSS-sized canvas simply stretches. The cap is
  // re-measured on every event rather than on the trailing edge — it is two
  // getBoundingClientRects and one custom property, and deferring it would let
  // the stage lag the window by 150ms of every drag.
  let resizeTimer = 0;
  const onResize = () => {
    fitStageCap();
    markStageClip();
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => void draw(), 150);
  };
  window.addEventListener("resize", onResize);

  /* The legend is the one piece of the measured chrome that changes height
     without the window changing size: switching to Raw gaze swaps a gradient
     strip for a participant key, and selecting one participant swaps it back.
     Watching it keeps the cap honest across view changes without every render
     path having to remember to re-measure. The stage's own height is not
     watched, so this cannot drive itself. */
  const legendWatcher =
    typeof ResizeObserver === "function"
      ? new ResizeObserver(() => {
          fitStageCap();
          markStageClip();
        })
      : null;
  legendWatcher?.observe(legend);

  // The caller replaces the host contents on navigation; clean up after it.
  const observer = new MutationObserver(() => {
    if (!host.isConnected || !stage.isConnected) {
      window.removeEventListener("resize", onResize);
      document.removeEventListener("pointerdown", onDocumentPointer);
      document.removeEventListener("keydown", onDocumentKey);
      window.clearTimeout(resizeTimer);
      legendWatcher?.disconnect();
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

/** Leads a sentence with a fragment written to be usable mid-sentence too. */
function capitalise(text: string): string {
  return text.length === 0 ? text : text[0].toUpperCase() + text.slice(1);
}

function averageOf(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * A soft density field, not a stack of hard discs.
 *
 * Each sample was a 2.5px disc filled once at 0.35 alpha, and roughly thirty
 * samples land inside a single fixation: they saturated to opaque and their
 * union came out as a ~14px rounded square with a hard edge — in the one view
 * whose whole job is to show the tracker's raw honesty, the samples read as a
 * sprite artifact. The legend promises "density reads as saturation", which
 * needs a mark that is faint at the edge and stacks smoothly.
 *
 * So the mark is a radial-gradient sprite drawn once per participant colour into
 * an offscreen canvas and blitted per sample: wider (so a cluster is a cloud
 * rather than a tile), much fainter (so it takes real density to approach
 * opaque), and with no edge to align into a straight one. A minute of gaze is
 * ~1800 samples; drawImage of a small sprite is faster than the arc-and-fill it
 * replaces, and this runs on a debounced redraw rather than per frame.
 */
function renderRawPoints(
  canvas: HTMLCanvasElement,
  recordings: Recording[],
  scale = 1
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Stated rather than inherited: the canvas is shared with the heat overlay,
  // and a stacking density field is only a density field under source-over.
  ctx.globalCompositeOperation = "source-over";

  const radius = Math.max(4, 6 * scale);
  recordings.forEach((recording, index) => {
    const sprite = dotSprite(index, radius);
    if (!sprite) return;
    for (const p of recording.points) {
      ctx.drawImage(
        sprite,
        p.x * canvas.width - radius,
        p.y * canvas.height - radius
      );
    }
  });
}

/**
 * One gaze sample, as a pre-rendered sprite in a participant's colour: a faint
 * flat core so a lone sample is still visible as a sample, falling to nothing at
 * the rim so overlapping ones add into a cloud instead of tiling into a slab.
 *
 * All three stops come from the same `participantColour` the legend keys, so
 * there is no second definition of "P02's colour" to drift.
 */
function dotSprite(participantIndex: number, radius: number): HTMLCanvasElement | null {
  const size = Math.ceil(radius * 2);
  const sprite = document.createElement("canvas");
  sprite.width = size;
  sprite.height = size;
  const ctx = sprite.getContext("2d");
  if (!ctx) return null;
  const gradient = ctx.createRadialGradient(radius, radius, 0, radius, radius, radius);
  // 0.30 / 0.20, not 0.14 / 0.10. The lower pair was tuned against the risk of
  // a cluster saturating into a slab and overshot it: on the cream ground, over
  // a light-grey wireframe, a single sample was invisible and a whole session's
  // scatter came out fainter than the legend swatch keying it — the legend was
  // louder than the data. This is the one view whose stated job is showing the
  // tracker's raw honesty, so a sample it cannot show is a claim it cannot make.
  // A lone mark now reads; the rim still falls to nothing, so a cluster is still
  // a cloud with a soft edge rather than a tile.
  gradient.addColorStop(0, participantColour(participantIndex, 0.3));
  gradient.addColorStop(0.4, participantColour(participantIndex, 0.2));
  gradient.addColorStop(1, participantColour(participantIndex, 0));
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return sprite;
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
      {
        // A region drawn against the top edge has no room above it for the name
        // chip, which is drawn 23px up: it would be cut off by the stage — the
        // same clipping the numbered badge used to suffer. The chip drops inside
        // the box for those, which is the one case where covering a few pixels
        // of stimulus beats not being readable at all, and it is only up while
        // the box is hovered or its table row is.
        class: aoi.y < 0.035 ? "aoi-box is-top" : "aoi-box",
        "data-aoi": aoi.id,
      },
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
