import {
  legendFor,
  type LegendScale,
  type LegendSpec,
  type OverlayMode,
} from "../analysis/legend";
import { el } from "./dom";

/** Renders a legend spec as DOM, under the stage it explains. `scale` is what
 * the renderer set the hot end of the ramp to for this selection; without it
 * the strip keeps its words and drops its numbers. */
export function legendElement(
  mode: OverlayMode,
  participants: string[],
  scale: LegendScale | null = null
): HTMLElement {
  return specElement(legendFor(mode, participants, scale));
}

export function gradientCss(spec: LegendSpec): string {
  const stops = spec.stops ?? [];
  if (spec.banded) {
    // Hard steps: a blend would promise a continuum the banded renderer does
    // not draw.
    const parts = stops.flatMap((colour, i) => {
      const from = ((i / stops.length) * 100).toFixed(2);
      const to = (((i + 1) / stops.length) * 100).toFixed(2);
      return [`${colour} ${from}%`, `${colour} ${to}%`];
    });
    return `linear-gradient(to right, ${parts.join(", ")})`;
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function specElement(spec: LegendSpec): HTMLElement {
  const body: HTMLElement[] = [];

  if (spec.stops) {
    const strip = el("div", { class: "legend-strip", "aria-hidden": "true" });
    strip.style.backgroundImage = gradientCss(spec);

    // The numbers, where the axis has any. They are positioned by the fraction
    // of the ramp they name rather than spread evenly, so the value sits under
    // the colour that means it; the two end ticks pull back inside the strip
    // instead of hanging off it (see .legend-ticks in styles.css).
    const ticks = spec.ticks
      ? el(
          "div",
          { class: "legend-ticks" },
          ...spec.ticks.map((tick) => {
            const mark = el("span", { class: "legend-tick" }, tick.label);
            mark.style.left = `${tick.at * 100}%`;
            return mark;
          })
        )
      : null;

    body.push(
      el(
        "div",
        { class: "legend-scale" },
        strip,
        ticks,
        el(
          "div",
          { class: "legend-ends" },
          el("span", {}, spec.minLabel),
          el("span", {}, spec.maxLabel)
        )
      )
    );
  }

  if (spec.swatches) {
    body.push(
      el(
        "ul",
        { class: "legend-swatches" },
        ...spec.swatches.map((swatch) => {
          const chip = el("span", { class: "legend-chip", "aria-hidden": "true" });
          chip.style.background = swatch.colour;
          return el("li", {}, chip, el("span", {}, swatch.label));
        })
      )
    );
  }

  return el(
    "figcaption",
    { class: "legend" },
    el("p", { class: "label legend-title" }, spec.title),
    ...body,
    /**
     * The caption stays; the essay folds away.
     *
     * Every mode used to print its full `note` under the stage on every view —
     * 68 words and a measured 174px for the heatmap — which is a paragraph a
     * researcher reads once and then scrolls past forever, on the screen where
     * the region table most needs the room. What survives inline is the ramp,
     * its endpoint labels and one clause naming the quantity; the caveats a
     * reader needs before *citing* the figure are one disclosure away, and the
     * print stylesheet forces that disclosure open so a paper figure still
     * carries its full caption.
     */
    el(
      "div",
      { class: "legend-read" },
      el("p", { class: "legend-caption" }, spec.caption),
      el(
        "details",
        { class: "legend-how" },
        el("summary", {}, "How to read this"),
        el("p", { class: "legend-note muted" }, spec.note)
      )
    )
  );
}
