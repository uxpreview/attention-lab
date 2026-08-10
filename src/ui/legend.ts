import { legendFor, type LegendSpec, type OverlayMode } from "../analysis/legend";
import { el } from "./dom";

/** Renders a legend spec as DOM, under the stage it explains. */
export function legendElement(mode: OverlayMode, participants: string[]): HTMLElement {
  return specElement(legendFor(mode, participants));
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
    body.push(
      el(
        "div",
        { class: "legend-scale" },
        strip,
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
    el("p", { class: "legend-note muted" }, spec.note)
  );
}
