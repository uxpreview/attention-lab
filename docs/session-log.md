# Session log

## 2026-08-06 — Initial build

**Goal.** Test whether webcam-based eye tracking is feasible for producing
attention heatmaps over wireframes and web pages, and if so, build enough of it
to run a real study.

**Outcome.** Feasible and built. Working prototype at `src/`, deployed for
hands-on testing.

### Approach and why

Started from an empty repo, so every choice was open.

| Decision | Alternative considered | Why |
| --- | --- | --- |
| MediaPipe `FaceLandmarker` (478 pts) | WebGazer.js | WebGazer regresses on raw eye-patch pixels, which is sensitive to lighting and needs constant recalibration. The refined mesh gives explicit iris landmarks, so the gaze signal is geometric rather than photometric. |
| Ridge regression on a hand-built basis | Neural net / CNN | A few hundred calibration samples is not enough to train anything deep. Ridge with a designed basis is fittable in-browser in milliseconds and inspectable when it goes wrong. |
| Click-to-calibrate | Dwell-only calibration | A click proves the participant was actually looking at the target. Dwell assumes cooperation and silently poisons the training set when it does not happen. |
| One Euro filter | Exponential moving average | Gaze needs heavy smoothing at rest and none during saccades. A fixed filter forces a choice between jitter and lag; One Euro adapts its cutoff to signal speed. |
| I-DT fixation detection | Raw-sample heatmaps | Saccade samples smear heat across regions nobody read. Attention lives in fixations. |
| Vanilla TS + Vite | React | The app is a state machine over full-screen canvases. A framework would sit between the code and the pixels without earning it. |
| IndexedDB | localStorage | Stimulus images are blobs and recordings run to thousands of points. |

**Feature basis.** The non-obvious part. Iris offset alone only works if the head
never moves, which no participant manages. Where you look is eyeball direction
*plus* head pose, and the two interact multiplicatively — a given iris offset
means a different screen position depending on head yaw. So the basis carries
explicit cross terms (`dx*yaw`, `dy*pitch`, `dx*scale`, …) alongside the linear
ones. 22 features total, in `src/tracker/features.ts`.

### Bug found during verification

I-DT window growth tested the *candidate* point's timestamp rather than the
current window's:

```ts
while (points[end + 1].t - points[start].t < minDuration) end++;   // wrong
while (points[end].t     - points[start].t < minDuration) end++;   // right
```

The loop stopped one sample short of `minDuration`, so no window ever qualified
and **every recording produced zero fixations** — which meant every heatmap
would have rendered empty. Found only because the test suite asserts against a
known-good synthetic scanpath rather than just checking the code runs.

This is the argument for the simulated-eye tests existing at all: the failure
mode of this system is silent. A miscalibrated or mis-analysed session still
produces a confident-looking heatmap.

### Verification status

| Area | Status |
| --- | --- |
| Ridge regression recovers unseen screen positions | Verified — simulated eye, incl. head drift |
| Fixation detection | Verified — 3 synthetic fixations at correct positions/durations |
| One Euro filter (jitter suppression + saccade tracking) | Verified |
| AOI statistics and cross-participant aggregation | Verified |
| Heatmap / spotlight / contour / scanpath rendering | Verified in Chromium via Playwright |
| IndexedDB persistence, results UI, exports | Verified in Chromium |
| **Webcam → landmarks → gaze** | **Not verified — no camera in the build environment** |

The last row is the one that matters and the reason this needs hands-on testing.
Everything downstream of the landmark stream is checked; the camera-to-landmarks
link is not. *(Where that row stands after the hands-on passes: see the
2026-08-10 entry.)*

Two rendering fixes came out of the browser pass: spotlight reveal was too weak
to read (and its `globalAlpha` was dead code — `putImageData` ignores composite
state, so overlay opacity has to be baked into the alpha channel), and scanpath
saccade lines were white-on-white against pale wireframes.

Also fixed: the camera stayed live if you abandoned calibration. Releasing it
on the study-list screen covers every exit path.

### Known limits

- **Accuracy is 2-4° of visual angle**, roughly 50-120px. Good for "which block",
  useless for "which word". Reported per session rather than hidden.
- **URL stimuli often will not load.** Many sites send `X-Frame-Options: DENY`.
  Screenshots are both more reliable and more rigorous, since every participant
  then sees byte-identical content.
- **Stimulus rect is captured once** at recording start; resizing the window
  mid-recording would drift the coordinate mapping.
- **Calibration is per-person, per-seating-position.** Cached in `sessionStorage`
  only — a stale calibration fails silently, which is worse than none.

### Open questions for testing

1. Real-world accuracy vs. the 2-4° estimate — does the reported validation
   error match where the heatmap actually lands?
2. How fast does accuracy decay over a 15-30s recording as the head drifts?
3. Is 13 calibration points the right trade against participant patience?
4. Does GPU delegate resolve on the target machines, or does it fall back to CPU
   and drop the frame rate?

## 2026-08-10 — What the hands-on passes turned up

**Goal.** Close the loop on the entry above: the prototype went up at
attention.ryankm.com, got used, and four passes came back off the deployed
build.

**Outcome.** Four passes' worth of fixes. Notably, none of them was "the gaze
signal is wrong": what real use surfaced were faults *around* the tracker —
stimuli, stored data, the invitation extended to devices that cannot take it —
plus one more instance of this system's signature failure mode, a view that
renders confidently and wrongly. The camera row of the matrix moves from
untested to exercised-but-unmeasured, which is progress and not the same thing
as verified.

### What real use surfaced

| Pass | Finding |
| --- | --- |
| URL stimuli | Iframed pages needed real guards, and the page had no metadata worth sharing. Both fixed; the honest advice remains "use a screenshot". |
| Handheld | The layout survived a phone fine. The problem was that a phone was being invited into a flow it physically cannot complete — camera a hand's length away and off-axis, calibration targets covered by the thumb that taps them, stimulus too small to be the stimulus. It would have produced output that looked exactly like data. A handheld now reads and reviews, and says why it will not record. |
| Stored studies | Studies created before URL validation existed still held raw text like `j` in the url field, which an iframe resolves as a relative path and renders as this app's own 404 — a stimulus that is silently the wrong page. Normalisation now runs at render time and blocks the session when the address cannot load. |
| Maths and durability | Cross-validation was interleaving rows across folds. Calibration samples arrive as bursts of consecutive frames per dot, so every held-out row had near-copies in the training set: leaked CV error, a flattering accuracy number, and a lambda too weak to survive head movement. Folds now hold out whole dots. Alongside: IndexedDB writes resolve on transaction completion rather than request success, deletes are one transaction, and the overlay screens are properly inert. |

### The second silent-render bug

Generating the README figures meant running the real renderers headlessly,
which put the spotlight view under a magnifying glass for the first time. Its
mask had holes. Pixels carrying a splat too faint to round up to one unit of
intensity were skipped by the colouring pass, so they stayed fully transparent
while their neighbours — both hotter and completely cold — were dimmed:

```ts
if (intensity === 0) continue;   // wrong: leaves an undimmed halo in the mask
```

The result was a bright ring around every hot spot, reading as a second,
phantom region of attention. Spotlight is a mask, not an overlay, so it has to
write every pixel; `paintField` now does, and the test suite asserts a faint rim
is dimmed rather than skipped. Same shape as the I-DT off-by-one from the first
entry: no error, no crash, just a picture that means something other than what
it appears to.

The figures themselves are generated by `scripts/make-figures.ts`, which drives
the app's own `detectFixations` → `renderHeatmap` / `renderScanpath` path
through a minimal canvas shim and a PNG encoder over `node:zlib`. Nothing there
re-implements the look of a heatmap, so a README picture cannot quietly drift
from what the app draws — and it is why the halo showed up at all.

### Verification status, revised

| Area | Status |
| --- | --- |
| Webcam → landmarks → gaze | Exercised in a browser during the passes above, with no tracking failure reported — but exercised is not measured |
| Accuracy against the 2-4° claim | **Still unquantified** — nothing has been run against a known ground truth |
| Handheld capture | Ruled out on physics, not attempted |
| Heatmap / spotlight / contour / scanpath rendering | Verified, and now regression-tested at the pixel-mapping level |
| Calibration model selection | Verified — grouped folds, 62 headless checks |

The accuracy row is the one still worth work. "It runs and nobody reported the
dot going somewhere strange" is a different claim from "error is 2-4° of visual
angle", and only the first is supported. Answering the second properly needs a
fixed viewing distance, a printed target grid, and someone patient — which is
the next session, not this one. Open questions 2 and 3 sit behind the same
measurement. Question 4 is answered structurally but not empirically: the
tracker asks for the GPU delegate and falls back to CPU on failure, and nobody
has yet checked which branch a given machine takes or what it costs in frame
rate.
