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
link is not.

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
