/**
 * One Euro filter (Casiez, Roussel & Vogel, 2012).
 *
 * Gaze prediction is noisy at rest but must not lag during saccades. A fixed
 * low-pass filter forces a choice between the two; the One Euro filter adapts
 * its cutoff to the signal's speed, so it smooths hard while the eye is still
 * and gets out of the way when the eye jumps.
 */

class LowPass {
  private value: number | null = null;

  filter(x: number, alpha: number): number {
    this.value = this.value === null ? x : alpha * x + (1 - alpha) * this.value;
    return this.value;
  }

  get last(): number | null {
    return this.value;
  }

  reset(): void {
    this.value = null;
  }
}

export interface OneEuroOptions {
  /** Cutoff frequency at zero speed, in Hz. Lower means smoother but laggier. */
  minCutoff?: number;
  /** How aggressively the cutoff rises with speed. */
  beta?: number;
  /** Cutoff for the derivative estimate itself. */
  dCutoff?: number;
}

class OneEuroScalar {
  private readonly minCutoff: number;
  private readonly beta: number;
  private readonly dCutoff: number;
  private readonly xFilter = new LowPass();
  private readonly dxFilter = new LowPass();
  private lastTime: number | null = null;

  constructor({ minCutoff = 1.0, beta = 0.007, dCutoff = 1.0 }: OneEuroOptions = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  filter(x: number, timestamp: number): number {
    const dt = this.lastTime === null ? 1 / 30 : Math.max((timestamp - this.lastTime) / 1000, 1e-4);
    this.lastTime = timestamp;

    const prev = this.xFilter.last;
    const dx = prev === null ? 0 : (x - prev) / dt;
    const edx = this.dxFilter.filter(dx, alphaFor(this.dCutoff, dt));

    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(x, alphaFor(cutoff, dt));
  }

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTime = null;
  }
}

function alphaFor(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

/**
 * 3-sample sliding median despiker.
 *
 * The One Euro filter deliberately follows fast jumps, which leaves it
 * defenceless against the classic webcam-gaze artifact: during a partial blink
 * the descending eyelid drags the iris centroid down for a frame or two before
 * openness crosses the rejection threshold, producing a spurious downward
 * saccade. A median over three consecutive samples deletes any single-frame
 * excursion outright, at the cost of one frame of latency — well under the One
 * Euro filter's own lag at 30 fps. Genuine saccades survive because they
 * persist past one frame.
 */
export class MedianPoint {
  private readonly xs: number[] = [];
  private readonly ys: number[] = [];

  filter(x: number, y: number): [number, number] {
    this.xs.push(x);
    this.ys.push(y);
    if (this.xs.length > 3) {
      this.xs.shift();
      this.ys.shift();
    }
    if (this.xs.length < 3) return [x, y];
    return [median3(this.xs[0], this.xs[1], this.xs[2]), median3(this.ys[0], this.ys[1], this.ys[2])];
  }

  reset(): void {
    this.xs.length = 0;
    this.ys.length = 0;
  }
}

function median3(a: number, b: number, c: number): number {
  return Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
}

export class OneEuroPoint {
  private readonly fx: OneEuroScalar;
  private readonly fy: OneEuroScalar;

  constructor(options: OneEuroOptions = {}) {
    this.fx = new OneEuroScalar(options);
    this.fy = new OneEuroScalar(options);
  }

  filter(x: number, y: number, timestamp: number): [number, number] {
    return [this.fx.filter(x, timestamp), this.fy.filter(y, timestamp)];
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }
}
