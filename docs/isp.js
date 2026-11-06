// Browser port of isp/classical.py and isp/pipeline.py. Keep the two in step.
// Images are Float32Arrays: planes are row-major (h*w), RGB images are interleaved (h*w*3).

const XYZ_TO_SRGB = [3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252];
const RED_OFFSET = { rggb: [0, 0], grbg: [0, 1], gbrg: [1, 0], bggr: [1, 1] };
const PAD = 18; // even, keeps CFA phase; exceeds the 8-pixel border excluded from the training loss

const mirror = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i; return i; };
const symmetric = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i; return i; };
const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

function mat3Mul(a, b) {
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return o;
}

function mat3Inv(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), Cc = d * h - e * g;
  const det = a * A + b * B + c * Cc;
  if (Math.abs(det) < 1e-12) throw new Error("The camera colour matrix cannot be inverted.");
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), Cc, -(a * h - b * g), a * e - b * d].map(v => v / det);
}

const normRows = m => { const o = m.slice(); for (let r = 0; r < 3; r++) { const s = m[r * 3] + m[r * 3 + 1] + m[r * 3 + 2]; for (let c = 0; c < 3; c++) o[r * 3 + c] = m[r * 3 + c] / s; } return o; };

// White-balanced camera RGB to linear sRGB. Same normalisation as classical.cam_to_srgb.
export function camToSrgb(colorMatrix) {
  return mat3Mul(XYZ_TO_SRGB, normRows(mat3Inv(normRows(colorMatrix))));
}

export function toCanonical(bayer, h, w, cfa, pad = PAD) {
  const [oy, ox] = RED_OFFSET[cfa];
  const hc = ((h + 2 * pad - oy) >> 1) << 1, wc = ((w + 2 * pad - ox) >> 1) << 1;
  const out = new Float32Array(hc * wc);
  for (let y = 0; y < hc; y++) {
    const sy = mirror(y + oy - pad, h) * w;
    for (let x = 0; x < wc; x++) out[y * wc + x] = bayer[sy + mirror(x + ox - pad, w)];
  }
  return { canvas: out, hc, wc, y0: pad - oy, x0: pad - ox };
}

// Four planes R, G1, G2, B at half resolution.
export function pack(c, hc, wc) {
  const hp = hc >> 1, wp = wc >> 1, n = hp * wp;
  const p = [new Float32Array(n), new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let y = 0; y < hp; y++) for (let x = 0; x < wp; x++) {
    const i = y * wp + x, s = 2 * y * wc + 2 * x;
    p[0][i] = c[s]; p[1][i] = c[s + 1]; p[2][i] = c[s + wc]; p[3][i] = c[s + wc + 1];
  }
  return { p, hp, wp };
}

function unpack(p, hp, wp) {
  const wc = 2 * wp, out = new Float32Array(4 * hp * wp);
  for (let y = 0; y < hp; y++) for (let x = 0; x < wp; x++) {
    const i = y * wp + x, s = 2 * y * wc + 2 * x;
    out[s] = p[0][i]; out[s + 1] = p[1][i]; out[s + wc] = p[2][i]; out[s + wc + 1] = p[3][i];
  }
  return out;
}

// Bilinear demosaic of an RGGB canvas, reflect-101 borders, same kernels as classical.py.
export function demosaicBilinear(c, h, w) {
  const out = new Float32Array(h * w * 3);
  const at = (y, x) => c[mirror(y, h) * w + mirror(x, w)];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = c[y * w + x], ey = y & 1, ex = x & 1, o = (y * w + x) * 3;
    const cross = (at(y - 1, x) + at(y + 1, x) + at(y, x - 1) + at(y, x + 1)) / 4;
    const diag = (at(y - 1, x - 1) + at(y - 1, x + 1) + at(y + 1, x - 1) + at(y + 1, x + 1)) / 4;
    const hor = (at(y, x - 1) + at(y, x + 1)) / 2, ver = (at(y - 1, x) + at(y + 1, x)) / 2;
    if (!ey && !ex) { out[o] = v; out[o + 1] = cross; out[o + 2] = diag; }
    else if (ey && ex) { out[o] = diag; out[o + 1] = cross; out[o + 2] = v; }
    else if (!ey) { out[o] = hor; out[o + 1] = v; out[o + 2] = ver; }
    else { out[o] = ver; out[o + 1] = v; out[o + 2] = hor; }
  }
  return out;
}

export function srgbOetf(l) {
  l = clamp(l, 0, 1);
  return l <= 0.0031308 ? 12.92 * l : 1.055 * Math.pow(Math.max(l, 1e-8), 1 / 2.4) - 0.055;
}
const srgbInverse = s => (s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4));

function median(arrays, stride = 7) {
  const v = [];
  for (const a of arrays) for (let i = 0; i < a.length; i += stride) v.push(a[i]);
  v.sort((x, y) => x - y);
  return v[v.length >> 1];
}

export function relativeNoise(p, a, b) {
  const m = Math.max(median(p), 1e-3);
  return Math.sqrt(a * m + b) / m;
}

// Poisson-Gaussian fit from the difference of the two green planes.
export function estimateNoise(p, hp, wp, block = 8) {
  const by = Math.floor(hp / block), bx = Math.floor(wp / block), bm = [], bv = [];
  for (let i = 0; i < by; i++) for (let j = 0; j < bx; j++) {
    let sg = 0, sd = 0, sdd = 0;
    for (let y = 0; y < block; y++) for (let x = 0; x < block; x++) {
      const k = (i * block + y) * wp + j * block + x;
      const d = (p[1][k] - p[2][k]) / Math.SQRT2;
      sg += (p[1][k] + p[2][k]) / 2; sd += d; sdd += d * d;
    }
    const n = block * block;
    bm.push(sg / n); bv.push(sdd / n - (sd / n) ** 2);
  }
  const order = bm.map((_, i) => i).sort((x, y) => bm[x] - bm[y]);
  const xs = [], ys = [], bins = 16;
  for (let q = 0; q < bins; q++) {
    const idx = order.slice(Math.floor(q * order.length / bins), Math.floor((q + 1) * order.length / bins));
    if (idx.length <= 8) continue;
    const vals = idx.map(i => bv[i]).sort((x, y) => x - y);
    xs.push(idx.reduce((s, i) => s + bm[i], 0) / idx.length);
    ys.push(vals[Math.floor(vals.length * 0.3)] * 1.25);
  }
  if (xs.length < 3) return { a: 1e-7, b: Math.max(ys[0] || 1e-9, 1e-9) };
  const n = xs.length, mx = xs.reduce((s, v) => s + Math.max(v, 0), 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0;
  xs.forEach((x, i) => { const d = Math.max(x, 0) - mx; sxy += d * (ys[i] - my); sxx += d * d; });
  const a = sxx > 0 ? sxy / sxx : 0;
  return { a: Math.max(a, 1e-7), b: Math.max(my - a * mx, 1e-9) };
}

function bilateralPlane(src, h, w, sigmaColor, sigmaSpace) {
  const offs = [];
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const r2 = dy * dy + dx * dx;
    if (r2 <= 4) offs.push([dy, dx, Math.exp(-r2 / (2 * sigmaSpace * sigmaSpace))]);
  }
  const gc = -1 / (2 * sigmaColor * sigmaColor), out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = src[y * w + x];
    let s = 0, n = 0;
    for (const [dy, dx, ws] of offs) {
      const u = src[mirror(y + dy, h) * w + mirror(x + dx, w)], d = u - v, k = ws * Math.exp(d * d * gc);
      s += k * u; n += k;
    }
    out[y * w + x] = s / n;
  }
  return out;
}

function denoisePacked(p, hp, wp, a, b) {
  const off = 3 / 8 + b / (a * a);
  return p.map(plane => {
    const v = new Float32Array(plane.length);
    for (let i = 0; i < v.length; i++) v[i] = 2 * Math.sqrt(Math.max(plane[i] / a + off, 0));
    const f = bilateralPlane(v, hp, wp, 2.5, 1.5);
    for (let i = 0; i < f.length; i++) f[i] = ((f[i] / 2) ** 2 - off) * a;
    return f;
  });
}

function box(src, h, w, r) {
  const tmp = new Float32Array(h * w), out = new Float32Array(h * w), n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[y * w + symmetric(x, w)];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / n;
      s += src[y * w + symmetric(x + r + 1, w)] - src[y * w + symmetric(x - r, w)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[symmetric(y, h) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / n;
      s += tmp[symmetric(y + r + 1, h) * w + x] - tmp[symmetric(y - r, h) * w + x];
    }
  }
  return out;
}

function guided(g, src, h, w, r, eps) {
  const n = h * w, gs = new Float32Array(n), gg = new Float32Array(n);
  for (let i = 0; i < n; i++) { gs[i] = g[i] * src[i]; gg[i] = g[i] * g[i]; }
  const mg = box(g, h, w, r), ms = box(src, h, w, r), mgs = box(gs, h, w, r), mgg = box(gg, h, w, r);
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) { a[i] = (mgs[i] - mg[i] * ms[i]) / (mgg[i] - mg[i] * mg[i] + eps); b[i] = ms[i] - a[i] * mg[i]; }
  const ma = box(a, h, w, r), mb = box(b, h, w, r), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ma[i] * g[i] + mb[i];
  return out;
}

function toYcc(rgb, n) {
  const Y = new Float32Array(n), Cb = new Float32Array(n), Cr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2], y = 0.299 * r + 0.587 * g + 0.114 * b;
    Y[i] = y; Cb[i] = (b - y) * 0.564; Cr[i] = (r - y) * 0.713;
  }
  return [Y, Cb, Cr];
}

function fromYcc(Y, Cb, Cr, out) {
  for (let i = 0; i < Y.length; i++) {
    const r = Y[i] + Cr[i] / 0.713, b = Y[i] + Cb[i] / 0.564;
    out[3 * i] = r; out[3 * i + 1] = (Y[i] - 0.299 * r - 0.114 * b) / 0.587; out[3 * i + 2] = b;
  }
  return out;
}

function denoiseChroma(rgb, h, w, r, eps) {
  const [Y, Cb, Cr] = toYcc(rgb, h * w);
  return fromYcc(Y, guided(Y, Cb, h, w, r, eps), guided(Y, Cr, h, w, r, eps), rgb);
}

function gaussianBlur(src, h, w, sigma) {
  const r = Math.ceil(3 * sigma), k = [];
  let s = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i / (2 * sigma * sigma)); k.push(v); s += v; }
  const tmp = new Float32Array(h * w), out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let t = 0;
    for (let i = -r; i <= r; i++) t += k[i + r] * src[y * w + mirror(x + i, w)];
    tmp[y * w + x] = t / s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let t = 0;
    for (let i = -r; i <= r; i++) t += k[i + r] * tmp[mirror(y + i, h) * w + x];
    out[y * w + x] = t / s;
  }
  return out;
}

function sharpen(rgb, h, w, amount) {
  if (amount <= 0) return rgb;
  const [Y, Cb, Cr] = toYcc(rgb, h * w), blur = gaussianBlur(Y, h, w, 1);
  for (let i = 0; i < Y.length; i++) {
    const d = Y[i] - blur[i];
    if (Math.abs(d) > 0.01) Y[i] += amount * d;
  }
  return fromYcc(Y, Cb, Cr, rgb);
}

// Shared back end: colour matrix, gamma, tone curve, sharpening, then crop to the frame.
function finish(lin, hc, wc, scene, look, y0, x0, h, w) {
  const m = camToSrgb(scene.colorMatrix), n = hc * wc, e = look.exposure, out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = lin[3 * i] * e, g = lin[3 * i + 1] * e, b = lin[3 * i + 2] * e;
    for (let c = 0; c < 3; c++) {
      let v = srgbOetf(m[3 * c] * r + m[3 * c + 1] * g + m[3 * c + 2] * b);
      if (look.tone > 0) v = (1 - look.tone) * v + look.tone * (3 * v * v - 2 * v * v * v);
      out[3 * i + c] = v;
    }
  }
  sharpen(out, hc, wc, look.sharpen);
  const crop = new Float32Array(h * w * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    crop[(y * w + x) * 3 + c] = clamp(out[((y + y0) * wc + x + x0) * 3 + c], 0, 1);
  }
  return crop;
}

const gainsOf = n => [1 / n[0], 1 / n[1], 1 / n[1], 1 / n[2]];

export function classicalIsp(bayer, h, w, scene, look) {
  const { canvas, hc, wc, y0, x0 } = toCanonical(bayer, h, w, scene.cfa);
  const { p, hp, wp } = pack(canvas, hc, wc);
  const rel = relativeNoise(p, scene.a, scene.b);
  const g = gainsOf(scene.neutral);
  const dn = denoisePacked(p, hp, wp, scene.a, scene.b).map((plane, c) => plane.map(v => v * g[c]));
  const rgb = demosaicBilinear(unpack(dn, hp, wp), hc, wc);
  denoiseChroma(rgb, hc, wc, Math.trunc(clamp(4 + 40 * rel, 4, 16)), clamp(rel, 1e-3, 0.3) ** 2 * 0.05);
  for (let i = 0; i < rgb.length; i++) rgb[i] = clamp(rgb[i], 0, 1);
  return finish(rgb, hc, wc, scene, look, y0, x0, h, w);
}

// Network on overlapping tiles. `run(data, th, tw)` resolves to a Float32Array (12, th, tw).
export async function hybridIsp(bayer, h, w, scene, look, run, onProgress) {
  const { canvas, hc, wc, y0, x0 } = toCanonical(bayer, h, w, scene.cfa);
  const { p, hp, wp } = pack(canvas, hc, wc);
  const g = gainsOf(scene.neutral);
  const f = p.map((plane, c) => plane.map(v => { const x = v * g[c]; return Math.sign(x) * Math.sqrt(Math.abs(x) + 1e-8); }));
  const res = new Float32Array(hc * wc * 3), TILE = 256, OVER = 16, step = TILE - 2 * OVER;
  const total = Math.ceil(hp / step) * Math.ceil(wp / step);
  let done = 0;
  for (let y = 0; y < hp; y += step) for (let x = 0; x < wp; x += step) {
    const ya = Math.max(y - OVER, 0), xa = Math.max(x - OVER, 0);
    const yb = Math.min(y + step + OVER, hp), xb = Math.min(x + step + OVER, wp);
    const th = yb - ya, tw = xb - xa, ph = th + (-th & 7), pw = tw + (-tw & 7);
    const t = new Float32Array(6 * ph * pw); // planes 4 and 5 stay zero: the student has no noise input
    for (let c = 0; c < 4; c++) for (let i = 0; i < ph; i++) {
      const sy = (ya + Math.min(i, th - 1)) * wp;
      for (let j = 0; j < pw; j++) t[(c * ph + i) * pw + j] = f[c][sy + xa + Math.min(j, tw - 1)];
    }
    const o = await run(t, ph, pw);
    const ye = Math.min(y + step, hp), xe = Math.min(x + step, wp);
    for (let i = y - ya; i < ye - ya; i++) for (let j = x - xa; j < xe - xa; j++) {
      for (let c = 0; c < 3; c++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        res[((2 * (ya + i) + dy) * wc + 2 * (xa + j) + dx) * 3 + c] = o[((c * 4 + dy * 2 + dx) * ph + i) * pw + j];
      }
    }
    onProgress?.(++done / total);
  }
  const base = demosaicBilinear(unpack(f, hp, wp), hc, wc);
  for (let i = 0; i < res.length; i++) { const v = base[i] + res[i]; res[i] = clamp(Math.sign(v) * v * v, 0, 1); }
  return finish(res, hc, wc, scene, look, y0, x0, h, w);
}

export function analyse(frame, displayLook) {
  const { canvas, hc, wc } = toCanonical(frame.bayer, frame.h, frame.w, frame.cfa, 2);
  const { p, hp, wp } = pack(canvas, hc, wc);
  const { a, b } = frame.noise || estimateNoise(p, hp, wp);
  let neutral = frame.neutral, wbSource = frame.wbSource || "as-shot metadata";
  if (!neutral) {
    const mean = plane => plane.reduce((s, v) => s + v, 0) / plane.length;
    const G = (mean(p[1]) + mean(p[2])) / 2;
    neutral = [mean(p[0]) / G, 1, mean(p[3]) / G];
    wbSource = "estimated (gray-world)";
  }
  const rel = relativeNoise(p, a, b), med = median(p);
  let exposure = 1;
  if (displayLook && med < 0.03) {
    const y = [];
    for (let i = 0; i < p[0].length; i += 7) {
      y.push(0.2126 * p[0][i] / neutral[0] + 0.7152 * (p[1][i] + p[2][i]) / 2 / neutral[1] + 0.0722 * p[3][i] / neutral[2]);
    }
    y.sort((u, v) => u - v);
    exposure = clamp(0.18 / Math.max(y[y.length >> 1], 1e-4), 1, 8);
  }
  const look = displayLook ? { tone: 0.35, sharpen: clamp(0.6 - 6 * rel, 0, 0.6), exposure } : { tone: 0, sharpen: 0, exposure: 1 };
  return { a, b, neutral, wbSource, rel, look };
}

// Inverts gamma, colour and white balance, then mosaics: lets ordinary pictures exercise the RAW path.
export function unprocess(srgb, h, w, colorMatrix, neutral) {
  const inv = mat3Inv(camToSrgb(colorMatrix)), H = h & ~1, W = w & ~1, bayer = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * w + x) * 3, r = srgbInverse(srgb[i]), g = srgbInverse(srgb[i + 1]), b = srgbInverse(srgb[i + 2]);
    const c = (y & 1) === 0 ? ((x & 1) === 0 ? 0 : 1) : ((x & 1) === 0 ? 1 : 2);
    bayer[y * W + x] = clamp(inv[3 * c] * r + inv[3 * c + 1] * g + inv[3 * c + 2] * b, 0, 1) * neutral[c];
  }
  return { bayer, h: H, w: W };
}

// Averages NxN same-colour Bayer tiles so big frames stay within the browser's time budget.
export function binFrame(frame, maxPixels) {
  const k = Math.ceil(Math.sqrt(frame.h * frame.w / maxPixels));
  if (k <= 1) return 1;
  const { canvas, hc, wc } = toCanonical(frame.bayer, frame.h, frame.w, frame.cfa, 2);
  const { p, hp, wp } = pack(canvas, hc, wc);
  const bh = Math.floor(hp / k), bw = Math.floor(wp / k);
  const q = p.map(plane => {
    const o = new Float32Array(bh * bw);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      let s = 0;
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) s += plane[(y * k + i) * wp + x * k + j];
      o[y * bw + x] = s / (k * k);
    }
    return o;
  });
  frame.bayer = unpack(q, bh, bw);
  frame.h = 2 * bh; frame.w = 2 * bw; frame.cfa = "rggb"; frame.reference = null; frame.noise = null;
  return k;
}

export function addSensorNoise(frame, shot, read) {
  if (shot <= 0 && read <= 0) return;
  let seed = 12345;
  const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return (seed + 0.5) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand());
  const out = new Float32Array(frame.bayer.length);
  for (let i = 0; i < out.length; i++) out[i] = frame.bayer[i] + gauss() * Math.sqrt(shot * clamp(frame.bayer[i], 0, 1) + read);
  frame.bayer = out;
  if (frame.noise) frame.noise = { a: frame.noise.a + shot, b: frame.noise.b + read };
}

export function psnr(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return -10 * Math.log10(s / a.length + 1e-12);
}

function rgbToLab(r, g, b) {
  const l = [srgbInverse(r), srgbInverse(g), srgbInverse(b)];
  const X = (0.412453 * l[0] + 0.357580 * l[1] + 0.180423 * l[2]) / 0.95047;
  const Y = 0.212671 * l[0] + 0.715160 * l[1] + 0.072169 * l[2];
  const Z = (0.019334 * l[0] + 0.119193 * l[1] + 0.950227 * l[2]) / 1.08883;
  const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
}

function ciede2000(p, q) {
  const [L1, a1, b1] = p, [L2, a2, b2] = q, rad = Math.PI / 180;
  const C = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2, G = 0.5 * (1 - Math.sqrt(C ** 7 / (C ** 7 + 25 ** 7)));
  const A1 = (1 + G) * a1, A2 = (1 + G) * a2, C1 = Math.hypot(A1, b1), C2 = Math.hypot(A2, b2);
  const hue = (b, a) => { const t = Math.atan2(b, a) / rad; return t < 0 ? t + 360 : t; };
  const h1 = hue(b1, A1), h2 = hue(b2, A2);
  let dh = h2 - h1;
  if (C1 * C2 === 0) dh = 0; else if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
  const dL = L2 - L1, dC = C2 - C1, dH = 2 * Math.sqrt(C1 * C2) * Math.sin(dh * rad / 2);
  const Lm = (L1 + L2) / 2, Cm = (C1 + C2) / 2;
  let hm = h1 + h2;
  if (C1 * C2 !== 0) hm = Math.abs(h1 - h2) > 180 ? (h1 + h2 + (h1 + h2 < 360 ? 360 : -360)) / 2 : (h1 + h2) / 2;
  const T = 1 - 0.17 * Math.cos((hm - 30) * rad) + 0.24 * Math.cos(2 * hm * rad) + 0.32 * Math.cos((3 * hm + 6) * rad) - 0.2 * Math.cos((4 * hm - 63) * rad);
  const SL = 1 + 0.015 * (Lm - 50) ** 2 / Math.sqrt(20 + (Lm - 50) ** 2), SC = 1 + 0.045 * Cm, SH = 1 + 0.015 * Cm * T;
  const RT = -2 * Math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)) * Math.sin(60 * rad * Math.exp(-(((hm - 275) / 25) ** 2)));
  return Math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH));
}

export function deltaE(a, b) {
  let s = 0;
  const n = a.length / 3;
  for (let i = 0; i < n; i++) s += ciede2000(rgbToLab(a[3 * i], a[3 * i + 1], a[3 * i + 2]), rgbToLab(b[3 * i], b[3 * i + 1], b[3 * i + 2]));
  return s / n;
}

export function centreCrop(img, h, w, size = 256) {
  const s = Math.min(size, h, w), y0 = (h - s) >> 1, x0 = (w - s) >> 1, out = new Float32Array(s * s * 3);
  for (let y = 0; y < s; y++) out.set(img.subarray(((y + y0) * w + x0) * 3, ((y + y0) * w + x0 + s) * 3), y * s * 3);
  return out;
}
