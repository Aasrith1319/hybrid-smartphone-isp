// File decoding for the browser app: uncompressed DNG, .npy Bayer arrays and pictures.
import { unprocess } from "./isp.js";

const SIMULATED_NEUTRAL = [0.55, 1, 0.55];
const MAX_SIDE = 1400;

export class InputError extends Error {}

function readIfd(v, off, le) {
  const n = v.getUint16(off, le), tags = {}, size = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8, 11: 4, 12: 8 };
  for (let i = 0; i < n; i++) {
    const p = off + 2 + 12 * i, id = v.getUint16(p, le), type = v.getUint16(p + 2, le), count = v.getUint32(p + 4, le);
    const bytes = (size[type] || 1) * count, at = bytes <= 4 ? p + 8 : v.getUint32(p + 8, le);
    tags[id] = { type, count, at };
  }
  return tags;
}

function values(v, t, le) {
  if (!t) return null;
  const o = [];
  for (let i = 0; i < t.count; i++) {
    switch (t.type) {
      case 1: case 7: o.push(v.getUint8(t.at + i)); break;
      case 3: o.push(v.getUint16(t.at + 2 * i, le)); break;
      case 4: o.push(v.getUint32(t.at + 4 * i, le)); break;
      case 5: o.push(v.getUint32(t.at + 8 * i, le) / v.getUint32(t.at + 8 * i + 4, le)); break;
      case 9: o.push(v.getInt32(t.at + 4 * i, le)); break;
      case 10: o.push(v.getInt32(t.at + 8 * i, le) / v.getInt32(t.at + 8 * i + 4, le)); break;
      case 11: o.push(v.getFloat32(t.at + 4 * i, le)); break;
      case 12: o.push(v.getFloat64(t.at + 8 * i, le)); break;
      default: throw new InputError("This DNG uses a field type the browser decoder does not support.");
    }
  }
  return o;
}

export function decodeDng(buffer) {
  const v = new DataView(buffer), mark = v.getUint16(0, false);
  if (mark !== 0x4949 && mark !== 0x4d4d) throw new InputError("This is not a valid DNG file.");
  const le = mark === 0x4949, ifd0 = readIfd(v, v.getUint32(4, le), le);
  const candidates = [ifd0, ...(values(v, ifd0[330], le) || []).map(o => readIfd(v, o, le))];
  const img = candidates.find(t => (values(v, t[262], le) || [])[0] === 32803);
  if (!img) throw new InputError("No Bayer image found. Only 2x2 Bayer DNG files are supported.");
  if (values(v, img[259], le)[0] !== 1) {
    throw new InputError("This DNG is compressed. The browser app reads uncompressed DNG only.");
  }
  const bits = values(v, img[258], le)[0], W = values(v, img[256], le)[0], H = values(v, img[257], le)[0];
  if (bits !== 16 && bits !== 8) throw new InputError(`This DNG stores ${bits}-bit packed samples, which the browser decoder does not support.`);
  const dim = values(v, img[33421], le) || values(v, ifd0[33421], le), pattern = values(v, img[33422], le) || values(v, ifd0[33422], le);
  if (!pattern || pattern.length !== 4 || (dim && (dim[0] !== 2 || dim[1] !== 2))) throw new InputError("Only 2x2 Bayer sensors are supported.");
  const cfa = pattern.map(c => "rgb"[c] ?? "?").join("");
  if (!["rggb", "grbg", "gbrg", "bggr"].includes(cfa)) throw new InputError(`Unsupported colour filter pattern: ${cfa}.`);

  const offsets = values(v, img[273], le), counts = values(v, img[279], le);
  if (!offsets) throw new InputError("This DNG uses tiles, which the browser decoder does not support.");
  const raw = new Float32Array(W * H);
  let n = 0;
  offsets.forEach((o, s) => {
    const len = counts[s] / (bits / 8);
    for (let i = 0; i < len && n < raw.length; i++, n++) raw[n] = bits === 16 ? v.getUint16(o + 2 * i, le) : v.getUint8(o + i);
  });
  if (n !== raw.length) throw new InputError("This DNG looks truncated.");

  const blackV = values(v, img[50714], le) || values(v, ifd0[50714], le) || [0];
  const white = (values(v, img[50717], le) || values(v, ifd0[50717], le) || [2 ** bits - 1])[0];
  const black = blackV.reduce((s, x) => s + x, 0) / blackV.length;
  const [t, l, b, r] = values(v, img[50829], le) || values(v, ifd0[50829], le) || [0, 0, H, W];
  const h = (b - t) & ~1, w = (r - l) & ~1, bayer = new Float32Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y + t) * W + x + l, k = blackV.length === 4 ? blackV[((y + t) & 1) * 2 + ((x + l) & 1)] : black;
    bayer[y * w + x] = (raw[i] - k) / (white - black);
  }
  // An active area with an odd offset changes the phase of the pattern.
  const phase = (t & 1) * 2 + (l & 1), order = [0, 1, 2, 3].map(i => cfa[i ^ phase]).join("");
  const cm = values(v, ifd0[50722], le) || values(v, ifd0[50721], le);
  if (!cm || cm.length !== 9) throw new InputError("This DNG has no 3-colour matrix.");
  const neutral = values(v, ifd0[50728], le);
  return { bayer, h, w, cfa: order, colorMatrix: cm, neutral: neutral ? neutral.map(x => x / neutral[1]) : null, source: "DNG file" };
}

export function decodeNpy(buffer, fallbackMatrix) {
  const v = new DataView(buffer), u8 = new Uint8Array(buffer);
  if (String.fromCharCode(...u8.subarray(1, 6)) !== "NUMPY") throw new InputError("This is not a valid .npy file.");
  const major = u8[6], first = major === 1 ? 10 : 12, hl = major === 1 ? v.getUint16(8, true) : v.getUint32(8, true), start = first + hl;
  const header = String.fromCharCode(...u8.subarray(first, start));
  const dtype = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1].split(",").map(s => s.trim()).filter(Boolean).map(Number);
  const fortran = /'fortran_order':\s*True/.test(header);
  if (!shape || shape.length !== 2) throw new InputError("A .npy file must hold one 2-D Bayer mosaic.");
  const [h, w] = shape, types = { "<f4": Float32Array, "<u2": Uint16Array, "|u1": Uint8Array, "<f8": Float64Array };
  if (!types[dtype]) throw new InputError(`Unsupported .npy type ${dtype}. Use float32, float64, uint16 or uint8.`);
  const src = new types[dtype](buffer.slice(start, start + h * w * types[dtype].BYTES_PER_ELEMENT));
  let max = 0;
  for (const x of src) if (x > max) max = x;
  const scale = max > 1.5 ? 2 ** Math.ceil(Math.log2(max + 1)) - 1 : 1, H = h & ~1, W = w & ~1, bayer = new Float32Array(H * W);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) bayer[y * W + x] = src[fortran ? x * h + y : y * w + x] / scale;
  return { bayer, h: H, w: W, cfa: "rggb", colorMatrix: fallbackMatrix, neutral: null, source: "Bayer array",
           note: "No metadata: assumed RGGB and a generic colour matrix." };
}

export async function decodePicture(file, matrix) {
  let bmp;
  try { bmp = await createImageBitmap(file); } catch { throw new InputError("This file is not a readable image or RAW file."); }
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height)), w = Math.round(bmp.width * k), h = Math.round(bmp.height * k);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0, w, h);
  const px = g.getImageData(0, 0, w, h).data, srgb = new Float32Array(w * h * 3);
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) { srgb[j] = px[i] / 255; srgb[j + 1] = px[i + 1] / 255; srgb[j + 2] = px[i + 2] / 255; }
  const u = unprocess(srgb, h, w, matrix, SIMULATED_NEUTRAL), ref = new Float32Array(u.h * u.w * 3);
  for (let y = 0; y < u.h; y++) ref.set(srgb.subarray(y * w * 3, (y * w + u.w) * 3), y * u.w * 3);
  return { ...u, cfa: "rggb", colorMatrix: matrix, neutral: SIMULATED_NEUTRAL, wbSource: "known (simulated)", reference: ref, source: "Picture, inverted to simulated RAW",
           note: "This is not real sensor data. Use the noise slider to simulate a sensor; the reference is your picture." };
}

export async function decodeFile(file, matrix) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".dng")) {
    const buffer = await file.arrayBuffer();
    try { return decodeDng(buffer); } catch (e) {
      if (e instanceof RangeError) throw new InputError("This DNG looks damaged or truncated.");
      throw e;
    }
  }
  if (name.endsWith(".npy")) return decodeNpy(await file.arrayBuffer(), matrix);
  if (/\.(cr2|cr3|nef|arw|orf|rw2|raf|pef|srw)$/.test(name)) throw new InputError("Proprietary RAW formats are not supported here. Convert to DNG, or use a JPG or PNG.");
  return decodePicture(file, matrix);
}

const f16 = (() => {
  const t = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) {
    const s = i & 0x8000 ? -1 : 1, e = (i >> 10) & 31, m = i & 1023;
    t[i] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  return t;
})();

export function decodeF16(buffer) {
  const u = new Uint16Array(buffer), o = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) o[i] = f16[u[i]];
  return o;
}
