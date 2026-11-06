// Page logic for the GitHub Pages app. The INT8 model runs on this device through ONNX Runtime Web.
import * as I from "./isp.js";
import { decodeF16, decodeFile, InputError } from "./inputs.js";

const ORT = new URL("vendor/", location.href).href;
const MAX_PIXELS = 1_500_000;
const $ = id => document.getElementById(id);
const state = { samples: [], sessions: {}, cameras: null, file: null };

const nextPaint = () => new Promise(r => setTimeout(r, 30));

function status(text, kind = "") {
  const s = $("status");
  s.textContent = text;
  s.className = kind;
}

function draw(id, rgb, h, w) {
  const c = $(id), px = new Uint8ClampedArray(h * w * 4);
  for (let i = 0, j = 0; i < h * w; i++, j += 3) {
    px[4 * i] = rgb[j] * 255 + 0.5; px[4 * i + 1] = rgb[j + 1] * 255 + 0.5; px[4 * i + 2] = rgb[j + 2] * 255 + 0.5; px[4 * i + 3] = 255;
  }
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(new ImageData(px, w, h), 0, 0);
  c.closest("figure").hidden = false;
}

// What the sensor recorded: demosaiced, no colour processing, brightened for viewing.
function sensorPreview(frame) {
  const { canvas, hc, wc } = I.toCanonical(frame.bayer, frame.h, frame.w, frame.cfa, 2);
  const { p, hp, wp } = I.pack(canvas, hc, wc), out = new Float32Array(hp * wp * 3);
  let peak = 1e-3;
  for (let i = 0; i < p[0].length; i += 11) peak = Math.max(peak, p[1][i]);
  for (let i = 0; i < hp * wp; i++) {
    out[3 * i] = Math.sqrt(Math.min(Math.max(p[0][i] / peak, 0), 1));
    out[3 * i + 1] = Math.sqrt(Math.min(Math.max((p[1][i] + p[2][i]) / 2 / peak, 0), 1));
    out[3 * i + 2] = Math.sqrt(Math.min(Math.max(p[3][i] / peak, 0), 1));
  }
  return { rgb: out, h: hp, w: wp };
}

async function session(name) {
  if (!state.sessions[name]) {
    status("Loading the model…");
    state.sessions[name] = await ort.InferenceSession.create(`assets/${name}.onnx`, { executionProviders: ["wasm"] });
  }
  return state.sessions[name];
}

async function loadSample(meta) {
  const [bin, png] = await Promise.all([fetch(`assets/${meta.file}.f16`), fetch(`assets/${meta.file}.png`)]);
  if (!bin.ok || !png.ok) throw new Error("Could not download the sample. Check your connection.");
  const bmp = await createImageBitmap(await png.blob()), c = document.createElement("canvas");
  c.width = meta.w; c.height = meta.h;
  const g = c.getContext("2d");
  g.drawImage(bmp, 0, 0);
  const px = g.getImageData(0, 0, meta.w, meta.h).data, ref = new Float32Array(meta.w * meta.h * 3);
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) { ref[j] = px[i] / 255; ref[j + 1] = px[i + 1] / 255; ref[j + 2] = px[i + 2] / 255; }
  return { bayer: decodeF16(await bin.arrayBuffer()), h: meta.h, w: meta.w, cfa: meta.cfa, colorMatrix: meta.color_matrix,
           neutral: meta.neutral, noise: { a: meta.a, b: meta.b }, reference: ref, source: "SIDD-CC sample, unseen scene" };
}

const scoreOf = (out, ref, h, w) => {
  const a = I.centreCrop(out, h, w), b = I.centreCrop(ref, h, w);
  return `${I.psnr(a, b).toFixed(2)} dB, ΔE ${I.deltaE(a, b).toFixed(2)}`;
};

async function processImage() {
  const go = $("go");
  go.disabled = true;
  document.querySelectorAll("#result figure").forEach(f => { f.hidden = true; });
  $("report").replaceChildren();
  try {
    status("Reading the input…");
    await nextPaint();
    const matrix = state.cameras.IP.color_matrix;
    const frame = state.file ? await decodeFile(state.file, matrix) : await loadSample(state.samples[+$("sample").value]);
    if (Math.min(frame.h, frame.w) < 64) throw new InputError("The image is too small: at least 64 × 64 pixels are needed.");
    const k = I.binFrame(frame, MAX_PIXELS);
    const level = +$("noise").value;
    I.addSensorNoise(frame, 4e-4 * level, 4e-6 * level * level);
    const display = $("look").checked, d = I.analyse(frame, display);
    const scene = { cfa: frame.cfa, colorMatrix: frame.colorMatrix, neutral: d.neutral, a: d.a, b: d.b };
    const light = $("saver").checked, net = await session(light ? "student_s_int8" : "student_int8");

    const s = sensorPreview(frame);
    draw("c-raw", s.rgb, s.h, s.w);
    status("Running the network on this device…");
    await nextPaint();
    let t = performance.now();
    const run = async (data, h, w) => (await net.run({ x: new ort.Tensor("float32", data, [1, 6, h, w]) })).y.data;
    const hybrid = await I.hybridIsp(frame.bayer, frame.h, frame.w, scene, d.look, run, p => status(`Running the network on this device… ${Math.round(p * 100)}%`));
    const tHybrid = (performance.now() - t) / 1000;
    draw("c-hybrid", hybrid, frame.h, frame.w);

    status("Running the classical ISP…");
    await nextPaint();
    t = performance.now();
    const classical = I.classicalIsp(frame.bayer, frame.h, frame.w, scene, d.look);
    const tClassical = (performance.now() - t) / 1000;
    draw("c-classical", classical, frame.h, frame.w);

    const rows = [["Input", `${frame.source}, ${frame.w} × ${frame.h}, CFA ${frame.cfa.toUpperCase()}${k > 1 ? `, binned ${k}×${k}` : ""}`],
                  ["Noise estimate", `a = ${d.a.toExponential(2)}, b = ${d.b.toExponential(2)}, noise-to-signal ${d.rel.toFixed(3)}`],
                  ["Adaptive control", `${light ? "light" : "full"} network, exposure gain ×${d.look.exposure.toFixed(2)}, sharpening ${d.look.sharpen.toFixed(2)}`],
                  ["White balance", `${d.wbSource}, neutral ${d.neutral.map(x => x.toFixed(3)).join(", ")}`],
                  ["Time on this device", `hybrid ${tHybrid.toFixed(2)} s, classical ${tClassical.toFixed(2)} s`]];
    if (frame.reference) {
      draw("c-ref", frame.reference, frame.h, frame.w);
      if (!display) {
        rows.push(["Hybrid vs reference", scoreOf(hybrid, frame.reference, frame.h, frame.w)]);
        rows.push(["Classical vs reference", scoreOf(classical, frame.reference, frame.h, frame.w)]);
      } else rows.push(["Scores", "Turn off the display look to score against the reference."]);
    }
    if (frame.note) rows.push(["Note", frame.note]);
    $("report").replaceChildren(...rows.flatMap(([a, b]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = a; dd.textContent = b;
      return [dt, dd];
    }));
    window.__result = { hybrid, classical, frame };
    status("Done.", "ok");
  } catch (e) {
    console.error(e);
    status(e instanceof InputError ? e.message : `Something went wrong: ${e.message}`, "bad");
  } finally {
    go.disabled = false;
  }
}

function results(summary) {
  const t = $("table");
  const head = t.createTHead().insertRow();
  summary.headers.forEach(h => { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); });
  const body = t.createTBody();
  summary.rows.forEach(r => {
    const tr = body.insertRow();
    if (r[0] === "Hybrid student INT8") tr.className = "mine";
    r.forEach(v => { tr.insertCell().textContent = v; });
  });
}

async function init() {
  try {
    if (!window.ort) throw new Error("Could not load ONNX Runtime. Check your connection and reload.");
    ort.env.wasm.wasmPaths = ORT;
    ort.env.wasm.numThreads = 1; // GitHub Pages cannot send the headers that browser threads need
    const [samples, cameras, summary] = await Promise.all(["samples", "cameras", "summary"].map(n => fetch(`assets/${n}.json`).then(r => r.json())));
    state.samples = samples; state.cameras = cameras;
    samples.forEach((s, i) => $("sample").add(new Option(s.name, i)));
    $("sample").value = Math.max(samples.findIndex(s => s.file.startsWith("N6")), 0);
    results(summary);
    $("go").disabled = false;
    status("Ready. Press Process.");
  } catch (e) {
    status(e.message, "bad");
  }
  $("go").onclick = processImage;
  $("noise").oninput = () => { $("noise-v").textContent = $("noise").value; };
  $("file").onchange = () => {
    state.file = $("file").files[0] || null;
    $("clear").hidden = !state.file;
    $("sample").disabled = !!state.file;
  };
  $("clear").onclick = () => { $("file").value = ""; $("file").onchange(); };
}

init();
