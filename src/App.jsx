import { useState, useEffect, useRef, useCallback } from "react";

/*
 * Social Hierarchy & Deception Simulator — "When Deception Spreads" (ALife 2026)
 * Two conditions: A (Honest Baseline) & B (Evolvable Deception).
 * α_i = C_i  (Eq.1): the rank signal IS true (innate) capability — no experience/ξ term, no τ.
 * Speaker selection: P^i(S) ∝ softmax(β·α̂_i)  (Eq.3).
 * Deception spreads via 3 channels: self-reinforcement (Ch1, Eqs.7–10),
 *   observational learning (Ch2, Eq.11), vertical inheritance (Ch3, Eqs.12–13).
 * NOTE: legacy ξ/Hill-function & ω-weight scaffolding remains in the code but is kept
 *   dormant (omegaXi=0, hillBeta=0, expListenerWt=0 ⇒ α=C) and is hidden from the UI.
 */

const WP = 950, HP = 700;
const CX = WP / 2, CY = HP / 2;

// Villages: 5 circles, radii matching Python scale (scaled from 1100×850 → 700×700)
const S = 700 / 1100; // scale factor
const VILLAGES = [
  { x: 250, y: 175, r: 52 }, // 左上
  { x: 250, y: 525, r: 52 }, // 左下
  { x: 475, y: 350, r: 66 }, // 中心（半径略大）
  { x: 700, y: 175, r: 52 }, // 右上
  { x: 700, y: 525, r: 52 }, // 右下
];

const K = 250;
const INIT_N = 100; // Python uses 100
const FOOD_ENERGY = 5; // each locked food = 5 energy, split among 3
const ENERGY_DECAY = 0.035; // Python: 0.035/tick
const AGE_RATE = 0.001; // Python: age += 0.001/tick
const COOP_NEARBY_AGENT = 100 * S; // Python: 100px scaled
const COOP_NEARBY_FOOD = 70 * S; // Python: 70px scaled
const SENSE_RADIUS = 200; // agents can see food from far
const MOVE_SPEED = 15 * S; // Python: 15px/tick scaled
const IDLE_MAX_SPEED = 10 * S; // Python: [-10,10] scaled

// Capability: N(100, σ_c)
const CAP_MEAN = 100;
const CAP_STD = 0.05;
const H_COEFF = 0.9;
const MUT_P = 0.2;
const MUT_S = 1;
const SOFTMAX_BETA = 4; // β in Eq.(3) — paper default 4.0 (Table 2)

// ═══ Model spec — "When Deception Spreads" (ALife 2026) ═══
// α_i = C_i  (Eq.1) — true capability == innate capability; NO experience/ξ term.
//   (The ξ / Hill-function scaffolding below is kept dormant: par.omegaXi=0,
//    hillBeta=0, hillKappa=0, expListenerWt=0 ⇒ α = C, matching the paper.)
// Cond A: deception OFF (honest baseline)   Cond B: evolvable deception ON
// Speaker selection: P^i(S) ∝ softmax(β·α̂_i)  (Eq.3)
//
// NOTE: The constants below mirror paper Table 2 for reference ONLY.
// The LIVE simulation reads values from the `par` state object, not these.
const ETA_XI = 0.06;
const LAMBDA_XI = 0.001;
// ══ Evolvable Deception Mechanism — Table 2 reference values ══
// Speaker self-reinforcement (Ch1, Eqs.7–10)
const ETA_P_PLUS = 0.10;   // η⁺_p positive self-reinforcement (Table 2)
const ETA_M_PLUS = 0.20;   // η⁺_m
const ETA_P_MINUS = 0.05;  // η⁻_p negative self-reinforcement
const ETA_M_MINUS = 0.10;  // η⁻_m
// Listener observational learning (Ch2, Eq.11)
const ETA_P_OBS = 0.10;    // η^obs_p
const ETA_M_OBS = 0.10;    // η^obs_m
const LAMBDA_P_DECAY = 0.0; // ψ_p listener forgetting (Table 2: 0)
const LAMBDA_M_DECAY = 0.0; // ψ_m
// Baseline decay (Eq.10)
const DELTA_P = 0.001;     // δ_p (Table 2)
const DELTA_M = 0.001;     // δ_m
const P_BASELINE = 0;
const M_BASELINE = 0;
// Detection (Eq.4)
const K_DETECT = 3.0;      // k_d detection sigmoid steepness (Table 2)
// Soft caps and scaling
const M_MAX = 10;          // M_max deception magnitude upper bound (Table 2)
const IMITATION_SCALE = 1.0; // κ_imit (Eq.11)
// Inheritance (Ch3, Eqs.12–13)
const SIGMA_P = 0.05;      // σ_p mutation noise (Table 2)
const SIGMA_M = 0.5;       // σ_m
const CHILD_BETA = 0.9;    // β_c parental transmission fidelity (Table 2)
// Legacy aliases (kept for backward compat in case anything references them)
const ETA_P = ETA_P_PLUS;
const ETA_M = ETA_M_PLUS;

// Death: every 300 ticks, kill 2-3 by priority score
const DEATH_INTERVAL = 300;

// ══════════════════════════════════════════════════════
// §Reproducibility: Seeded PRNG (mulberry32)
// ══════════════════════════════════════════════════════
function mulberry32(seed) {
  let s = seed >>> 0;
  return function() {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let _rand = Math.random; // replaced with seeded version in init()

let _id = 0;
const uid = () => ++_id;
const rng = (a, b) => a + _rand() * (b - a);
const dst = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pick = arr => arr[Math.floor(_rand() * arr.length)];
const lerp = (a, b, t) => a + (b - a) * t;
const gauss = (m, s) => { const u = _rand() || 1e-10, v = _rand(); return m + s * Math.sqrt(-2 * Math.log(u)) * Math.cos(6.28318 * v); };
const sigmoid = x => 1 / (1 + Math.exp(-x));

// Gini coefficient: G = Σ_i Σ_j |x_i - x_j| / (2·N·Σ_i x_i)
// Returns 0 (perfect equality) to ~1 (maximum inequality)
function computeGini(values) {
  const n = values.length;
  if (n < 2) return 0;
  // Shift to non-negative: Gini requires non-negative values
  const mn = Math.min(...values);
  const shifted = mn < 0 ? values.map(v => v - mn + 1e-12) : values;
  const sumAll = shifted.reduce((s, v) => s + v, 0);
  if (sumAll < 1e-15) return 0;
  // Efficient O(n log n) via sorted array:
  // G = (2·Σ_i (i+1)·x_sorted[i]) / (n·Σ x) − (n+1)/n
  const sorted = [...shifted].sort((a, b) => a - b);
  let weightedSum = 0;
  for (let i = 0; i < n; i++) weightedSum += (i + 1) * sorted[i];
  return (2 * weightedSum) / (n * sumAll) - (n + 1) / n;
}

function insideVillage(x, y) {
  for (const v of VILLAGES) if (Math.hypot(x - v.x, y - v.y) <= v.r) return v;
  return null;
}

// Softmax with temperature β (matching Python)
function softmaxBeta(vals, beta) {
  const scaled = vals.map(v => beta * v);
  const mx = Math.max(...scaled);
  const ex = scaled.map(v => Math.exp(v - mx));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / s);
}
// Plain softmax (beta=1) for diagnostics
function softmax(vals) {
  const mx = Math.max(...vals);
  const ex = vals.map(v => Math.exp(v - mx));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(e => e / s);
}
function sampleW(p) { let r = _rand(), c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (r < c) return i; } return p.length - 1; }

// ══════════════════════════════════════════════════════════════════════════════
// §Diagnostics: Compare raw α vs normalized α (from pasted-text methodology)
// ══════════════════════════════════════════════════════════════════════════════

// Diagnostic function: Spearman, KL divergence, top-k overlap
// inputs: agents with alpha_raw, alpha_norm fields
function diagnostics_compare_alphas(agents, beta = 1.5, topk = [5, 10]) {
  const raw = agents.map(a => a.alpha_raw);
  const norm = agents.map(a => a.alpha_norm); // 0..1 normalized values

  // ranks
  function rankArray(arr) {
    return arr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((x, idx) => ({ idx: x[1], rank: idx + 1 }));
  }
  const rawRank = rankArray(raw).sort((a, b) => a.idx - b.idx).map(x => x.rank);
  const normRank = rankArray(norm).sort((a, b) => a.idx - b.idx).map(x => x.rank);

  // spearman correlation
  const n = agents.length;
  let d2sum = 0;
  for (let i = 0; i < n; i++) { let d = rawRank[i] - normRank[i]; d2sum += d * d; }
  const spearman = 1 - (6 * d2sum) / (n * (n * n - 1));

  // selection probs
  const P_raw = softmax(raw.map(x => x * beta));
  const P_norm = softmax(norm.map(x => x * beta));

  // KL divergence KL(P_raw || P_norm)
  function kl(p, q) {
    let s = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0) s += p[i] * Math.log((p[i] + 1e-12) / (q[i] + 1e-12));
    }
    return s;
  }
  const KL = kl(P_raw, P_norm);

  // top-k overlap
  function topk_indices(arr, k) {
    return arr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1]);
  }
  const overlap = {};
  for (const k of topk) {
    const r = topk_indices(P_raw, k), s = topk_indices(P_norm, k);
    const inter = r.filter(x => s.includes(x)).length;
    overlap[`top${k}`] = inter / k;
  }

  return { spearman, KL, overlap, P_raw, P_norm };
}

// Calibration: find optimal scale to match raw distribution
function fit_scale_to_match_raw(agents, beta_base = 1.0, scales = [0.5, 1, 1.5, 2, 3, 5, 10, 15, 20, 50, 100]) {
  const raw = agents.map(a => a.alpha_raw);
  const P_raw = softmax(raw.map(x => x * beta_base));
  const norm = agents.map(a => a.alpha_norm); // 0..1

  function kl(p, q) {
    let ss = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0) ss += p[i] * Math.log((p[i] + 1e-12) / (q[i] + 1e-12));
    }
    return ss;
  }

  let best = { scale: null, KL: Infinity, P_norm: null };
  for (const s of scales) {
    const Pn = softmax(norm.map(x => x * (s * beta_base)));
    const k = kl(P_raw, Pn);
    if (k < best.KL) best = { scale: s, KL: k, P_norm: Pn };
  }
  return best; // 返回最合适的 scale
}

// Normalize α to [0,1] using robust IQR-based method (tanh)
function normalizeAlphaToDisplay(alphaRaw, allAlphas) {
  if (allAlphas.length < 4) return clamp(alphaRaw / 200, 0, 1); // fallback
  const sorted = [...allAlphas].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const median = sorted[Math.floor(sorted.length * 0.5)];
  const iqr = Math.max(q3 - q1, 1e-6);
  // tanh normalization: maps to roughly (-1,1) then shift to (0,1)
  return clamp(0.5 + 0.5 * Math.tanh((alphaRaw - median) / iqr), 0, 1);
}

// Food: uniform in margin, not in village (matching Python)
function mkFood() {
  const mx = WP / 8, my = HP / 8;
  for (let i = 0; i < 50; i++) {
    const x = rng(mx, WP - mx), y = rng(my, HP - my);
    if (!insideVillage(x, y)) return { id: uid(), x, y, ph: _rand() * 6.28, energy: FOOD_ENERGY };
  }
  return { id: uid(), x: rng(mx, WP - mx), y: rng(my, HP - my), ph: 0, energy: FOOD_ENERGY };
}

function mkAgent(x, y, gender, cap, capStd) {
  const c = cap !== undefined ? cap : Math.max(1, gauss(CAP_MEAN, capStd !== undefined ? capStd : CAP_STD));
  return {
    id: uid(), x, y, gender,
    capability: c,        // C^i — innate, fixed at birth
    xi: 0,                // ξ ∈ [0,1] — leadership experience (Eq.12)
    alpha: c,             // α = ω_C·C + ω_ξ·ξ (recomputed) — used for decision
    alpha_raw: c,         // §Diagnostics: store raw α (before any normalization)
    alpha_norm: 0.5,      // §Diagnostics: normalized α ∈ [0,1] for display only
    p: 0,                 // p ∈ [0,1] — deception propensity (Eq.14)
    m: 0,                 // m ≥ 0 — deception magnitude (Eq.15)
    alpha_reported: 0,    // α̂ — reported α (after noise + lie bias)
    energy: rng(7, 8),
    age: 0, alive: true, gen: 0,
    action: "IDLE", prevAction: "IDLE",
    target: null, inVillage: null, targetVillage: null,
    sx: rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED),
    sy: rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED),
    ax: _rand() < 0.5 ? -0.7 : 0.7,
    ay: _rand() < 0.5 ? -0.7 : 0.7,
    teamMembers: [], speaker: null, listeners: [],
    timeSpeaker: 0, timeListener: 0,
    n_exp: 0,       // n_i — raw experience counter (speaker+1, listener+γ_L) for Hill-function ξ
    targetFood: null, cooldown: 0,
    coopCount: 0, coopSuccess: 0, coopFail: 0,
    lieCount: 0,
    lastGamma: 0, lastAstar: 0,  // §Γ mechanism tracking
    flash: 0, fCol: null,
  };
}

// TI — Trophic Incoherence F (paper Eq.15; Rodgers, Tino & Johnson 2023/2024)
// Returns the RAW (un-normalised) incoherence F = Σ w_ij·(h_j - h_i - 1)² / Σ w_ij.
// Lower F = more layered/consistent hierarchy; higher F = more structural disorder.
// NOTE: this function deliberately returns raw F (NOT F/(1+F)); the paper reports raw F
//   (typically ~0.3 under the honest baseline, rising to ~0.75–0.8 under Full deception).
function computeTI(wMat, ids) {
  const n = ids.length; if (n < 3) return NaN;

  // Build adjacency only for nodes that have at least one edge
  const idSet = new Set(ids);
  const edgeList = [];
  for (const [k, val] of wMat.entries()) {
    const parts = k.split("-").map(Number);
    const a = parts[0], b = parts[1];
    if (!idSet.has(a) || !idSet.has(b)) continue;
    edgeList.push({ a, b, w: val });
  }
  if (edgeList.length < 2) return NaN; // not enough edges to compute

  // Collect only nodes that participate in at least one edge
  const nodeSet = new Set();
  for (const e of edgeList) { nodeSet.add(e.a); nodeSet.add(e.b); }
  const nodeArr = [...nodeSet];
  const m = nodeArr.length;
  if (m < 3) return NaN;

  const ix = {}; nodeArr.forEach((id, i) => ix[id] = i);
  const W = Array.from({ length: m }, () => new Float64Array(m));
  for (const e of edgeList) W[ix[e.a]][ix[e.b]] = e.w;

  // Step 1: Compute u (in+out weight) and v (in-out weight) for each connected node
  const u = new Float64Array(m), v = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let inW = 0, outW = 0;
    for (let j = 0; j < m; j++) {
      inW += W[j][i];   // edges into i
      outW += W[i][j];  // edges out of i
    }
    u[i] = inW + outW;
    v[i] = inW - outW;
  }

  // Step 2: Solve Λh = v via Gauss-Seidel with SOR (ω=1.4), 500 iterations
  // Λ = diag(u) - (W + W^T) is the symmetrized graph Laplacian (singular)
  // We center h after each sweep to avoid drift in the null space
  const h = new Float64Array(m);
  const SOR_OMEGA = 1.4;
  const MAX_ITER = 500;
  for (let it = 0; it < MAX_ITER; it++) {
    for (let i = 0; i < m; i++) {
      if (u[i] < 1e-12) continue;
      let s = v[i];
      for (let j = 0; j < m; j++) if (j !== i) s += (W[i][j] + W[j][i]) * h[j];
      const hNew = s / u[i];
      h[i] = h[i] + SOR_OMEGA * (hNew - h[i]); // SOR update
    }
    // Center h to prevent drift (Λ has null space span{1})
    if (it % 10 === 0) {
      let mean = 0; for (let i = 0; i < m; i++) mean += h[i]; mean /= m;
      for (let i = 0; i < m; i++) h[i] -= mean;
    }
  }
  // Final centering
  let mean = 0; for (let i = 0; i < m; i++) mean += h[i]; mean /= m;
  for (let i = 0; i < m; i++) h[i] -= mean;

  // Step 3: Compute F(h) = Σ w_ij · (h_j - h_i - 1)² / Σ w_ij
  let totalW = 0, fh = 0;
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) if (W[i][j] > 0) {
    totalW += W[i][j];
    fh += W[i][j] * (h[j] - h[i] - 1) ** 2;
  }
  const rawFh = totalW > 0 ? fh / totalW : 0;

  // Return RAW trophic incoherence F (paper Eq.15). No [0,1] normalisation is applied.
  return rawFh;
}

// Sigmoid fertility (matching Python)
function calcLambda(nAlive) {
  const theta = 0.01, beta1 = 0.5, beta2 = 0.5, ageRate = 0.1, estrusRate = 0.1;
  const fert = 1 / (1 + Math.exp(-(theta + beta1 * ageRate + beta2 * estrusRate)));
  return Math.max(0, fert * (1 - nAlive / K));
}

// ══════════════════════════════
// §Theme tokens + global reset. Light/dark switch via [data-theme] on the root.
// Structural surfaces & reusable text colours read these vars so one toggle re-skins everything.
const THEME_CSS = `
  html, body { margin: 0 !important; padding: 0 !important; background: #070b08; }
  * { box-sizing: border-box; }
  [data-theme] {
    --accent: #6ee7b7; --warn: #fbbf24; --danger: #ef4444; --violet: #c084fc; --sky: #7dd3fc;
  }
  [data-theme="dark"] {
    --bg: #070b08; --panel: rgba(12,18,13,0.96); --panel-solid: #0b110c;
    --surface: #0b110c; --input: #0a0e0b; --line: rgba(120,180,100,0.14);
    --line-soft: rgba(120,180,100,0.07); --text: #c6d6bd; --text2: #93a48c;
    --text-dim: #647a5c; --text-faint: #43543c; --shadow: rgba(0,0,0,0.4);
    --arena-grad: radial-gradient(ellipse at center, #0d150e, #060a07);
    --food-swatch: #a98bd6; --village-swatch: rgba(200,170,110,0.6);
  }
  [data-theme="light"] {
    --bg: #e9ede3; --panel: #ffffff; --panel-solid: #ffffff;
    --surface: #f4f6f0; --input: #ffffff; --line: rgba(70,90,55,0.20);
    --line-soft: rgba(70,90,55,0.10); --text: #1d2719; --text2: #45533d;
    --text-dim: #69785f; --text-faint: #9aa890; --shadow: rgba(60,80,40,0.12);
    --arena-grad: radial-gradient(ellipse at center, #eef2e8, #dfe6d6);
    --food-swatch: #7c5cc4; --village-swatch: rgba(150,120,60,0.6);
  }
  .side-panel::-webkit-scrollbar { width: 6px; }
  .side-panel::-webkit-scrollbar-track { background: transparent; }
  .side-panel::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }
  .side-panel::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
  input[type="range"] { accent-color: var(--accent); }
`;

// ══════════════════════════════
export default function App() {
  const cvs = useRef(null), chart = useRef(null), tiC = useRef(null);
  const popC = useRef(null), alphaC = useRef(null), energyC = useRef(null), coopC = useRef(null), xiC = useRef(null);
  const mChartC = useRef(null), varAlphaC = useRef(null), spearRepC = useRef(null);
  const giniAlphaC = useRef(null), netDensityC = useRef(null);
  const alphaDistC = useRef(null);
  const alphaRepDistC = useRef(null);
  const simR = useRef(null), pRef = useRef(null);
  // §Script: refs that are always current inside step() callbacks
  const condRef = useRef("A");
  const setCondRef = useRef(null); // will hold setCond
  const scriptR = useRef([]); // active script events
  const firedR = useRef(new Set()); // fired event ids (prevents re-fire)
  const autoExportR = useRef(false); // flag to trigger auto-export on next render
  // §CSV Export: lightweight time-series buffers (never sliced, persist across cond switches)
  const coreTSref = useRef([]); // core TS: tick, TI, N, meanAlpha, varAlpha, meanXi, condition
  const decTSref = useRef([]);  // deception TS: tick, meanP, meanM, varP, varM, spearRawVsRep, fracNearTop, coopRate, meanGamma, condition
  const [run, setRun] = useState(false);
  const [spd, setSpd] = useState(1);
  const [tick, setTick] = useState(0);
  const [cnt, setCnt] = useState({});
  const [evts, setEvts] = useState([]);
  const [sel, setSel] = useState(null);
  const [hist, setHist] = useState([]);
  const [ti, setTi] = useState(0);
  const [tiH, setTiH] = useState([]);
  const [diag, setDiag] = useState({ spearman: 0, KL: 0, overlap: {}, bestScale: 1, bestKL: Infinity, varAlphaTrue: 0, varAlphaHat: 0, spearRawVsRep: 1, fracNearTop: 0.05, meanP: 0, meanM: 0, varP: 0, varM: 0, isPooling: false, meanGamma: 0, meanAstar: 0 });
  // §Reproducibility
  const [seed, setSeed] = useState(() => (Date.now() ^ (Math.random() * 0xFFFFFFFF)) >>> 0);
  const [seedInput, setSeedInput] = useState("");
  // §Analysis export: full diagnostic history (never sliced, never cleared on cond switch)
  const [diagH, setDiagH] = useState([]);
  const diagHref = useRef([]); // mirror of diagH for use in autoExport (avoids stale closure)
  const tiHref = useRef([]);   // mirror of tiH for autoExport
  const [showAnalysis, setShowAnalysis] = useState(false);
  // §Script: default experiment — A→B at 30000, autoExport at 60000
  const DEFAULT_SCRIPT = [
    { id: "s1", t: 40000, action: "setCond", value: "B", label: "Switch → Cond B (Deception)" },
    { id: "s2", t: 100000, action: "autoExport", label: "Auto-save all figures & CSVs" },
  ];
  const [script, setScript] = useState(DEFAULT_SCRIPT);
  const [scriptEnabled, setScriptEnabled] = useState(true);
  const [showScriptEditor, setShowScriptEditor] = useState(false);
  const [scriptLog, setScriptLog] = useState([]); // fired event log for display
  // Hidden off-screen canvases for auto-export PNG rendering
  const hc1 = useRef(null), hc2 = useRef(null), hc3 = useRef(null);
  const hc4 = useRef(null), hc5 = useRef(null), hc6 = useRef(null);
  const [cond, setCond] = useState("A"); // A=honest, B=deception (Paper I only, no C)
  // §Theme: dark (default) / light, driven by CSS variables on the root via data-theme
  const [theme, setTheme] = useState("dark");
  const themeRef = useRef(theme);
  themeRef.current = theme;
  // Canvas colour tokens (JS — canvas can't read CSS vars). Kept in sync with the <style> block.
  const canvasTheme = (theme === "light")
    ? { arena: "#e7ede1", grid: "rgba(60,90,50,0.06)", village: "rgba(150,120,60,0.55)", villageFill: "rgba(190,160,90,0.10)",
        food: "#7c5cc4", foodGlow: "rgba(124,92,196,0.35)", text: "rgba(30,40,28,0.55)", agentStroke: "rgba(30,40,28,0.28)", chartBg: "#f4f6f1" }
    : { arena: "#0c130c", grid: "rgba(120,180,100,0.05)", village: "rgba(200,170,110,0.28)", villageFill: "rgba(200,160,90,0.06)",
        food: "#a98bd6", foodGlow: "rgba(169,139,214,0.40)", text: "rgba(220,235,210,0.45)", agentStroke: "rgba(0,0,0,0.35)", chartBg: "#0b110c" };
  const ctRef = useRef(canvasTheme);
  ctRef.current = canvasTheme;
  condRef.current = cond; // always current inside step()
  setCondRef.current = setCond;
  const lieOn = cond !== "A";
  const [par, setPar] = useState({
    capStd: 0.05, mutS: 1,  // σ_C = 0.05 (paper Table 2). 0 → no initial capability heterogeneity
    omegaC: 1.00, omegaXi: 0.00,
    thetaD: 1.5, commNoise: 0.0,
    // Evolvable deception params — baseline from §Document spec
    kDetect: 3.0,
    etaPplus: 0.10, etaMplus: 0.20,
    etaPminus: 0.05, etaMminus: 0.10,
    gamma: 200, // γ for Γ=tanh(γ·A*/max(E,1)) — calibrated: share_diff≈0.003 → Γ≈tanh(0.6)≈0.54
    etaPobs: 0.10, etaMobs: 0.10,
    lambdaPdecay: 0.00, lambdaMdecay: 0.00,
    deltaP: 0.001, deltaM: 0.001,
    Mmax: 10, imitScale: 1.0,
    childBeta: 0.9, sigmaMutP: 0.05, sigmaMutM: 0.5,
    hardDetect: false,
    // Softmax selection temperature β (Eq.3) — R3.3 ablation: lower β weakens selection amplification
    softmaxBeta: 4.0,
    // Cooperation success (sigmoid threshold)
    T: 150, kappa: 5.0,
    // ═══ Hill-function Experience Mechanism (Matthew Effect) ═══
    // n_i accumulation weights
    expListenerWt: 0,   // γ_L: listener experience weight (speaker always +1)
    // Hill function: ξ = n^hillBeta / (n^hillBeta + hillKappa^hillBeta)
    hillBeta: 0,        // Hill coefficient β_H (steepness; >1 → S-curve / threshold effect)
    hillKappa: 0,        // Half-saturation κ_H (n at which ξ=0.5)
    // Pooling detection thresholds
    poolingVarPThresh: 1e-3, poolingVarMThresh: 1e-2, poolingStableWindow: 500,
    // §Ignition: initial heterogeneous perturbation when switching A→B
    ignitionP: 0.05,  // E[Δp] per agent (actual: U(0, 2·ignitionP))
    ignitionM: 0.5,   // E[Δm] per agent (actual: U(0, 2·ignitionM))
  });
  pRef.current = par;
  scriptR.current = script; // always current inside step()

  // Constrained ω setter: ω_C + ω_ξ = 1 (Paper I: no τ)
  const setOmega = useCallback((key, newVal) => {
    setPar(p => {
      const next = { ...p, [key]: newVal };
      if (key === "omegaC") next.omegaXi = Math.max(0, 1 - newVal);
      else next.omegaC = Math.max(0, 1 - newVal);
      return next;
    });
  }, []);

  function recomputeAlpha(ag, p, condition, allAlphas = null) {
    // Paper Eq.1: α_i = C_i. The ω_ξ·ξ term is dormant (par.omegaXi=0 by default ⇒ α=C).
    ag.alpha = p.omegaC * ag.capability + p.omegaXi * ag.xi;
    // §Diagnostics: store raw α for comparison
    ag.alpha_raw = ag.alpha;
    // §Diagnostics: compute normalized α for display (if allAlphas provided)
    if (allAlphas && allAlphas.length > 3) {
      ag.alpha_norm = normalizeAlphaToDisplay(ag.alpha_raw, allAlphas);
    }
  }

  // §Ignition: inject heterogeneous initial perturbation when switching A→B
  // p_i += U(0, 2·ignitionP), m_i += U(0, 2·ignitionM) — provides initial
  // heterogeneity so all three channels (reinforcement, learning, inheritance) can bootstrap
  function applyIgnition() {
    const Z = simR.current; if (!Z) return;
    const p = pRef.current;
    const alive = Z.agents.filter(a => a.alive);
    alive.forEach(a => {
      a.p = clamp(a.p + _rand() * 2 * p.ignitionP, 0, 1);
      a.m = Math.max(0, a.m + _rand() * 2 * p.ignitionM);
    });
  }

  const upd = useCallback(() => {
    const s = simR.current; if (!s) return;
    const c = { total: 0, food: s.food.length, m: 0, f: 0, n: 0, lo: 0, mi: 0, hi: 0, ap: 0, am: 0, ax: 0 };
    let ps = 0, ms = 0, xs = 0;
    s.agents.forEach(a => {
      if (!a.alive) return;
      c.total++;
      a.gender === "M" ? c.m++ : c.f++;
      a.p < 0.1 ? c.n++ : a.p < 0.35 ? c.lo++ : a.p < 0.65 ? c.mi++ : c.hi++;
      ps += a.p; ms += a.m; xs += a.xi;
    });
    c.ap = c.total > 0 ? ps / c.total : 0;
    c.am = c.total > 0 ? ms / c.total : 0;
    c.ax = c.total > 0 ? xs / c.total : 0;
    setCnt(c);
  }, []);

  const init = useCallback((overrideSeed, clearHistory = true) => {
    _id = 0;
    const usedSeed = overrideSeed !== undefined ? overrideSeed : seed;
    _rand = mulberry32(usedSeed);
    if (overrideSeed !== undefined) setSeed(overrideSeed);
    const p = pRef.current;
    const agents = [];
    for (let i = 0; i < INIT_N; i++) {
      agents.push(mkAgent(rng(30, WP - 30), rng(30, HP - 30), _rand() < 0.5 ? "M" : "F", undefined, p.capStd));
    }
    const food = [];
    for (let i = 0; i < 25; i++) food.push(mkFood());
    simR.current = { agents, food, events: [], tick: 0, wMat: new Map(), anims: [], totalFoodEnergy: () => food.reduce((s, f) => s + f.energy, 0) };
    setTick(0); setEvts([]); setSel(null); setTi(0);
    firedR.current = new Set(); // reset script fired state
    setScriptLog([]);
    if (clearHistory) { setHist([]); setTiH([]); setDiagH([]); coreTSref.current = []; decTSref.current = []; diagHref.current = []; tiHref.current = []; }
    upd();
  }, [upd, seed]);

  useEffect(() => { init(); }, [init]);

  const step = useCallback(() => {
    const Z = simR.current; if (!Z) return;
    const p = pRef.current;
    Z.tick++;
    const nev = [];
    Z.anims = Z.anims.filter(c => c.ttl-- > 0);

    // ── Food spawn: if total energy < 100, spawn 15-16 (Python logic) ──
    const totalE = Z.food.reduce((s, f) => s + f.energy, 0);
    if (totalE < 200) {
      const num = 20 + Math.floor(_rand() * 2);
      for (let i = 0; i < num; i++) Z.food.push(mkFood());
    }

    const alive = Z.agents.filter(a => a.alive);
    const nAlive = alive.length;

    // ══ PHASE 1: State check (every tick, matching Python check_state) ══
    for (const a of alive) {
      a.age += AGE_RATE;
      a.energy -= ENERGY_DECAY;
      // E3 guard (paper Fig.1 / Table 1): an agent coordinating inside the cooperation
      // region receives a minimal +1/tick energy supplement when E ≤ 1, preventing
      // starvation during team formation / synchronised joint action.
      if ((a.action === "MOVE_COOP" || a.action === "PAUSE" || a.action === "COOP") && a.energy <= 1) {
        a.energy += 1;
      }
      if (a.flash > 0) a.flash -= 0.02;
      if (a.cooldown > 0) a.cooldown--;
      a.prevAction = a.action;

      // Dead check
      if (a.energy <= 0) { a.alive = false; a.action = "DEAD"; continue; }

      // Skip if in cooperation flow
      if (a.action === "MOVE_COOP" || a.action === "COOP" || a.action === "PAUSE") continue;

      // Village check
      a.inVillage = insideVillage(a.x, a.y);

      // Python priority: COOP/MOVE_COOP/PAUSE > E>10 village logic > E<10 cooperation > IDLE
      if (a.energy > 10) {
        if (a.inVillage) {
          a.action = "REPRODUCING";
        } else {
          // Find closest village
          let cv = null, cd = Infinity;
          for (const v of VILLAGES) { const d = dst(a, v); if (d < cd) { cd = d; cv = v; } }
          a.action = "MOVING_VILLAGE";
          a.targetVillage = cv;
        }
      } else {
        // E < 10: check cooperation conditions
        const nearbyFood = Z.food.filter(f => f.energy > 0 && dst(a, f) <= COOP_NEARBY_FOOD);
        const nearbyIdle = alive.filter(b => b.id !== a.id && b.action === "IDLE" && dst(a, b) <= COOP_NEARBY_AGENT);
        if (nearbyFood.length >= 1 && nearbyIdle.length >= 2 && a.cooldown <= 0) {
          a.action = "INIT_COOP";
          // Pick target food (closest)
          a.targetFood = nearbyFood.reduce((best, f) => dst(a, f) < dst(a, best) ? f : best, nearbyFood[0]);
        } else {
          a.action = "IDLE";
        }
      }

      // (deception adapts via Steps 5-7 in cooperation resolution)
    }

    // ══ PHASE 2: Initiate cooperation (3 agents) ══
    const initiators = alive.filter(a => a.action === "INIT_COOP" && a.cooldown <= 0);
    const used = new Set();

    for (const ini of initiators) {
      if (used.has(ini.id)) continue;
      const nearbyIdle = alive.filter(b =>
        b.id !== ini.id && !used.has(b.id) &&
        b.action === "IDLE" && dst(ini, b) <= COOP_NEARBY_AGENT
      );
      if (nearbyIdle.length < 2) { ini.action = "IDLE"; continue; }

      // Python: random.sample(nearby, k=2) — RANDOM selection
      const shuffled = [...nearbyIdle].sort(() => _rand() - 0.5);
      const team = [ini, shuffled[0], shuffled[1]];
      team.forEach(a => used.add(a.id));

      if (!ini.targetFood || ini.targetFood.energy <= 0) {
        team.forEach(a => { a.action = "IDLE"; a.cooldown = 5; });
        continue;
      }

      // Assign team
      team.forEach(a => {
        a.teamMembers = team;
        a.targetFood = ini.targetFood;
        a.action = "MOVE_COOP";
      });
    }

    // ══ Move toward food & check arrival ══
    for (const a of alive) {
      if (a.action === "MOVE_COOP" && a.targetFood) {
        const d = dst(a, a.targetFood);
        if (d < 12) {
          a.action = "PAUSE";
        } else {
          // Move toward food
          const dx = a.targetFood.x - a.x, dy = a.targetFood.y - a.y;
          const dd = Math.hypot(dx, dy);
          if (dd > 0) { a.x += (dx / dd) * MOVE_SPEED; a.y += (dy / dd) * MOVE_SPEED; }
        }
      }
    }

    // ══ Check if all team paused → execute cooperation ══
    const pausedTeams = new Set();
    for (const a of alive) {
      if (a.action !== "PAUSE" || !a.teamMembers.length) continue;
      const teamKey = a.teamMembers.map(m => m.id).sort().join("-");
      if (pausedTeams.has(teamKey)) continue;

      const allPaused = a.teamMembers.every(m => m.action === "PAUSE");
      const anyLeft = a.teamMembers.some(m => !m.alive || (m.action !== "PAUSE" && m.action !== "MOVE_COOP"));

      if (anyLeft) {
        a.teamMembers.forEach(m => { m.action = "IDLE"; m.teamMembers = []; m.targetFood = null; m.cooldown = 5; });
        continue;
      }

      if (!allPaused) continue;
      pausedTeams.add(teamKey);

      const team = a.teamMembers;
      const food = a.targetFood;
      if (!food || food.energy <= 0) {
        team.forEach(m => { m.action = "IDLE"; m.teamMembers = []; m.targetFood = null; });
        continue;
      }

      const deceptionOn = condRef.current !== "A"; // §Script: always-current condRef
      // ═══ Evolvable Deception Mechanism (Steps A–H) ═══

      // Step A: Each agent forms reported value (α̂ = α_true + ε + 1[U<p]·m)
      const rep = team.map(ag => {
        const eps = deceptionOn ? gauss(0, p.commNoise) : 0; // ε ~ N(0,σ²)
        const willLie = deceptionOn && _rand() < ag.p && ag.m > 0;
        const reported = ag.alpha + eps + (willLie ? ag.m : 0);
        ag.alpha_reported = reported; // §Spec: store on agent for diagnostics
        if (willLie) ag.lieCount++;
        return { a: ag, r: reported, act: ag.alpha, l: willLie, eps };
      });

      // Step B: Choose speaker by softmax over REPORTED values (not true α)
      // β (softmaxBeta) already amplifies selection; Matthew effect arises from ξ feedback loop
      const probs = softmaxBeta(rep.map(r => r.r), p.softmaxBeta);
      const si = sampleW(probs);
      const spk = team[si];
      const lstn = team.filter((_, i) => i !== si);
      spk.timeSpeaker++; lstn.forEach(l => l.timeListener++);
      team.forEach(ag => ag.coopCount++);

      // Step C: Cooperation success — NON-PAPER switch (default OFF). The paper's main model
      // has NO cooperation-success probability: every cooperative event allocates resources.
      // `hardDetect` (with T, kappa) is optional scaffolding; with hardDetect=false (default)
      // coopSuccess is always true and only the deception-driven shares matter — matching the paper.
      const sumTrue = team.reduce((s, ag) => s + ag.alpha_raw, 0);
      const pSucc = sigmoid((sumTrue - p.T) / p.kappa);
      const coopSuccess = p.hardDetect
        ? _rand() < pSucc   // hard: success depends on sigmoid (non-paper)
        : true;                   // soft (paper default): always succeed; deception affects shares only

      // Step D: Probabilistic detection (Paper I: constant baseline vigilance v₀)
      const spkRep = rep[si];
      const Delta = Math.abs(spkRep.r - spkRep.act);
      const V0 = 0.5; // baseline vigilance (constant in Paper I)
      const Pdetect = sigmoid(p.kDetect * (Delta - p.thetaD)) * V0;
      const detected = deceptionOn ? (_rand() < Pdetect) : false;

      if (coopSuccess) {
        // === COOPERATION SUCCEEDS — food is shared ===
        team.forEach(ag => ag.coopSuccess++);
        for (const l of lstn) { const k = `${l.id}-${spk.id}`; Z.wMat.set(k, (Z.wMat.get(k) || 0) + 1); }

        // Key difference: if detected, shares based on TRUE α; if not, on REPORTED
        const shareSource = detected ? rep.map(r => ({ ...r, r: r.act })) : rep;
        const sortedDesc = [...shareSource].sort((a, b) => b.r - a.r);
        const ranks = {}; sortedDesc.forEach((r, i) => { ranks[r.a.id] = (i + 1) / team.length; });
        const Rs = ranks[spk.id], ds = 0.5 + 0.5 * Rs;
        const dtq = 1 + lstn.reduce((sum, l) => {
          const dl = 0.5 * ranks[l.id];
          return sum + (shareSource.find(r => r.a.id === spk.id).r - shareSource.find(r => r.a.id === l.id).r) * (ds - dl);
        }, 0) / lstn.length;
        const den = shareSource.reduce((s, r) => s + (1 + dtq * r.r), 0);
        // §Fix: when dtq drives den ≤ 0 or any share negative, fall back to equal split
        let shares;
        if (den > 1e-6) {
          shares = shareSource.map(r => (1 + dtq * r.r) / den);
          if (shares.some(s => s < 0)) shares = shareSource.map(() => 1 / team.length);
        } else {
          shares = shareSource.map(() => 1 / team.length);
        }
        const totalE = food.energy, fair = totalE / team.length;
        team.forEach((ag, i) => { ag.energy += totalE * shares[i]; ag.flash = 1; ag.fCol = rep[i].l ? (detected ? "#f59e0b" : "#f87171") : "#6ee7b7"; });

        // (Paper I: no credibility update — τ excluded)

        // Step F: Speaker reinforcement via Counterfactual Advantage A* and Γ signal (Eq.18–23)
        if (deceptionOn && spkRep.l) {
          const spkShareIdx = rep.findIndex(r => r.a.id === spk.id);
          const actualShare = shares[spkShareIdx];

          if (!detected && coopSuccess) {
            // Compute counterfactual: what share would speaker get if honest?
            const honRep = rep.map(r => r.a.id === spk.id ? { ...r, r: r.act } : r);
            const honSorted = [...honRep].sort((a, b) => b.r - a.r);
            const honRanks = {}; honSorted.forEach((r, i) => { honRanks[r.a.id] = (i + 1) / team.length; });
            const honRs = honRanks[spk.id], honDs = 0.5 + 0.5 * honRs;
            const honDtq = 1 + lstn.reduce((sum, l) => {
              const dl = 0.5 * honRanks[l.id];
              return sum + (honRep.find(r => r.a.id === spk.id).r - honRep.find(r => r.a.id === l.id).r) * (honDs - dl);
            }, 0) / lstn.length;
            const honDen = honRep.reduce((s, r) => s + (1 + honDtq * r.r), 0);
            const honShare = honDen > 1e-6 ? Math.max(0, (1 + honDtq * spkRep.act) / honDen) : 1 / team.length;

            // A* = E_food · (actual_share − counterfactual_share) (Eq.18)
            const Astar = totalE * (actualShare - honShare);
            // Γ = tanh(γ · A* / max(E_food, 1)) ∈ (−1, 1) (Eq.20)
            const Gamma = Math.tanh(p.gamma * Astar / Math.max(totalE, 1));

            spk.lastGamma = Gamma;
            spk.lastAstar = Astar;

            if (Gamma >= 0) {
              // Positive advantage: reinforce scaled by Γ (Eq.21)
              spk.p = clamp(spk.p + p.etaPplus * Gamma * (1 - spk.p), 0, 1);
              spk.m = spk.m + p.etaMplus * Gamma * (1 - spk.m / p.Mmax);
            } else {
              // Negative advantage: punish scaled by Γ<0 (Eq.9 lower branch: θ + η⁻·Γ·θ/θmax)
              spk.p = clamp(spk.p + p.etaPminus * Gamma * spk.p, 0, 1);            // θmax=1 for p
              spk.m = Math.max(0, spk.m + p.etaMminus * Gamma * (spk.m / p.Mmax)); // θmax=Mmax for m
            }
          } else {
            // Detected OR cooperation failed: full punishment (Γ = −1), Eq.9 lower branch
            spk.lastGamma = -1;
            spk.lastAstar = 0;
            spk.p = clamp(spk.p - p.etaPminus * spk.p, 0, 1);              // θmax=1 for p
            spk.m = Math.max(0, spk.m - p.etaMminus * (spk.m / p.Mmax));   // θmax=Mmax for m
          }
        }

        // Step G: Listener observational learning (only when NOT detected)
        if (deceptionOn && !detected) {
          const spkShareIdx = rep.findIndex(r => r.a.id === spk.id);
          const rewardShare = shares[spkShareIdx] || (1 / team.length);
          const w = Math.min(1, rewardShare * p.imitScale);
          lstn.forEach(l => {
            l.p = clamp((1 - p.lambdaPdecay) * l.p + p.etaPobs * w * spk.p, 0, 1);
            l.m = clamp((1 - p.lambdaMdecay) * l.m + p.etaMobs * w * spk.m, 0, p.Mmax); // Π_[0,Mmax] (Eq.11)
          });
        }

        food.energy = 0; Z.food = Z.food.filter(f => f.energy > 0);
        Z.anims.push({ pts: team.map(ag => ({ x: ag.x, y: ag.y })), fx: food.x, fy: food.y, sId: spk.id, lie: rep.some(r => r.l), d: dtq.toFixed(2), ttl: 16 });
        nev.push({ t: Z.tick, type: detected ? "coop_d" : "coop", spk: spk.id, ids: team.map(ag => ag.id), d: dtq.toFixed(2) });
      } else {
        // === HARD FAILURE (only in hardDetect mode) ===
        team.forEach(ag => ag.coopFail++);
        if (deceptionOn && spkRep.l) {
          spk.lastGamma = -1; spk.lastAstar = 0;
          spk.p = clamp(spk.p - p.etaPminus * spk.p, 0, 1);
          spk.m = Math.max(0, spk.m - p.etaMminus * (spk.m / p.Mmax)); // θmax=Mmax for m
        }
        nev.push({ t: Z.tick, type: "fail", ids: team.map(ag => ag.id) });
      }

      // (Paper Eq.1: α_i = C_i, fixed at birth.) The legacy Hill-function "experience"
      // mechanism (n_exp / ξ → α) is intentionally NOT updated here: capability does not
      // co-evolve after birth, so any post-switch hierarchy disruption cannot be attributed
      // to capability change. n_exp and ξ therefore stay at their birth values (0) and feed
      // diagnostic columns only — they never enter α, rank, speaker selection, or endorsement.
      // (To re-enable the dormant scaffolding, restore the n_exp/ξ update and set par.omegaXi>0.)

      // Step H: Baseline decay (paper Eq.10) — speaker only. Channel 1 modifies only the
      // acting speaker, so only the speaker's traits decay; listeners decay via ψ in Eq.11
      // (ψ=0 by default ⇒ no listener decay). NOTE: the paper phrases this as "following each
      // reinforcing step", which strictly means the speaker who actually LIED this event.
      // The line below decays the speaker every event (incl. honest speakers), a tiny deviation
      // that very slightly accelerates trait erosion. The published runs use THIS behaviour.
      //   → For the literal paper reading, change `if (deceptionOn)` to `if (deceptionOn && spkRep.l)`.
      if (deceptionOn) {
        spk.p = (1 - p.deltaP) * spk.p + p.deltaP * P_BASELINE;
        spk.m = (1 - p.deltaM) * spk.m + p.deltaM * M_BASELINE;
      }

      // Reset team
      team.forEach(m => { m.action = "IDLE"; m.teamMembers = []; m.targetFood = null; m.cooldown = 8; });
    }

    // ══ PHASE 3: Reproduction ══
    const lambda = calcLambda(nAlive);
    if (lambda > 0) {
      // Check each village for reproducing pairs
      for (const v of VILLAGES) {
        const inV = alive.filter(a => a.action === "REPRODUCING" && a.inVillage === v);
        const males = inV.filter(a => a.gender === "M");
        const females = inV.filter(a => a.gender === "F");
        if (!males.length || !females.length) continue;

        for (const self of males) {
          const partner = females.find(f => f.id !== self.id);
          if (!partner) continue;

          // Determine children count based on alpha ranking (Python logic)
          const sorted = [...alive].sort((a, b) => a.alpha - b.alpha);
          const medianRank = sorted.length >> 1;
          const selfRank = sorted.indexOf(self);
          const partnerRank = sorted.indexOf(partner);
          const baseChildren = (selfRank >= medianRank || partnerRank >= medianRank) ? 2 : 1;
          const numChildren = Math.min(baseChildren, Math.max(0, Math.round(gauss(2.2, 1) * lambda)));

          if (numChildren <= 0) continue;
          self.energy -= 2 * numChildren;
          partner.energy -= 2 * numChildren;

          for (let c = 0; c < numChildren; c++) {
            let Ck = H_COEFF * Math.max(self.capability, partner.capability) + (1 - H_COEFF) * (self.capability + partner.capability) / 2;
            if (_rand() < MUT_P) Ck += gauss(0, p.mutS);
            Ck = Math.max(1, Ck);
            // Eq.16: inherit deception traits with mutation (evolvable)
            const cP = condRef.current !== "A" ? clamp(p.childBeta * Math.max(self.p, partner.p) + gauss(0, p.sigmaMutP), 0, 1) : 0;
            const cM = condRef.current !== "A" ? clamp(p.childBeta * Math.max(self.m, partner.m) + gauss(0, p.sigmaMutM), 0, p.Mmax) : 0; // Π_[0,Mmax] (Eq.13)
            const child = mkAgent(v.x + rng(-8, 8), v.y + rng(-8, 8), _rand() < 0.5 ? "M" : "F", Ck);
            child.gen = Math.max(self.gen, partner.gen) + 1;
            child.p = cP; child.m = cM;
            recomputeAlpha(child, p, condRef.current);
            child.energy = rng(6, 7);
            Z.agents.push(child);
            nev.push({ t: Z.tick, type: "birth", a: child.id });
          }

          // Parents leave village
          self.sx = rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED);
          self.sy = rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED);
          partner.sx = rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED);
          partner.sy = rng(-IDLE_MAX_SPEED, IDLE_MAX_SPEED);
          self.action = "IDLE"; partner.action = "IDLE";
          break; // one pair per village per tick
        }
      }
    }

    // ══ PHASE 4: Movement ══
    for (const a of alive) {
      if (a.action === "DEAD" || a.action === "MOVE_COOP" || a.action === "PAUSE" || a.action === "COOP") continue;

      if (a.action === "IDLE") {
        // Python bouncing ball + food sensing
        let bf = null, bd = Infinity;
        for (const f of Z.food) { if (f.energy <= 0) continue; const d = dst(a, f); if (d < SENSE_RADIUS && d < bd) { bd = d; bf = f; } }
        if (bf && bd > 20) {
          const dx = bf.x - a.x, dy = bf.y - a.y, dd = Math.hypot(dx, dy);
          a.sx = lerp(a.sx, (dx / dd) * IDLE_MAX_SPEED * 0.8, 0.08);
          a.sy = lerp(a.sy, (dy / dd) * IDLE_MAX_SPEED * 0.8, 0.08);
        }
        a.x += a.sx; a.y += a.sy;
        if (a.x <= 5 || a.x >= WP - 5) { a.sx = -a.sx; a.ax = -a.ax; }
        if (a.y <= 5 || a.y >= HP - 5) { a.sy = -a.sy; a.ay = -a.ay; }
        a.sx = clamp(a.sx + a.ax * 0.3, -IDLE_MAX_SPEED, IDLE_MAX_SPEED);
        a.sy = clamp(a.sy + a.ay * 0.3, -IDLE_MAX_SPEED, IDLE_MAX_SPEED);
        a.x = clamp(a.x, 5, WP - 5); a.y = clamp(a.y, 5, HP - 5);
      } else if (a.action === "MOVING_VILLAGE" && a.targetVillage) {
        const dx = a.targetVillage.x - a.x, dy = a.targetVillage.y - a.y, dd = Math.hypot(dx, dy);
        if (dd > 5) { a.x += (dx / dd) * MOVE_SPEED; a.y += (dy / dd) * MOVE_SPEED; }
        a.x = clamp(a.x, 5, WP - 5); a.y = clamp(a.y, 5, HP - 5);
      } else if (a.action === "REPRODUCING") {
        // Slow drift inside village
        a.x += rng(-0.5, 0.5); a.y += rng(-0.5, 0.5);
        a.energy -= 0.015; // Python: self.energy -= 0.05 per reproduce tick
      }
    }

    // ══ PHASE 5: Selective death (every 300 ticks, Python logic) ══
    if (Z.tick % DEATH_INTERVAL === 0 && Z.tick > 0) {
      const idleAlive = alive.filter(a => a.action === "IDLE");
      if (idleAlive.length > 0) {
        // Calculate death priority score
        const sortedByCap = [...idleAlive].sort((a, b) => b.capability - a.capability);
        const scores = idleAlive.map(ag => {
          let ageFactor = 0;
          if (ag.age < 2) ageFactor = 0.5;
          else if (ag.age > 40) ageFactor = 2 + Math.floor((ag.age - 40) / 10);
          const energyFactor = ag.energy < 3 ? (3 - ag.energy) : 0;
          const capRank = (sortedByCap.indexOf(ag) + 1) / idleAlive.length;
          return { ag, score: ageFactor + energyFactor + capRank };
        });
        scores.sort((a, b) => b.score - a.score);
        const numKill = 2 + Math.floor(_rand() * 2); // 2-3
        scores.slice(0, numKill).forEach(({ ag }) => {
          ag.alive = false; ag.action = "DEAD";
          nev.push({ t: Z.tick, type: "death", a: ag.id, score: scores.find(s => s.ag === ag)?.score.toFixed(1) });
        });
      }
    }

    // Remove dead from rendering list periodically
    if (Z.tick % 100 === 0) Z.agents = Z.agents.filter(a => a.alive);

    // ══ TI ══ (decay + prune dead edges)
    if (Z.tick % 50 === 0 && alive.length > 2) {
      const aliveSet = new Set(alive.map(a => a.id));
      // 修改4: Weight decay δ=0.98 to avoid infinite accumulation
      const DECAY_DELTA = 0.98;
      const EDGE_THRESHOLD = 1e-6;
      for (const [k, val] of Z.wMat.entries()) {
        const [a, b] = k.split("-").map(Number);
        if (!aliveSet.has(a) || !aliveSet.has(b)) { Z.wMat.delete(k); continue; }
        const decayed = val * DECAY_DELTA;
        if (decayed < EDGE_THRESHOLD) Z.wMat.delete(k);
        else Z.wMat.set(k, decayed);
      }
      const tv = computeTI(Z.wMat, alive.map(a => a.id));
      const tiVal = isNaN(tv) ? 0 : tv;
      setTi(tiVal);
      // SYNC push to tiHref — guaranteed before useEffect
      const tiEntry = { t: Z.tick, v: tiVal, cond };
      tiHref.current = [...tiHref.current, tiEntry];
      setTiH(prev => [...prev, tiEntry]);

      // §CSV: Record core time-series (lightweight, every 50 ticks)
      const aliveForCore = alive;
      const sumAlphaCore = aliveForCore.reduce((s, a) => s + a.alpha, 0);
      const meanAlphaCore = sumAlphaCore / (aliveForCore.length || 1);
      const varAlphaCore = aliveForCore.reduce((s, a) => s + (a.alpha - meanAlphaCore) ** 2, 0) / (aliveForCore.length || 1);
      const meanXiCore = aliveForCore.reduce((s, a) => s + a.xi, 0) / (aliveForCore.length || 1);
      const meanEnergyCore = aliveForCore.reduce((s, a) => s + a.energy, 0) / (aliveForCore.length || 1);
      const meanGenCore = aliveForCore.reduce((s, a) => s + a.gen, 0) / (aliveForCore.length || 1);
      // ★ Gini(α) — inequality measure (Exp-1 §4.1)
      const giniAlphaCore = computeGini(aliveForCore.map(a => a.alpha));
      // ★ Network density — wMat edges / N(N-1) (Exp-1 §4.1)
      const nAliveCore = aliveForCore.length;
      const netDensityCore = nAliveCore > 1 ? Z.wMat.size / (nAliveCore * (nAliveCore - 1)) : 0;
      coreTSref.current.push({
        tick: Z.tick, TI: tiVal, N: aliveForCore.length,
        meanAlpha: meanAlphaCore, varAlpha: varAlphaCore, meanXi: meanXiCore,
        meanEnergy: meanEnergyCore, meanGen: meanGenCore,
        giniAlpha: giniAlphaCore, netDensity: netDensityCore,
        condition: condRef.current,
      });

      // ══ §Diagnostics: Compare raw vs normalized α ══
      // First, recompute all alpha_norm based on current population
      const allAlphas = alive.map(a => a.alpha_raw);
      alive.forEach(ag => {
        ag.alpha_norm = normalizeAlphaToDisplay(ag.alpha_raw, allAlphas);
      });

      // Run diagnostics every 100 ticks
      if (Z.tick % 100 === 0 && alive.length >= 10) {
        const diagResult = diagnostics_compare_alphas(alive, p.softmaxBeta);
        const calibResult = fit_scale_to_match_raw(alive, 1.0);

        // §Deception diagnostics: var(α_true), var(α̂), Spearman raw vs reported, fraction near top
        const alphaRaws = alive.map(a => a.alpha_raw);
        const meanAR = alphaRaws.reduce((s,v) => s+v, 0) / alive.length;
        const varAlphaTrue = alphaRaws.reduce((s,v) => s + (v-meanAR)**2, 0) / alive.length;

        // Use stored alpha_reported from agents (set during cooperation)
        // Fall back to simulation if not yet set
        const reported = alive.map(a => a.alpha_reported > 0 ? a.alpha_reported : a.alpha_raw);
        const meanRep = reported.reduce((s,v) => s+v, 0) / alive.length;
        const varAlphaHat = reported.reduce((s,v) => s + (v-meanRep)**2, 0) / alive.length;

        // var(p), var(m) for pooling detection
        const ps = alive.map(a => a.p);
        const ms = alive.map(a => a.m);
        const meanP = ps.reduce((s,v) => s+v, 0) / alive.length;
        const meanM = ms.reduce((s,v) => s+v, 0) / alive.length;
        const varP = ps.reduce((s,v) => s + (v-meanP)**2, 0) / alive.length;
        const varM = ms.reduce((s,v) => s + (v-meanM)**2, 0) / alive.length;

        // Spearman: true α rank vs reported α rank
        function spearmanCorr(arr1, arr2) {
          const n = arr1.length;
          const rank = arr => arr.map((v,i) => [v,i]).sort((a,b) => b[0]-a[0]).map((x,idx) => ({idx:x[1],rank:idx+1})).sort((a,b) => a.idx-b.idx).map(x => x.rank);
          const r1 = rank(arr1), r2 = rank(arr2);
          let d2 = 0; for (let i = 0; i < n; i++) d2 += (r1[i]-r2[i])**2;
          return 1 - 6*d2 / (n*(n*n-1));
        }
        const spearRawVsRep = spearmanCorr(alphaRaws, reported);

        // Fraction agents with reported > 95th percentile of true α
        const p95 = [...alphaRaws].sort((a,b)=>a-b)[Math.floor(alive.length*0.95)] || meanAR;
        const fracNearTop = reported.filter(r => r > p95).length / alive.length;

        // §Pooling detection: check var(p) and var(m) thresholds
        // Track poolingStableCount in Z (persists across ticks)
        if (!Z.poolingHistory) Z.poolingHistory = [];
        Z.poolingHistory.push({ varP, varM, t: Z.tick });
        // Keep only last poolingStableWindow ticks worth of entries (sampled every 100 ticks)
        const maxEntries = Math.ceil(p.poolingStableWindow / 100);
        if (Z.poolingHistory.length > maxEntries) Z.poolingHistory = Z.poolingHistory.slice(-maxEntries);
        const isPooling = Z.poolingHistory.length >= maxEntries &&
          Z.poolingHistory.every(e => e.varP < p.poolingVarPThresh && e.varM < p.poolingVarMThresh);

        // §Γ mechanism: track mean Gamma across agents who have lied (moved before setDiag to fix reference order)
        const curCoopRate = alive.reduce((s,a) => s + a.coopSuccess, 0) /
          Math.max(1, alive.reduce((s,a) => s + a.coopCount, 0));
        const liars = alive.filter(a => a.lieCount > 0);
        const meanGamma = liars.length > 0 ? liars.reduce((s,a) => s + a.lastGamma, 0) / liars.length : 0;
        const meanAstar = liars.length > 0 ? liars.reduce((s,a) => s + a.lastAstar, 0) / liars.length : 0;
        // Speaker/Listener cumulative counts
        const totalSpeaker = alive.reduce((s,a) => s + a.timeSpeaker, 0);
        const totalListener = alive.reduce((s,a) => s + a.timeListener, 0);

        setDiag({
          spearman: diagResult.spearman,
          KL: diagResult.KL,
          overlap: diagResult.overlap,
          bestScale: calibResult.scale,
          bestKL: calibResult.KL,
          varAlphaTrue, varAlphaHat, spearRawVsRep, fracNearTop, meanP, meanM,
          varP, varM, isPooling, meanGamma, meanAstar,
        });
        // Build the diagnostic entry FIRST, then push to ref synchronously, then queue state update
        const diagEntry = {
          t: Z.tick, N: alive.length,
          meanP, meanM, varP, varM,
          spearRawVsRep, fracNearTop,
          varAlphaTrue, varAlphaHat,
          TI: tiVal, coopRate: curCoopRate,
          meanAlpha: meanAR,
          giniAlpha: computeGini(alphaRaws),
          netDensity: alive.length > 1 ? Z.wMat.size / (alive.length * (alive.length - 1)) : 0,
          spearman: diagResult.spearman, KL: diagResult.KL,
          isPooling, condition: condRef.current,
          softmaxBeta: p.softmaxBeta, childBeta: p.childBeta,
          etaPobs: p.etaPobs, etaMobs: p.etaMobs,
          meanGamma, meanAstar,
          alphaSnap: alive.map(a => a.alpha_raw),
        };
        // SYNC push to ref — guaranteed available before any useEffect fires
        diagHref.current = [...diagHref.current, diagEntry];
        // Async state update for React re-render
        setDiagH(prev => [...prev, diagEntry]);
        // §CSV: Record deception time-series (every 100 ticks)
        decTSref.current.push({
          tick: Z.tick, meanP, meanM, varP, varM,
          spearRawVsRep, fracNearTop, coopRate: curCoopRate,
          condition: condRef.current, meanGamma, meanAstar,
          totalSpeaker, totalListener,
        });
        // JSON log for reproducibility
        console.log(`[t=${Z.tick}] §Diagnostics:`, JSON.stringify({
          tick: Z.tick,
          spearman: diagResult.spearman.toFixed(4),
          KL: diagResult.KL.toFixed(6),
          top5: diagResult.overlap.top5?.toFixed(2),
          top10: diagResult.overlap.top10?.toFixed(2),
          bestScale: calibResult.scale,
          bestKL: calibResult.KL.toFixed(6),
          varAlphaTrue: varAlphaTrue.toFixed(4),
          varAlphaHat: varAlphaHat.toFixed(4),
          spearRawVsRep: spearRawVsRep.toFixed(4),
          fracNearTop: fracNearTop.toFixed(3),
          meanP: meanP.toFixed(4),
          meanM: meanM.toFixed(4),
          varP: varP.toFixed(6),
          varM: varM.toFixed(6),
          isPooling,
          TI: tiVal.toFixed(4),
          N: alive.length,
          giniAlpha: computeGini(alphaRaws).toFixed(4),
          netDensity: (alive.length > 1 ? Z.wMat.size / (alive.length * (alive.length - 1)) : 0).toFixed(6),
          condition: condRef.current,
          softmaxBeta: p.softmaxBeta,
          meanGamma: meanGamma.toFixed(4),
          meanAstar: meanAstar.toFixed(3),
        }));
      }
    }

    Z.events = [...Z.events, ...nev].slice(-120);
    setTick(Z.tick); setEvts([...Z.events]); upd();
    setHist(prev => {
      const al = Z.agents.filter(a => a.alive);
      const aN = al.length || 1;
      let sumAlpha = 0, sumXi = 0, sumE = 0, sumP = 0, sumM = 0, coopS = 0, coopT = 0;
      al.forEach(a => { sumAlpha += a.alpha; sumXi += a.xi; sumE += a.energy; sumP += a.p; sumM += a.m; coopS += a.coopSuccess; coopT += a.coopCount; });
      // Variance of alpha (true)
      const meanA = sumAlpha / aN;
      const varA = al.reduce((s, a) => s + (a.alpha - meanA) ** 2, 0) / aN;
      // ★ Gini & density for mini-charts
      const giniA = computeGini(al.map(a => a.alpha));
      const nDens = aN > 1 ? Z.wMat.size / (aN * (aN - 1)) : 0;
      return [...prev, {
        t: Z.tick,
        n: al.filter(a => a.p < 0.1).length,
        lo: al.filter(a => a.p >= 0.1 && a.p < 0.35).length,
        mi: al.filter(a => a.p >= 0.35 && a.p < 0.65).length,
        hi: al.filter(a => a.p >= 0.65).length,
        total: al.length,
        avgAlpha: sumAlpha / aN,
        avgXi: sumXi / aN,
        avgE: sumE / aN,
        avgP: sumP / aN,
        avgM: sumM / aN,
        varAlpha: varA,
        giniAlpha: giniA,
        netDensity: nDens,
        coopRate: coopT > 0 ? coopS / coopT : 0,
      }].slice(-350);
    });

    // ══════════════════════════════════════════════════
    // §Script execution — fires events at specified ticks
    // ══════════════════════════════════════════════════
    if (scriptEnabled) {
      for (const ev of scriptR.current) {
        if (firedR.current.has(ev.id)) continue;
        if (Z.tick >= ev.t) {
          firedR.current.add(ev.id);
          if (ev.action === "setCond") {
            const prevCond = condRef.current;
            setCondRef.current(ev.value);
            if (prevCond === "A" && ev.value === "B") applyIgnition();
            setScriptLog(prev => [...prev, { t: Z.tick, msg: `✓ t=${Z.tick}: switched to Cond ${ev.value}${prevCond === "A" && ev.value === "B" ? " (ignition applied)" : ""}` }]);
          } else if (ev.action === "setParam") {
            // §Ablation: parse "paramKey=value" from ev.value
            const [paramKey, paramValStr] = (ev.value || "").split("=");
            const paramVal = parseFloat(paramValStr);
            if (paramKey && !isNaN(paramVal)) {
              setPar(prev => ({ ...prev, [paramKey]: paramVal }));
              setScriptLog(prev => [...prev, { t: Z.tick, msg: `✓ t=${Z.tick}: set ${paramKey} → ${paramVal}` }]);
            }
          } else if (ev.action === "autoExport") {
            autoExportR.current = true; // picked up by useEffect below
            setScriptLog(prev => [...prev, { t: Z.tick, msg: `✓ t=${Z.tick}: auto-export triggered` }]);
          } else if (ev.action === "pause") {
            setRun(false);
            setScriptLog(prev => [...prev, { t: Z.tick, msg: `✓ t=${Z.tick}: simulation paused` }]);
          }
        }
      }
    }
  }, [upd, scriptEnabled]);

  useEffect(() => { if (!run) return; const iv = setInterval(() => { for (let i = 0; i < spd; i++) step(); }, 33); return () => clearInterval(iv); }, [run, spd, step]);

  // ══════════════════════════════════════════════════
  // §Auto-export: fires when autoExportR.current = true
  // Draws to hidden canvases, opens in new tabs, exports CSVs
  // ══════════════════════════════════════════════════
  useEffect(() => {
    if (!autoExportR.current) return;
    autoExportR.current = false;
    const Z = simR.current; if (!Z) return;

    // Safety: if refs are somehow empty, defer to next frame
    const dH = diagHref.current;
    const tH = tiHref.current;
    if (!dH.length && !tH.length) {
      // Data not ready yet — retry next frame
      autoExportR.current = true;
      return;
    }

    // Build the same pubChart + network drawing functions (duplicated here for direct access)
    function drawHidden(canvas, { title, xLabel, yLabel, series, xKey = "t", yRange }) {
      if (!canvas) return;
      const W = 900, H = 500; canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext("2d");
      const P = { l: 72, r: 24, t: 46, b: 58 };
      const CW = W - P.l - P.r, CH = H - P.t - P.b;
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#f9fafb"; ctx.fillRect(P.l, P.t, CW, CH);
      let yMin = Infinity, yMax = -Infinity;
      series.forEach(s => s.data.forEach(d => { const v = d[s.yKey ?? "v"]; if (v != null && isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); } }));
      if (yRange) { [yMin, yMax] = yRange; } else { const pad = (yMax - yMin) * 0.08 || 0.05; yMin -= pad; yMax += pad; }
      if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
      let xMin = Infinity, xMax = -Infinity;
      series.forEach(s => s.data.forEach(d => { const v = d[xKey]; if (v != null) { xMin = Math.min(xMin, v); xMax = Math.max(xMax, v); } }));
      if (!isFinite(xMin)) { xMin = 0; xMax = 1; } if (xMax === xMin) xMax = xMin + 1;
      const tx = x => P.l + ((x - xMin) / (xMax - xMin)) * CW;
      const ty = y => P.t + CH - ((clamp(y, yMin, yMax) - yMin) / (yMax - yMin)) * CH;
      ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
      ctx.font = "12px Arial"; ctx.fillStyle = "#6b7280";
      for (let i = 0; i <= 5; i++) {
        const yv = yMin + (yMax - yMin) * (i / 5); ctx.textAlign = "right";
        ctx.beginPath(); ctx.moveTo(P.l, ty(yv)); ctx.lineTo(P.l + CW, ty(yv)); ctx.stroke();
        ctx.fillText(yv.toFixed(Math.abs(yMax) < 5 ? 3 : 0), P.l - 5, ty(yv) + 4);
      }
      for (let i = 0; i <= 7; i++) {
        const xv = xMin + (xMax - xMin) * (i / 7); ctx.textAlign = "center";
        ctx.beginPath(); ctx.moveTo(tx(xv), P.t); ctx.lineTo(tx(xv), P.t + CH); ctx.stroke();
        ctx.fillText(Math.round(xv), tx(xv), P.t + CH + 18);
      }
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1.5; ctx.strokeRect(P.l, P.t, CW, CH);
      series.forEach(s => {
        if (!s.data.length) return;
        ctx.strokeStyle = s.color; ctx.lineWidth = s.lw || 2; ctx.setLineDash(s.dash || []);
        ctx.beginPath(); let first = true;
        s.data.forEach(d => { const xv = d[xKey], yv = d[s.yKey ?? "v"]; if (xv == null || !isFinite(yv ?? 0)) return; first ? ctx.moveTo(tx(xv), ty(yv)) : ctx.lineTo(tx(xv), ty(yv)); first = false; });
        ctx.stroke(); ctx.setLineDash([]);
      });
      let lx = P.l + 8;
      series.forEach(s => { ctx.fillStyle = s.color; ctx.fillRect(lx, P.t + 12, 20, 5); ctx.fillStyle = "#374151"; ctx.font = "11px Arial"; ctx.textAlign = "left"; ctx.fillText(s.label, lx + 24, P.t + 18); lx += ctx.measureText(s.label).width + 48; });
      ctx.fillStyle = "#111827"; ctx.font = "bold 14px Arial"; ctx.textAlign = "center"; ctx.fillText(title, W / 2, 26);
      ctx.fillStyle = "#6b7280"; ctx.font = "12px Arial"; ctx.fillText(xLabel, P.l + CW / 2, H - 10);
      ctx.save(); ctx.translate(14, P.t + CH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
    }

    // Draw all 6 figures to hidden canvases (dH and tH already read from refs above)
    // Fig 1: TI
    drawHidden(hc1.current, { title: `Fig 1 — TI over Time  [seed:${seed} t=0→${tick}]`, xLabel: "Simulation Tick", yLabel: "TI (disorder)", series: [{ data: tH, yKey: "v", color: "#7c3aed", label: "TI", lw: 2 }] });
    // Fig 2: Deception
    drawHidden(hc2.current, { title: `Fig 2 — Deception Evolution  [seed:${seed}]`, xLabel: "Simulation Tick", yLabel: "Value", series: [{ data: dH, yKey: "meanP", color: "#ef4444", label: "p̄", lw: 2 }, { data: dH, yKey: "meanM", color: "#f97316", label: "m̄", lw: 2 }, { data: dH, yKey: "varP", color: "#fca5a5", label: "Var(p)", lw: 1.2 }] });
    // Fig 3: Network
    if (hc3.current) {
      const canvas = hc3.current; canvas.width = 900; canvas.height = 600;
      const ctx = canvas.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 900, 600);
      const alive = Z.agents.filter(a => a.alive);
      const PAD = 50, sx = x => PAD + x * ((900-2*PAD)/WP), sy = y => PAD + y * ((520-PAD)/HP);
      const inDeg = new Map(); alive.forEach(a => inDeg.set(a.id, 0));
      const edges = [];
      for (const [k, w] of Z.wMat) { const [f,t2] = k.split("-").map(Number); const fa = alive.find(a=>a.id===f), ta = alive.find(a=>a.id===t2); if (!fa||!ta) continue; inDeg.set(t2,(inDeg.get(t2)||0)+w); edges.push({from:fa,to:ta,w}); }
      const maxDeg = Math.max(1,...inDeg.values()), maxW = Math.max(1,...edges.map(e=>e.w));
      edges.forEach(({from,to,w})=>{ const a=Math.min(0.7,0.08+0.62*w/maxW); ctx.strokeStyle=`rgba(99,102,241,${a.toFixed(2)})`; ctx.lineWidth=0.4+2.2*w/maxW; ctx.beginPath(); ctx.moveTo(sx(from.x),sy(from.y)); ctx.lineTo(sx(to.x),sy(to.y)); ctx.stroke(); const ang=Math.atan2(sy(to.y)-sy(from.y),sx(to.x)-sx(from.x)); const ax=sx(to.x)-8*Math.cos(ang),ay=sy(to.y)-8*Math.sin(ang); ctx.fillStyle=`rgba(99,102,241,${a.toFixed(2)})`; ctx.beginPath(); ctx.moveTo(sx(to.x),sy(to.y)); ctx.lineTo(ax-3.5*Math.sin(ang),ay+3.5*Math.cos(ang)); ctx.lineTo(ax+3.5*Math.sin(ang),ay-3.5*Math.cos(ang)); ctx.fill(); });
      alive.forEach(a=>{ const deg=inDeg.get(a.id)||0; const r=2.5+14*(deg/maxDeg); const col=a.p<0.1?"#10b981":a.p<0.35?"#84cc16":a.p<0.65?"#f59e0b":"#ef4444"; ctx.beginPath(); ctx.arc(sx(a.x),sy(a.y),r,0,6.28); ctx.fillStyle=col; ctx.fill(); ctx.strokeStyle="#374151"; ctx.lineWidth=0.4; ctx.stroke(); });
      ctx.fillStyle="#111827"; ctx.font="bold 14px Arial"; ctx.textAlign="center"; ctx.fillText(`Fig 3 — Endorsement Network  t=${tick}  seed:${seed}  N=${alive.length}  edges=${edges.length}`,450,28);
    }
    // Fig 4: Fidelity
    drawHidden(hc4.current, { title: `Fig 4 — Signal Fidelity  [seed:${seed}]`, xLabel: "Simulation Tick", yLabel: "Value", series: [{ data: dH, yKey: "spearRawVsRep", color: "#0ea5e9", label: "ρ(α,α̂)", lw: 2.5 }, { data: dH, yKey: "fracNearTop", color: "#e11d48", label: "fracNearTop", lw: 1.8 }], yRange: [-0.05, 1.1] });
    // Fig 5: α dist (skip if no snapshots)
    if (hc5.current && dH.length >= 3) {
      const canvas = hc5.current; canvas.width = 900; canvas.height = 500;
      const ctx = canvas.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 900, 500);
      const P = { l: 60, r: 20, t: 50, b: 58 }, CW = 820, CH = 500-P.t-P.b;
      ctx.fillStyle = "#f9fafb"; ctx.fillRect(P.l, P.t, CW, CH);
      const idxs = [0, Math.floor(dH.length/2), dH.length-1];
      let gmax = 0;
      const NBINS = 30;
      const allBins = idxs.map(i => { const d = dH[i]; if (!d.alphaSnap?.length) return { bins: new Array(NBINS).fill(0), mn: 0, mx: 1 }; const mn=Math.min(...d.alphaSnap),mx=Math.max(...d.alphaSnap),rng=mx-mn||1e-6; const bins=new Array(NBINS).fill(0); d.alphaSnap.forEach(v=>bins[Math.min(NBINS-1,Math.floor(((v-mn)/rng)*NBINS))]++); const tot=d.alphaSnap.length; const dens=bins.map(b=>b/tot); gmax=Math.max(gmax,...dens); return{bins:dens,mn,mx,N:d.N,t:d.t}; });
      if (!gmax) gmax = 0.5;
      const bw = CW/NBINS/3;
      idxs.forEach((_,si)=>{ const {bins}=allBins[si]; const col=["#3b82f6","#f59e0b","#ef4444"][si]+"aa"; ctx.fillStyle=col; bins.forEach((v,i)=>{ const h=(v/gmax)*CH; ctx.fillRect(P.l+(i/NBINS)*CW+si*bw,P.t+CH-h,bw-0.5,h); }); });
      ctx.strokeStyle="#9ca3af"; ctx.lineWidth=1.5; ctx.strokeRect(P.l,P.t,CW,CH);
      ctx.fillStyle="#111827"; ctx.font="bold 14px Arial"; ctx.textAlign="center"; ctx.fillText(`Fig 5 — α Distribution Snapshots  [seed:${seed}]`,450,30);
      ctx.fillStyle="#6b7280"; ctx.font="12px Arial"; ctx.fillText("α (normalized 0–1 within snapshot)",P.l+CW/2,490);
      idxs.forEach((_,si)=>{ const{mn,mx,N,t:st}=allBins[si]; const col=["#3b82f6","#f59e0b","#ef4444"][si]; ctx.fillStyle=col; ctx.fillRect(P.l+8+si*250,P.t+10,16,12); ctx.fillStyle="#374151"; ctx.font="11px Arial"; ctx.textAlign="left"; ctx.fillText(`t=${st} [${mn.toFixed(1)}–${mx.toFixed(1)}] N=${N}`,P.l+28+si*250,P.t+20); });
    }
    // Fig 6: Coop×p
    if (hc6.current) {
      const canvas = hc6.current; canvas.width = 900; canvas.height = 500;
      const ctx = canvas.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 900, 500);
      const P={l:72,r:24,t:46,b:58},CW=804,CH=500-P.t-P.b;
      ctx.fillStyle="#f9fafb"; ctx.fillRect(P.l,P.t,CW,CH);
      const pts=dH.filter(d=>isFinite(d.coopRate)&&isFinite(d.meanP));
      if (pts.length) {
        const xMax=Math.max(0.01,...pts.map(d=>d.meanP))*1.15;
        const tx2=x=>P.l+(x/xMax)*CW, ty2=y=>P.t+CH-clamp(y,0,1)*CH;
        ctx.strokeStyle="#e5e7eb"; ctx.lineWidth=1; ctx.font="12px Arial";
        [0,.25,.5,.75,1].forEach(v=>{ ctx.beginPath(); ctx.moveTo(P.l,ty2(v)); ctx.lineTo(P.l+CW,ty2(v)); ctx.stroke(); ctx.fillStyle="#6b7280"; ctx.textAlign="right"; ctx.fillText(v.toFixed(2),P.l-5,ty2(v)+4); });
        ctx.strokeStyle="#9ca3af"; ctx.lineWidth=1.5; ctx.strokeRect(P.l,P.t,CW,CH);
        pts.forEach((d,i)=>{ const [r,g,b]=d.condition==="A"?[16,185,129]:d.condition==="B"?[239,68,68]:[139,92,246]; ctx.fillStyle=`rgba(${r},${g},${b},${0.3+0.5*i/pts.length})`; ctx.beginPath(); ctx.arc(tx2(d.meanP),ty2(d.coopRate),4,0,6.28); ctx.fill(); });
      }
      ctx.fillStyle="#111827"; ctx.font="bold 14px Arial"; ctx.textAlign="center"; ctx.fillText(`Fig 6 — Coop Rate vs p̄  [seed:${seed}]`,450,30);
      ctx.fillStyle="#6b7280"; ctx.font="12px Arial"; ctx.fillText("p̄ (mean lying probability)",P.l+CW/2,490);
      ctx.save(); ctx.translate(14,P.t+CH/2); ctx.rotate(-Math.PI/2); ctx.fillText("Cooperation Rate",0,0); ctx.restore();
    }

    // Open all 6 in new tabs
    const canvases = [hc1,hc2,hc3,hc4,hc5,hc6];
    const names = ["fig1_TI","fig2_deception","fig3_network","fig4_fidelity","fig5_alphadist","fig6_coopVsP"];
    const DELAY = 400; // stagger to avoid popup blocker
    canvases.forEach((ref, i) => {
      if (!ref.current) return;
      setTimeout(() => {
        const dataURL = ref.current.toDataURL("image/png");
        const w = window.open("", "_blank");
        if (w) {
          w.document.write(`<!DOCTYPE html><html><head><title>${names[i]}_seed${seed}_t${tick}.png</title></head><body style="margin:0;background:#111"><img src="${dataURL}" style="max-width:100%;display:block"/><p style="color:#aaa;font-family:monospace;padding:8px;font-size:11px">Right-click → Save image as: ${names[i]}_seed${seed}_t${tick}.png</p></body></html>`);
          w.document.close();
        }
      }, i * DELAY);
    });

    // Export CSVs via direct download (new lightweight CSVs + existing full diagnostics)
    setTimeout(() => {
      // Core TS CSV
      if (coreTSref.current.length) downloadCSV(coreTSref.current, `ts_core_seed${seed}`);
      // Deception TS CSV
      if (decTSref.current.length) downloadCSV(decTSref.current, `ts_deception_seed${seed}`);
      // Meta CSV
      const pp = pRef.current;
      downloadCSV([{
        seed, totalTicks: tick, condition: cond,
        omegaC: pp.omegaC, omegaXi: pp.omegaXi, softmaxBeta: pp.softmaxBeta,
        etaPplus: pp.etaPplus, etaMplus: pp.etaMplus, etaPminus: pp.etaPminus, etaMminus: pp.etaMminus,
        gamma: pp.gamma, etaPobs: pp.etaPobs, etaMobs: pp.etaMobs,
        childBeta: pp.childBeta, thetaD: pp.thetaD, kDetect: pp.kDetect,
        expListenerWt: pp.expListenerWt, hillBeta: pp.hillBeta, hillKappa: pp.hillKappa,
        ignitionP: pp.ignitionP, ignitionM: pp.ignitionM,
      }], `meta_seed${seed}`);
    }, canvases.length * DELAY + 100);

    setTimeout(() => {
      const dH2 = diagHref.current;
      if (!dH2.length) return;
      const keys = ["t","N","TI","meanP","meanM","varP","varM","spearRawVsRep","fracNearTop","varAlphaTrue","varAlphaHat","coopRate","meanAlpha","giniAlpha","netDensity","spearman","KL","isPooling","condition","softmaxBeta","childBeta","etaPobs","etaMobs","meanGamma","meanAstar"];
      downloadCSV(dH2.map(d => { const row = {}; keys.forEach(k => row[k] = d[k] ?? ""); return row; }), `diagnostics_seed${seed}`);
    }, canvases.length * DELAY + 200);

    setTimeout(() => {
      const alive = Z.agents.filter(a => a.alive);
      const keys2 = ["id","x","y","gender","capability","alpha","alpha_raw","p","m","xi","energy","age","gen","coopCount","coopSuccess","lieCount","lastGamma","lastAstar"];
      downloadCSV(alive.map(a => { const row = {}; keys2.forEach(k => row[k] = a[k] ?? ""); return row; }), `snap_agents_t${tick}_seed${seed}`);
    }, canvases.length * DELAY + 400);

    setScriptLog(prev => [...prev, { t: tick, msg: `📦 t=${tick}: exported 6 PNGs + 5 CSVs (core, deception, agents, meta, diagnostics)` }]);
  }, [tick]); // fires every tick, checks autoExportR flag

  // ── Render arena ──
  useEffect(() => {
    const c = cvs.current; if (!c || !simR.current) return;
    const ctx = c.getContext("2d"); c.width = WP; c.height = HP;
    const Z = simR.current;
    const CT = ctRef.current;
    const isLight = themeRef.current === "light";

    // Background + faint grid for depth
    ctx.fillStyle = CT.arena; ctx.fillRect(0, 0, WP, HP);
    ctx.strokeStyle = CT.grid; ctx.lineWidth = 1;
    for (let gx = 0; gx <= WP; gx += 50) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, HP); ctx.stroke(); }
    for (let gy = 0; gy <= HP; gy += 50) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(WP, gy); ctx.stroke(); }

    // Villages: soft "settlement" zones with dashed ring + label
    for (const v of VILLAGES) {
      const g = ctx.createRadialGradient(v.x, v.y, 0, v.x, v.y, v.r);
      g.addColorStop(0, CT.villageFill); g.addColorStop(1, "transparent");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(v.x, v.y, v.r, 0, 6.28); ctx.fill();
      ctx.strokeStyle = CT.village; ctx.lineWidth = 1.3; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.arc(v.x, v.y, v.r, 0, 6.28); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = CT.text; ctx.font = "9px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText("village", v.x, v.y - v.r - 5);
    }

    // Cooperation event animations (endorsement triangles)
    for (const ca of Z.anims) {
      const al = ca.ttl / 16;
      ctx.globalAlpha = al * 0.5;
      ctx.strokeStyle = ca.lie ? "#f87171" : "#6ee7b7"; ctx.lineWidth = 1.3;
      ctx.beginPath(); ca.pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)); ctx.closePath(); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Food: glowing gem (diamond) — shape distinguishes it from circular agents
    for (const f of Z.food) {
      if (f.energy <= 0) continue;
      ctx.save();
      ctx.translate(f.x, f.y); ctx.rotate(Math.PI / 4);
      ctx.shadowColor = CT.foodGlow; ctx.shadowBlur = 8;
      ctx.fillStyle = CT.food; ctx.fillRect(-4, -4, 8, 8);
      ctx.restore();
    }
    ctx.shadowBlur = 0;

    // Agents: circle, fill = deception level; size hints energy; speaker flash retained
    Z.agents.forEach(a => {
      if (!a.alive) return;
      const tc = a.p < 0.1 ? "#6ee7b7" : a.p < 0.35 ? "#a3e635" : a.p < 0.65 ? "#fbbf24" : "#ef4444";
      const r = 3.5 + Math.min(2.5, Math.max(0, a.energy) * 0.18); // subtle energy → radius
      if (a.flash > 0) {
        ctx.beginPath(); ctx.arc(a.x, a.y, r + 3 + a.flash * 4, 0, 6.28);
        ctx.fillStyle = `${a.fCol || "#ffffff"}${Math.floor(a.flash * 22).toString(16).padStart(2, "0")}`;
        ctx.fill();
      }
      ctx.fillStyle = tc; ctx.shadowColor = tc; ctx.shadowBlur = isLight ? 2 : 5;
      ctx.beginPath(); ctx.arc(a.x, a.y, r, 0, 6.28); ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = CT.agentStroke; ctx.lineWidth = 0.7; ctx.stroke();
      if (sel === a.id) {
        ctx.beginPath(); ctx.arc(a.x, a.y, r + 5, 0, 6.28);
        ctx.strokeStyle = isLight ? "#1d2719" : "#ffffff"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = CT.text; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
        ctx.fillText(`E ${Math.floor(a.energy)} · p ${a.p.toFixed(2)}`, a.x, a.y - r - 6);
      }
    });
  }, [tick, sel, theme]);

  // ── Generic mini-chart drawer ──
  const drawMiniChart = useCallback((canvasRef, data, lines, height = 55) => {
    const c = canvasRef.current; if (!c || data.length < 2) return;
    const ctx = c.getContext("2d");
    const cw = c.width = c.parentElement?.clientWidth || 260, ch = c.height = height;
    ctx.fillStyle = ctRef.current.chartBg; ctx.fillRect(0, 0, cw, ch);
    let mx = 0.001;
    for (const { key } of lines) mx = Math.max(mx, ...data.map(d => Math.abs(d[key] ?? 0)));
    for (const { key, col } of lines) {
      ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 1.1;
      data.forEach((d, i) => {
        const x = (i / (data.length - 1)) * cw;
        const y = ch - ((d[key] ?? 0) / mx) * (ch - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }, []);

  // Charts
  useEffect(() => {
    drawMiniChart(chart, hist, [
      { key: "n", col: "#6ee7b7" }, { key: "lo", col: "#a3e635" },
      { key: "mi", col: "#fbbf24" }, { key: "hi", col: "#ef4444" },
    ]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(tiC, tiH.map(h => ({ v: h.v })), [{ key: "v", col: "#c084fc" }]);
  }, [tiH, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(popC, hist, [{ key: "total", col: "#7dd3fc" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(alphaC, hist, [{ key: "avgAlpha", col: "#fbbf24" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(energyC, hist, [{ key: "avgE", col: "#f87171" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(coopC, hist, [{ key: "coopRate", col: "#6ee7b7" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(xiC, hist, [
      { key: "avgXi", col: "#86efac" },
    ]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(mChartC, hist, [{ key: "avgM", col: "#fb923c" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(varAlphaC, hist, [{ key: "varAlpha", col: "#38bdf8" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(giniAlphaC, hist, [{ key: "giniAlpha", col: "#e879f9" }]);
  }, [hist, drawMiniChart, theme]);
  useEffect(() => {
    drawMiniChart(netDensityC, hist, [{ key: "netDensity", col: "#2dd4bf" }]);
  }, [hist, drawMiniChart, theme]);
  // Spearman raw vs reported trace (from diag state, stored as single-value history)
  useEffect(() => {
    // Use a placeholder mini-chart approach: just render a static value indicator
    const c = spearRepC.current; if (!c) return;
    const ctx = c.getContext("2d");
    const cw = c.width = c.parentElement?.clientWidth || 260, ch = c.height = 20;
    ctx.fillStyle = ctRef.current.chartBg; ctx.fillRect(0, 0, cw, ch);
    const val = diag.spearRawVsRep ?? 1;
    const barW = Math.max(0, Math.min(1, (val + 1) / 2)) * cw; // map [-1,1] → [0,cw]
    ctx.fillStyle = val > 0.7 ? "rgba(110,231,183,0.3)" : val > 0.3 ? "rgba(251,191,36,0.3)" : "rgba(239,68,68,0.3)";
    ctx.fillRect(0, 0, barW, ch);
    ctx.fillStyle = "#8a9a84"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText(`ρ(α,α̂)=${val.toFixed(3)}`, cw / 2, 14);
  }, [diag, theme]);

  // ══ Real-time α Distribution Histogram ══
  useEffect(() => {
    const c = alphaDistC.current; if (!c || !simR.current) return;
    const ctx = c.getContext("2d");
    const cw = c.width = c.parentElement?.clientWidth || 260, ch = c.height = 80;
    ctx.fillStyle = ctRef.current.chartBg; ctx.fillRect(0, 0, cw, ch);

    const alive = simR.current.agents.filter(a => a.alive);
    if (alive.length < 2) return;

    // Get all raw alphas, find min/max for proportional normalization to [0,1]
    const alphas = alive.map(a => a.alpha);
    const aMin = Math.min(...alphas);
    const aMax = Math.max(...alphas);
    const range = aMax - aMin || 1e-6;

    // Bin into 20 buckets across [0,1]
    const NBINS = 20;
    const bins = new Array(NBINS).fill(0);
    alive.forEach(a => {
      const norm = (a.alpha - aMin) / range; // [0,1]
      const bi = Math.min(NBINS - 1, Math.floor(norm * NBINS));
      bins[bi]++;
    });
    const maxBin = Math.max(1, ...bins);

    // Draw bars
    const barW = cw / NBINS;
    const padY = 14; // top padding for axis label
    const drawH = ch - padY - 2;
    for (let i = 0; i < NBINS; i++) {
      const h = (bins[i] / maxBin) * drawH;
      // Color gradient: low alpha = blue, high alpha = green-yellow
      const t = i / (NBINS - 1);
      const r = Math.floor(30 + 200 * t);
      const g = Math.floor(180 + 60 * t);
      const b = Math.floor(220 - 180 * t);
      ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
      ctx.fillRect(i * barW + 0.5, padY + drawH - h, barW - 1, h);
    }

    // X-axis labels
    ctx.fillStyle = "#5a6e54"; ctx.font = "7px monospace"; ctx.textAlign = "center";
    ctx.fillText("0", 8, ch - 1);
    ctx.fillText("0.5", cw / 2, ch - 1);
    ctx.fillText("1", cw - 8, ch - 1);
    // Title with range info
    ctx.fillStyle = "#8a9a84"; ctx.font = "8px monospace"; ctx.textAlign = "left";
    ctx.fillText(`α dist [${aMin.toFixed(1)}–${aMax.toFixed(1)}] N=${alive.length}`, 3, 10);
    // Peak bin label
    const peakIdx = bins.indexOf(maxBin);
    const peakNorm = ((peakIdx + 0.5) / NBINS).toFixed(2);
    ctx.textAlign = "right"; ctx.fillStyle = "#fbbf24";
    ctx.fillText(`peak@${peakNorm} n=${maxBin}`, cw - 3, 10);
  }, [tick, theme]);

  // ══ Real-time α̂ Reported Distribution Histogram ══
  useEffect(() => {
    const c = alphaRepDistC.current; if (!c || !simR.current) return;
    const ctx = c.getContext("2d");
    const cw = c.width = c.parentElement?.clientWidth || 260, ch = c.height = 80;
    ctx.fillStyle = ctRef.current.chartBg; ctx.fillRect(0, 0, cw, ch);

    const alive = simR.current.agents.filter(a => a.alive);
    if (alive.length < 2) return;

    // Get all reported alphas (fall back to alpha_raw if not yet set)
    const alphas = alive.map(a => a.alpha_reported > 0 ? a.alpha_reported : a.alpha);
    const aMin = Math.min(...alphas);
    const aMax = Math.max(...alphas);
    const range = aMax - aMin || 1e-6;

    // Bin into 20 buckets across [0,1]
    const NBINS = 20;
    const bins = new Array(NBINS).fill(0);
    alive.forEach((a, idx) => {
      const norm = (alphas[idx] - aMin) / range; // [0,1]
      const bi = Math.min(NBINS - 1, Math.floor(norm * NBINS));
      bins[bi]++;
    });
    const maxBin = Math.max(1, ...bins);

    // Draw bars
    const barW = cw / NBINS;
    const padY = 14; // top padding for axis label
    const drawH = ch - padY - 2;
    for (let i = 0; i < NBINS; i++) {
      const h = (bins[i] / maxBin) * drawH;
      // Color gradient: low = cyan, high = orange-red (distinct from α dist)
      const t = i / (NBINS - 1);
      const r = Math.floor(30 + 220 * t);
      const g = Math.floor(200 - 80 * t);
      const b = Math.floor(240 - 200 * t);
      ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
      ctx.fillRect(i * barW + 0.5, padY + drawH - h, barW - 1, h);
    }

    // X-axis labels
    ctx.fillStyle = "#5a6e54"; ctx.font = "7px monospace"; ctx.textAlign = "center";
    ctx.fillText("0", 8, ch - 1);
    ctx.fillText("0.5", cw / 2, ch - 1);
    ctx.fillText("1", cw - 8, ch - 1);
    // Title with range info
    ctx.fillStyle = "#fb923c"; ctx.font = "8px monospace"; ctx.textAlign = "left";
    ctx.fillText(`α̂ dist [${aMin.toFixed(1)}–${aMax.toFixed(1)}] N=${alive.length}`, 3, 10);
    // Peak bin label
    const peakIdx2 = bins.indexOf(maxBin);
    const peakNorm2 = ((peakIdx2 + 0.5) / NBINS).toFixed(2);
    ctx.textAlign = "right"; ctx.fillStyle = "#fb923c";
    ctx.fillText(`peak@${peakNorm2} n=${maxBin}`, cw - 3, 10);
  }, [tick, theme]);

  const handleClick = (e) => {
    const Z = simR.current; if (!Z) return;
    const rect = cvs.current.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (WP / rect.width), my = (e.clientY - rect.top) * (HP / rect.height);
    let cl = null, cd = Infinity;
    Z.agents.filter(a => a.alive).forEach(a => { const d = Math.hypot(a.x - mx, a.y - my); if (d < cd && d < 14) { cd = d; cl = a; } });
    setSel(cl ? cl.id : null);
  };

  // ══════════════════════════════════════════════════════════
  // §Export helpers
  // savePNG: opens image in new tab (works in all iframe envs)
  // downloadCSV: direct Blob download for CSV data
  // ══════════════════════════════════════════════════════════
  const downloadCSV = useCallback((data, filename) => {
    if (!data || !data.length) return;
    const keys = Object.keys(data[0]);
    const csv = [keys.join(","), ...data.map(row => keys.map(k => row[k] ?? "").join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${filename}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }, []);

  const savePNG = useCallback((canvas, name) => {
    if (!canvas) return;
    const dataURL = canvas.toDataURL("image/png");
    const w = window.open("", "_blank");
    if (w) {
      w.document.write(`<!DOCTYPE html><html><head><title>${name}_seed${seed}_t${tick}.png</title></head><body style="margin:0;background:#111"><img src="${dataURL}" style="max-width:100%;display:block"/><p style="color:#aaa;font-family:monospace;padding:8px;font-size:11px">Right-click → Save image as: ${name}_seed${seed}_t${tick}.png</p></body></html>`);
      w.document.close();
    }
  }, [seed, tick]);

  const exportDiagCSV = useCallback(() => {
    if (!diagH.length) { alert("No diagnostic data yet — run the simulation for at least 100 ticks first."); return; }
    const keys = ["t","N","TI","meanP","meanM","varP","varM","spearRawVsRep","fracNearTop","varAlphaTrue","varAlphaHat","coopRate","meanAlpha","spearman","KL","isPooling","condition","softmaxBeta","childBeta","etaPobs","etaMobs","meanGamma","meanAstar"];
    const rows = diagH.map(d => keys.map(k => d[k] ?? "").join(","));
    const csv = [keys.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const w = window.open("", "_blank");
    if (w) {
      w.document.write(`<!DOCTYPE html><html><head><title>diagnostics_seed${seed}.csv</title></head><body style="background:#111;color:#aaa;font-family:monospace"><p>Copy all text below (Ctrl+A then Ctrl+C) and save as diagnostics_seed${seed}.csv</p><pre style="font-size:10px;overflow:auto">${csv.replace(/</g,"&lt;")}</pre></body></html>`);
      w.document.close();
    }
  }, [diagH, seed]);

  const exportAgentCSV = useCallback(() => {
    const Z = simR.current; if (!Z) return;
    const alive = Z.agents.filter(a => a.alive);
    const keys = ["id","x","y","gender","capability","alpha","alpha_raw","p","m","xi","n_exp","energy","age","gen","coopCount","coopSuccess","lieCount","lastGamma","lastAstar"];
    const rows = alive.map(a => keys.map(k => (a[k] ?? "")).join(","));
    const csv = [keys.join(","), ...rows].join("\n");
    const w = window.open("", "_blank");
    if (w) {
      w.document.write(`<!DOCTYPE html><html><head><title>agents_t${tick}_seed${seed}.csv</title></head><body style="background:#111;color:#aaa;font-family:monospace"><p>agents_t${tick}_seed${seed}.csv — ${alive.length} agents</p><pre style="font-size:10px;overflow:auto">${csv}</pre></body></html>`);
      w.document.close();
    }
  }, [tick, seed]);

  // §New CSV exports: lightweight time-series for multi-seed analysis
  const exportCoreTScsv = useCallback(() => {
    if (!coreTSref.current.length) { alert("No core TS data — run simulation first."); return; }
    downloadCSV(coreTSref.current, `ts_core_seed${seed}`);
  }, [seed, downloadCSV]);

  const exportDecTScsv = useCallback(() => {
    if (!decTSref.current.length) { alert("No deception TS data — run simulation first."); return; }
    downloadCSV(decTSref.current, `ts_deception_seed${seed}`);
  }, [seed, downloadCSV]);

  const exportMetaCSV = useCallback(() => {
    const p = pRef.current;
    downloadCSV([{
      seed, totalTicks: tick, condition: cond,
      omegaC: p.omegaC, omegaXi: p.omegaXi, softmaxBeta: p.softmaxBeta,
      etaPplus: p.etaPplus, etaMplus: p.etaMplus, etaPminus: p.etaPminus, etaMminus: p.etaMminus,
      gamma: p.gamma, etaPobs: p.etaPobs, etaMobs: p.etaMobs,
      lambdaPdecay: p.lambdaPdecay, lambdaMdecay: p.lambdaMdecay,
      deltaP: p.deltaP, deltaM: p.deltaM,
      childBeta: p.childBeta, sigmaMutP: p.sigmaMutP, sigmaMutM: p.sigmaMutM,
      thetaD: p.thetaD, kDetect: p.kDetect, commNoise: p.commNoise,
      Mmax: p.Mmax, imitScale: p.imitScale, T: p.T, kappa: p.kappa,
      hardDetect: p.hardDetect, capStd: p.capStd,
      expListenerWt: p.expListenerWt, hillBeta: p.hillBeta, hillKappa: p.hillKappa,
    }], `meta_seed${seed}`);
  }, [seed, tick, cond, downloadCSV]);

  // §Network snapshot: export wMat as edge list CSV (source, target, weight)
  const exportNetworkCSV = useCallback(() => {
    const Z = simR.current; if (!Z) return;
    const edges = [];
    for (const [k, val] of Z.wMat.entries()) {
      const parts = k.split("-").map(Number);
      edges.push({ source: parts[0], target: parts[1], weight: val.toFixed(6) });
    }
    if (!edges.length) { alert("No network edges yet — run simulation with cooperation first."); return; }
    downloadCSV(edges, `network_t${tick}_seed${seed}`);
  }, [tick, seed, downloadCSV]);
  const exportAll = useCallback(() => {
    // CSVs via Blob download
    if (coreTSref.current.length) downloadCSV(coreTSref.current, `ts_core_seed${seed}`);
    if (decTSref.current.length) downloadCSV(decTSref.current, `ts_deception_seed${seed}`);
    // Agent snapshot
    const Z = simR.current; if (Z) {
      const alive = Z.agents.filter(a => a.alive);
      const keys = ["id","capability","alpha","alpha_raw","p","m","xi","n_exp","energy","age","gen","coopCount","coopSuccess","lieCount","lastGamma","lastAstar"];
      downloadCSV(alive.map(a => { const row = {}; keys.forEach(k => row[k] = a[k] ?? 0); return row; }), `snap_agents_t${tick}_seed${seed}`);
      // Network snapshot (edge list)
      const edges = [];
      for (const [k, val] of Z.wMat.entries()) {
        const parts = k.split("-").map(Number);
        edges.push({ source: parts[0], target: parts[1], weight: val.toFixed(6) });
      }
      if (edges.length) downloadCSV(edges, `network_t${tick}_seed${seed}`);
    }
    // Meta
    const p = pRef.current;
    downloadCSV([{
      seed, totalTicks: tick, condition: cond,
      omegaC: p.omegaC, omegaXi: p.omegaXi, softmaxBeta: p.softmaxBeta,
      etaPplus: p.etaPplus, etaMplus: p.etaMplus, etaPminus: p.etaPminus, etaMminus: p.etaMminus,
      gamma: p.gamma, etaPobs: p.etaPobs, etaMobs: p.etaMobs,
      childBeta: p.childBeta, thetaD: p.thetaD, kDetect: p.kDetect,
      Mmax: p.Mmax, imitScale: p.imitScale,
      expListenerWt: p.expListenerWt, hillBeta: p.hillBeta, hillKappa: p.hillKappa,
    }], `meta_seed${seed}`);
    // Full diagnostics CSV (existing)
    if (diagH.length) {
      const dkeys = ["t","N","TI","meanP","meanM","varP","varM","spearRawVsRep","fracNearTop","varAlphaTrue","varAlphaHat","coopRate","meanAlpha","spearman","KL","isPooling","condition","softmaxBeta","childBeta","etaPobs","etaMobs","meanGamma","meanAstar"];
      downloadCSV(diagH.map(d => { const row = {}; dkeys.forEach(k => row[k] = d[k] ?? ""); return row; }), `diagnostics_seed${seed}`);
    }
  }, [seed, tick, cond, diagH, downloadCSV]);

  // ══════════════════════════════════════════════════════════
  // §AnalysisModal — 6 publication figures with proper axes
  // Defined as inner component so it can use hooks directly
  // ══════════════════════════════════════════════════════════
  function AnalysisModal() {
    const [activeTab, setActiveTab] = useState("fig1");
    const r1 = useRef(null), r2 = useRef(null), r3 = useRef(null);
    const r4 = useRef(null), r5 = useRef(null), r6 = useRef(null);
    const Z = simR.current;

    // ── Shared chart engine ──
    function pubChart(canvas, { title, xLabel, yLabel, series, xKey = "t", yRange }) {
      if (!canvas) return;
      const W = 760, H = 420;
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext("2d");
      const P = { l: 66, r: 22, t: 42, b: 52 };
      const CW = W - P.l - P.r, CH = H - P.t - P.b;
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#f9fafb"; ctx.fillRect(P.l, P.t, CW, CH);

      let yMin = Infinity, yMax = -Infinity;
      series.forEach(s => s.data.forEach(d => {
        const v = d[s.yKey ?? "v"]; if (v != null && isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
      }));
      if (yRange) { [yMin, yMax] = yRange; }
      else { const pad = (yMax - yMin) * 0.08 || 0.05; yMin -= pad; yMax += pad; }
      if (!isFinite(yMin)) { yMin = 0; yMax = 1; }

      let xMin = Infinity, xMax = -Infinity;
      series.forEach(s => s.data.forEach(d => { const v = d[xKey]; if (v != null) { xMin = Math.min(xMin, v); xMax = Math.max(xMax, v); } }));
      if (!isFinite(xMin)) { xMin = 0; xMax = 1; }
      if (xMax === xMin) xMax = xMin + 1;

      const tx = x => P.l + ((x - xMin) / (xMax - xMin)) * CW;
      const ty = y => P.t + CH - ((clamp(y, yMin, yMax) - yMin) / (yMax - yMin)) * CH;

      // Grid + axis ticks
      ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
      const nY = 5, nX = 7;
      ctx.font = "10.5px Arial"; ctx.fillStyle = "#6b7280"; ctx.textAlign = "right";
      for (let i = 0; i <= nY; i++) {
        const yv = yMin + (yMax - yMin) * (i / nY);
        ctx.beginPath(); ctx.moveTo(P.l, ty(yv)); ctx.lineTo(P.l + CW, ty(yv)); ctx.stroke();
        ctx.fillText(yv.toFixed(Math.abs(yMax) < 5 ? 3 : 0), P.l - 5, ty(yv) + 3.5);
      }
      ctx.textAlign = "center";
      for (let i = 0; i <= nX; i++) {
        const xv = xMin + (xMax - xMin) * (i / nX);
        ctx.strokeStyle = "#e5e7eb";
        ctx.beginPath(); ctx.moveTo(tx(xv), P.t); ctx.lineTo(tx(xv), P.t + CH); ctx.stroke();
        ctx.fillStyle = "#6b7280";
        ctx.fillText(Math.round(xv), tx(xv), P.t + CH + 16);
      }
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1.5; ctx.strokeRect(P.l, P.t, CW, CH);

      // Condition boundary markers on time-series charts
      if (xKey === "t") {
        const condChanges = [];
        let prevCond = null;
        diagH.forEach(d => { if (d.condition !== prevCond) { condChanges.push({ t: d.t, c: d.condition }); prevCond = d.condition; } });
        condChanges.forEach(({ t: ct, c: cc }) => {
          const xp = tx(ct);
          if (xp >= P.l && xp <= P.l + CW) {
            ctx.strokeStyle = cc === "A" ? "#10b981" : cc === "B" ? "#ef4444" : "#8b5cf6";
            ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
            ctx.beginPath(); ctx.moveTo(xp, P.t); ctx.lineTo(xp, P.t + CH); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = cc === "A" ? "#10b981" : cc === "B" ? "#ef4444" : "#8b5cf6";
            ctx.font = "9px Arial"; ctx.textAlign = "left";
            ctx.fillText(`Cond ${cc}`, xp + 2, P.t + 12);
          }
        });
      }

      // Lines
      series.forEach(s => {
        if (!s.data.length) return;
        ctx.strokeStyle = s.color; ctx.lineWidth = s.lw || 2; ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        let first = true;
        s.data.forEach(d => {
          const xv = d[xKey], yv = d[s.yKey ?? "v"];
          if (xv == null || yv == null || !isFinite(yv)) return;
          first ? ctx.moveTo(tx(xv), ty(yv)) : ctx.lineTo(tx(xv), ty(yv));
          first = false;
        });
        ctx.stroke(); ctx.setLineDash([]);
      });

      // Legend row
      let lx = P.l + 8, ly = P.t + 14;
      series.forEach(s => {
        ctx.fillStyle = s.color; ctx.fillRect(lx, ly - 6, 20, 4);
        ctx.fillStyle = "#374151"; ctx.font = "10.5px Arial"; ctx.textAlign = "left";
        ctx.fillText(s.label, lx + 24, ly);
        lx += ctx.measureText(s.label).width + 46;
      });

      // Titles
      ctx.fillStyle = "#111827"; ctx.font = "bold 13px Arial"; ctx.textAlign = "center";
      ctx.fillText(title, W / 2, 22);
      ctx.fillStyle = "#6b7280"; ctx.font = "11.5px Arial";
      ctx.fillText(xLabel, P.l + CW / 2, H - 8);
      ctx.save(); ctx.translate(12, P.t + CH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
    }

    // Fig 1: TI time series
    useEffect(() => {
      if (!r1.current || !tiH.length) return;
      pubChart(r1.current, {
        title: "Fig 1 — Trophic Incoherence (TI) over Time  [t=0 → present]",
        xLabel: "Simulation Tick", yLabel: "TI (hierarchy disorder)",
        series: [{ data: tiH, yKey: "v", color: "#7c3aed", label: "TI", lw: 2 }],
      });
    });

    // Fig 2: Deception evolution
    useEffect(() => {
      if (!r2.current || !diagH.length) return;
      pubChart(r2.current, {
        title: "Fig 2 — Deception Evolution: p̄(t)  m̄(t)  Var(p)(t)",
        xLabel: "Simulation Tick", yLabel: "Value",
        series: [
          { data: diagH, yKey: "meanP", color: "#ef4444", label: "p̄ (lying prob)", lw: 2.2 },
          { data: diagH, yKey: "meanM", color: "#f97316", label: "m̄ (lying mag)", lw: 2.2 },
          { data: diagH, yKey: "varP",  color: "#fca5a5", label: "Var(p)", lw: 1.2 },
          { data: diagH, yKey: "varM",  color: "#fed7aa", label: "Var(m)", lw: 1.2 },
        ],
      });
    });

    // Fig 3: Endorsement Network
    useEffect(() => {
      if (!r3.current || !Z) return;
      const canvas = r3.current;
      canvas.width = 760; canvas.height = 500;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 760, 500);
      const alive = Z.agents.filter(a => a.alive);
      if (!alive.length) return;
      const PAD = 48, W = 760, H = 500;
      const sx = x => PAD + x * ((W - 2 * PAD) / WP);
      const sy = y => PAD + y * ((H - 80 - PAD) / HP);
      const inDeg = new Map(); alive.forEach(a => inDeg.set(a.id, 0));
      const edges = [];
      for (const [k, w] of Z.wMat) {
        const [fid, tid] = k.split("-").map(Number);
        const from = alive.find(a => a.id === fid), to = alive.find(a => a.id === tid);
        if (!from || !to) continue;
        inDeg.set(tid, (inDeg.get(tid) || 0) + w); edges.push({ from, to, w });
      }
      const maxDeg = Math.max(1, ...inDeg.values()), maxW = Math.max(1, ...edges.map(e => e.w));
      // Draw edges
      edges.forEach(({ from, to, w }) => {
        const alpha = Math.min(0.85, 0.08 + 0.77 * w / maxW);
        ctx.strokeStyle = `rgba(99,102,241,${alpha.toFixed(2)})`; ctx.lineWidth = 0.4 + 2.6 * w / maxW;
        ctx.beginPath(); ctx.moveTo(sx(from.x), sy(from.y)); ctx.lineTo(sx(to.x), sy(to.y)); ctx.stroke();
        // Arrow
        const ang = Math.atan2(sy(to.y) - sy(from.y), sx(to.x) - sx(from.x));
        const ax = sx(to.x) - 9 * Math.cos(ang), ay = sy(to.y) - 9 * Math.sin(ang);
        ctx.fillStyle = `rgba(99,102,241,${alpha.toFixed(2)})`;
        ctx.beginPath(); ctx.moveTo(sx(to.x), sy(to.y));
        ctx.lineTo(ax - 4 * Math.sin(ang), ay + 4 * Math.cos(ang));
        ctx.lineTo(ax + 4 * Math.sin(ang), ay - 4 * Math.cos(ang)); ctx.fill();
      });
      // Draw nodes
      alive.forEach(a => {
        const deg = inDeg.get(a.id) || 0;
        const r = 2.5 + 15 * (deg / maxDeg);
        const col = a.p < 0.1 ? "#10b981" : a.p < 0.35 ? "#84cc16" : a.p < 0.65 ? "#f59e0b" : "#ef4444";
        ctx.beginPath(); ctx.arc(sx(a.x), sy(a.y), r, 0, 6.28);
        ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = "#374151"; ctx.lineWidth = 0.4; ctx.stroke();
        if (deg / maxDeg > 0.5) { // label top nodes
          ctx.fillStyle = "#111"; ctx.font = "8px Arial"; ctx.textAlign = "center";
          ctx.fillText(`#${a.id}`, sx(a.x), sy(a.y) - r - 2);
        }
      });
      ctx.fillStyle = "#111827"; ctx.font = "bold 13px Arial"; ctx.textAlign = "center";
      ctx.fillText("Fig 3 — Endorsement Network (node size = in-degree, color = p)", 380, 22);
      const leg = [["#10b981","p<0.1 honest"],["#84cc16","p<0.35"],["#f59e0b","p<0.65"],["#ef4444","p≥0.65 deceptive"]];
      leg.forEach(([col, lbl], i) => {
        ctx.fillStyle = col; ctx.fillRect(12 + i * 148, 465, 12, 12);
        ctx.fillStyle = "#374151"; ctx.font = "10.5px Arial"; ctx.textAlign = "left";
        ctx.fillText(lbl, 28 + i * 148, 476);
      });
      ctx.fillStyle = "#9ca3af"; ctx.font = "9.5px Arial"; ctx.textAlign = "right";
      ctx.fillText(`N=${alive.length}  edges=${edges.length}  t=${Z.tick}  seed:${seed}`, 750, 492);
    });

    // Fig 4: Signal fidelity
    useEffect(() => {
      if (!r4.current || !diagH.length) return;
      pubChart(r4.current, {
        title: "Fig 4 — Signal Fidelity: ρ(α_true, α̂)  and  fracNearTop(t)",
        xLabel: "Simulation Tick", yLabel: "Value",
        series: [
          { data: diagH, yKey: "spearRawVsRep", color: "#0ea5e9", label: "ρ(α, α̂) Spearman", lw: 2.5 },
          { data: diagH, yKey: "fracNearTop",   color: "#e11d48", label: "fracNearTop (inflation)", lw: 1.8 },
          { data: diagH, yKey: "spearman",      color: "#8b5cf6", label: "ρ(raw, norm) α", lw: 1.2, dash: [4,3] },
        ],
        yRange: [-0.05, 1.1],
      });
    });

    // Fig 5: α distribution snapshots
    useEffect(() => {
      if (!r5.current || diagH.length < 3) return;
      const canvas = r5.current; canvas.width = 760; canvas.height = 420;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 760, 420);
      const P = { l: 50, r: 20, t: 42, b: 52 };
      const CW = 760 - P.l - P.r, CH = 420 - P.t - P.b;
      ctx.fillStyle = "#f9fafb"; ctx.fillRect(P.l, P.t, CW, CH);
      const idxs = [0, Math.floor(diagH.length / 2), diagH.length - 1];
      const snaps = idxs.map(i => ({ lbl: `t=${diagH[i].t}`, col: ["#3b82f6","#f59e0b","#ef4444"][idxs.indexOf(i)], d: diagH[i] }));
      const NBINS = 30;
      let gmax = 0;
      const allBins = snaps.map(({ d }) => {
        if (!d.alphaSnap?.length) return { bins: new Array(NBINS).fill(0), mn: 0, mx: 1 };
        const mn = Math.min(...d.alphaSnap), mx = Math.max(...d.alphaSnap), rng = mx - mn || 1e-6;
        const bins = new Array(NBINS).fill(0);
        d.alphaSnap.forEach(v => bins[Math.min(NBINS-1, Math.floor(((v-mn)/rng)*NBINS))]++);
        const tot = d.alphaSnap.length; const dens = bins.map(b => b / tot);
        gmax = Math.max(gmax, ...dens); return { bins: dens, mn, mx };
      });
      if (!gmax) gmax = 0.5;
      const bw = CW / NBINS / snaps.length;
      snaps.forEach(({ col }, si) => {
        const { bins } = allBins[si];
        ctx.fillStyle = col + "aa";
        bins.forEach((v, i) => {
          const h = (v / gmax) * CH;
          ctx.fillRect(P.l + (i / NBINS) * CW + si * bw, P.t + CH - h, bw - 0.5, h);
        });
      });
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1.5; ctx.strokeRect(P.l, P.t, CW, CH);
      ctx.fillStyle = "#111827"; ctx.font = "bold 13px Arial"; ctx.textAlign = "center";
      ctx.fillText("Fig 5 — α Distribution at Early / Mid / Final Timepoints", 380, 22);
      ctx.fillStyle = "#6b7280"; ctx.font = "11px Arial";
      ctx.fillText("α value (normalized within each snapshot  0→1)", P.l + CW/2, 420 - 8);
      snaps.forEach(({ lbl, col, d }, i) => {
        const { mn, mx } = allBins[i];
        ctx.fillStyle = col; ctx.fillRect(P.l + 8 + i * 220, P.t + 8, 16, 12);
        ctx.fillStyle = "#374151"; ctx.font = "10.5px Arial"; ctx.textAlign = "left";
        ctx.fillText(`${lbl}  [${mn.toFixed(1)}–${mx.toFixed(1)}]  N=${d.N}`, P.l + 28 + i * 220, P.t + 19);
      });
    });

    // Fig 6: Cooperation Rate vs p̄ scatter
    useEffect(() => {
      if (!r6.current || !diagH.length) return;
      const canvas = r6.current; canvas.width = 760; canvas.height = 420;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 760, 420);
      const P = { l: 66, r: 22, t: 42, b: 52 };
      const CW = 760 - P.l - P.r, CH = 420 - P.t - P.b;
      ctx.fillStyle = "#f9fafb"; ctx.fillRect(P.l, P.t, CW, CH);
      const pts = diagH.filter(d => isFinite(d.coopRate) && isFinite(d.meanP));
      if (!pts.length) return;
      const xMax = Math.max(0.01, ...pts.map(d => d.meanP)) * 1.15;
      const tx = x => P.l + (x / xMax) * CW;
      const ty = y => P.t + CH - (clamp(y, 0, 1)) * CH;
      // Grid
      ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1; ctx.font = "10.5px Arial";
      [0,.25,.5,.75,1].forEach(v => {
        ctx.beginPath(); ctx.moveTo(P.l, ty(v)); ctx.lineTo(P.l+CW, ty(v)); ctx.stroke();
        ctx.fillStyle = "#6b7280"; ctx.textAlign = "right"; ctx.fillText(v.toFixed(2), P.l-5, ty(v)+3);
      });
      ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1.5; ctx.strokeRect(P.l, P.t, CW, CH);
      // Scatter: color by condition
      pts.forEach((d, i) => {
        const condCol = d.condition === "A" ? [16,185,129] : d.condition === "B" ? [239,68,68] : [139,92,246];
        const tFrac = i / pts.length;
        const [r,g,b] = condCol;
        ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + 0.5 * tFrac})`;
        ctx.beginPath(); ctx.arc(tx(d.meanP), ty(d.coopRate), 3.5, 0, 6.28); ctx.fill();
      });
      ctx.fillStyle = "#111827"; ctx.font = "bold 13px Arial"; ctx.textAlign = "center";
      ctx.fillText("Fig 6 — Cooperation Rate vs p̄  (color=condition, opacity=time)", 380, 22);
      ctx.fillStyle = "#6b7280"; ctx.font = "11.5px Arial";
      ctx.fillText("p̄ (mean lying probability)", P.l + CW/2, 420-8);
      ctx.save(); ctx.translate(12, P.t+CH/2); ctx.rotate(-Math.PI/2); ctx.fillText("Cooperation Rate", 0, 0); ctx.restore();
      const condLeg = [["#10b981","Cond A (Honest)"],["#ef4444","Cond B (Decep)"]];
      condLeg.forEach(([col, lbl], i) => {
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(P.l + 14 + i * 160, P.t + 14, 5, 0, 6.28); ctx.fill();
        ctx.fillStyle = "#374151"; ctx.font = "10.5px Arial"; ctx.textAlign = "left";
        ctx.fillText(lbl, P.l + 23 + i * 160, P.t + 18);
      });
    });

    const tabs = [["fig1","Fig 1 · TI(t)"],["fig2","Fig 2 · Decep"],["fig3","Fig 3 · Network"],["fig4","Fig 4 · Fidelity"],["fig5","Fig 5 · α Dist"],["fig6","Fig 6 · Coop×p"]];
    const refs = { fig1: r1, fig2: r2, fig3: r3, fig4: r4, fig5: r5, fig6: r6 };
    const figNames = { fig1: "fig1_TI", fig2: "fig2_deception", fig3: "fig3_network", fig4: "fig4_fidelity", fig5: "fig5_alphadist", fig6: "fig6_coopVsP" };
    const bs = { background: "#1f2937", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 5, padding: "5px 12px", cursor: "pointer", fontSize: 11, fontFamily: "monospace" };
    const ts = active => ({ padding: "7px 16px", cursor: "pointer", fontFamily: "monospace", fontSize: 11, border: "none", borderBottom: active ? "2.5px solid #6366f1" : "2.5px solid transparent", background: "none", color: active ? "#6366f1" : "#6b7280", fontWeight: active ? 700 : 400 });
    const descMap = {
      fig1: "TI(t) from t=0 to present, across all condition switches. Vertical dashed lines mark condition changes. Rising TI = hierarchy becoming more disordered as deception spreads.",
      fig2: "Evolution of deception parameters. p̄ caps near 1 when deception goes to fixation. m̄ can keep growing (no hard cap on reported advantage). Var(p) collapse = pooling / TI collapse.",
      fig3: `Endorsement network snapshot at t=${tick}. Node size = Σ endorsement received. Arrow = listener→speaker (who endorsed whom). Red = high-deception agents. If large nodes are red → deceptive agents dominating the hierarchy.`,
      fig4: "ρ(α, α̂) = Spearman correlation between true and reported influence. Starts ~1 (honest). Drops as deception rises. fracNearTop = fraction reporting > 95th pctile of true α — this is the 'inflation' signal.",
      fig5: "Distribution of true α at three timepoints. Skew shift right = Matthew effect (influence concentration). Width collapse = convergence. If early is wide and final is narrow → hierarchy has crystallized.",
      fig6: "Each point = one diagnostic checkpoint (every 100 ticks). Color = condition. As p̄ rises, does coopRate fall? If flat at 1.0 → hardDetect=false (soft mode always succeeds). Turn on hardDetect to see trade-off.",
    };

    return (
      <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={e => { if (e.target === e.currentTarget) setShowAnalysis(false); }}>
        <div style={{ background: "#fff", borderRadius: 10, width: "min(900px, 96vw)", maxHeight: "92vh", overflow: "hidden", display: "flex", flexDirection: "column", boxShadow: "0 25px 80px rgba(0,0,0,0.6)" }}>
          {/* Header */}
          <div style={{ padding: "12px 18px", borderBottom: "1px solid #e5e7eb", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#f9fafb" }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#111827", fontFamily: "monospace" }}>📊 Analysis Export — ALife Publication Figures</div>
              <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "monospace", marginTop: 2 }}>seed:{seed} · t={tick} · N={cnt.total} · Cond {cond} · {diagH.length} diagnostic pts · TI history: {tiH.length} pts (t=0→{tiH.length ? tiH[tiH.length-1].t : 0})</div>
            </div>
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
              <button onClick={exportCoreTScsv} style={{ ...bs, color: "#7dd3fc", border: "1px solid #7dd3fc" }}>📋 Core TS</button>
              <button onClick={exportDecTScsv} style={{ ...bs, color: "#fb923c", border: "1px solid #fb923c" }}>📋 Decep TS</button>
              <button onClick={exportDiagCSV} style={{ ...bs, color: "#fbbf24", border: "1px solid #fbbf24" }}>📋 Full Diag</button>
              <button onClick={exportAgentCSV} style={{ ...bs, color: "#34d399", border: "1px solid #34d399" }}>👤 Agent CSV</button>
              <button onClick={exportMetaCSV} style={{ ...bs, color: "#c084fc", border: "1px solid #c084fc" }}>⚙ Meta</button>
              <button onClick={exportAll} style={{ ...bs, color: "#6ee7b7", border: "2px solid #6ee7b7", fontWeight: 700 }}>📦 All CSVs</button>
              <button onClick={() => setShowAnalysis(false)} style={bs}>✕</button>
            </div>
          </div>
          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid #e5e7eb", padding: "0 8px", background: "#fff" }}>
            {tabs.map(([id, lbl]) => <button key={id} style={ts(activeTab === id)} onClick={() => setActiveTab(id)}>{lbl}</button>)}
          </div>
          {/* Figure area */}
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 18px", background: "#f9fafb" }}>
            <div style={{ fontSize: 11, color: "#374151", lineHeight: 1.5, marginBottom: 10, padding: "8px 12px", background: "#fff", borderRadius: 6, border: "1px solid #e5e7eb" }}>
              {descMap[activeTab]}
            </div>
            {tabs.map(([id]) => (
              <div key={id} style={{ display: activeTab === id ? "block" : "none" }}>
                <canvas ref={refs[id]} style={{ width: "100%", border: "1px solid #e5e7eb", borderRadius: 4, display: "block" }} />
                <button onClick={() => savePNG(refs[id].current, figNames[id])} style={{ ...bs, marginTop: 8, color: "#6ee7b7", border: "1px solid #6ee7b7" }}>
                  💾 Open {figNames[id]} in new tab (right-click → Save image)
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const selA = simR.current?.agents.find(a => a.id === sel && a.alive);
  const eLbl = { coop: "cooperate", coop_d: "coop(detected)", fail: "Cooperation failed", birth: "born", death: "死亡" };
  const eCol = { coop: "#6ee7b7", coop_d: "#f59e0b", fail: "#f87171", birth: "#86efac", death: "#6b7280" };

  // ══ Pre-compute ALL display values to avoid > inside JSX (Oxc parser chokes on >) ══
  const _spearRep = diag.spearRawVsRep ?? 1;
  const _fracTop = diag.fracNearTop ?? 0.05;
  const _meanP = diag.meanP ?? 0;
  const _spearRepCol = _spearRep > 0.7 ? "#6ee7b7" : "#fbbf24";
  const _fracTopCol = _fracTop > 0.2 ? "#ef4444" : "#6ee7b7";
  const _meanPcol = _meanP > 0.3 ? "#ef4444" : "#6ee7b7";
  const _regimeCol = _spearRep > 0.7 ? "#6ee7b7" : _spearRep > 0.3 ? "#fbbf24" : "#ef4444";
  const _regimeText = _spearRep > 0.7 ? "✓ Informative regime" : _spearRep > 0.3 ? "⚠ Exaggeration regime" : "✗ Saturation/Pooling — TI collapse likely";
  const _diagSpearCol = diag.spearman >= 0.95 ? "#6ee7b7" : "#fbbf24";
  const _diagKLcol = diag.KL <= 0.05 ? "#6ee7b7" : "#fbbf24";
  const _diagTop5col = (diag.overlap?.top5 || 0) >= 0.6 ? "#6ee7b7" : "#fbbf24";
  const _diagRankOk = diag.spearman >= 0.95 && diag.KL <= 0.05;
  const _diagRankCol = _diagRankOk ? "#6ee7b7" : "#ef4444";
  const _diagRankLbl = _diagRankOk ? "✓ Ranking preserved" : "⚠ Normalization affects ranking";
  const _lastH = hist.length > 0 ? hist[hist.length - 1] : null;
  const _avgAlphaStr = _lastH ? _lastH.avgAlpha?.toFixed(3) : "—";
  const _avgEstr = _lastH ? _lastH.avgE?.toFixed(2) : "—";
  const _coopRateStr = _lastH ? (_lastH.coopRate * 100).toFixed(1) : "—";
  const _avgMstr = _lastH ? (_lastH.avgM?.toFixed(3) || "0") : "—";
  const _varAlphaStr = _lastH ? (_lastH.varAlpha?.toFixed(4) || "0") : "—";
  const _selPcolor = selA ? (selA.p > 0.3 ? "#ef4444" : "#6ee7b7") : "#6ee7b7";

  return (
    <div data-theme={theme} style={{ fontFamily: "'IBM Plex Mono','Menlo',monospace", background: "var(--bg)", color: "var(--text)", position: "fixed", inset: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <style>{THEME_CSS}</style>
      {showAnalysis && <AnalysisModal />}
      <div style={{ padding: "6px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--line)", background: "var(--panel)" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, background: "var(--text)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Social Hierarchy & Deception Simulator — When Deception Spreads
          </div>
          <div style={{ fontSize: 9, color: "var(--text-dim)" }}>
            t={tick} · N={cnt.total} (♂{cnt.m}♀{cnt.f}) · TI={ti.toFixed(3)} · p̄={cnt.ap?.toFixed(3)} · m̄={cnt.am?.toFixed(3)} · <span style={{ color: "#c084fc" }}>Γ̄={diag.meanGamma?.toFixed(3)}</span> · Cond {cond} · <span style={{ color: "#c084fc" }}>seed:{seed}</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 5, alignItems: "left", flexWrap: "wrap" }}>
          {["A","B"].map(c => (
            <button key={c} onClick={() => { const prev = cond; setCond(c); if (prev === "A" && c === "B") applyIgnition(); }} style={{
              background: cond === c ? (c === "A" ? "rgba(110,231,183,0.2)" : "rgba(239,68,68,0.2)") : "transparent",
              border: `1.5px solid ${c === "A" ? "#6ee7b7" : "#ef4444"}`,
              color: c === "A" ? "#6ee7b7" : "#ef4444",
              padding: "3px 8px", borderRadius: 3, fontSize: 9, cursor: "pointer",
              fontFamily: "inherit", fontWeight: cond === c ? 700 : 400,
            }}>
              {c === "A" ? "A: Honest" : "B: Decep"}
            </button>
          ))}
          <Btn c={run ? "#f87171" : "#6ee7b7"} onClick={() => setRun(!run)}>{run ? "⏸" : "▶"}</Btn>
          <Btn c="#7dd3fc" onClick={step}>⏭</Btn>
          <Btn c="#c084fc" onClick={() => init(undefined, true)}>↺</Btn>
          <select value={spd} onChange={e => setSpd(+e.target.value)} style={ss}>
            <option value={1}>1×</option><option value={3}>3×</option><option value={5}>5×</option><option value={10}>10×</option>
          </select>
          {/* §Seed controls */}
          <div style={{ display: "flex", gap: 3, alignItems: "center", borderLeft: "1px solid rgba(110,180,100,0.15)", paddingLeft: 6 }}>
            <span style={{ fontSize: 8, color: "var(--text-dim)" }}>SEED</span>
            <input value={seedInput} onChange={e => setSeedInput(e.target.value)}
              placeholder={String(seed)} style={{ ...ss, width: 70, fontSize: 9, color: "#c084fc" }} />
            <button onClick={() => { const s = (parseInt(seedInput) >>> 0) || seed; setSeedInput(""); init(s, true); }}
              style={{ ...ss, cursor: "pointer", color: "#c084fc", border: "1px solid #c084fc", padding: "3px 6px", fontSize: 9 }}
              title="Reset with this seed">▶</button>
            <button onClick={() => { setSeedInput(""); init((Date.now() ^ (Math.floor(_rand() * 0xFFFFFF))) >>> 0, true); }}
              style={{ ...ss, cursor: "pointer", color: "#7dd3fc", border: "1px solid #7dd3fc", padding: "3px 6px", fontSize: 9 }}
              title="New random seed">⟳</button>
          </div>
          {/* §Analysis */}
          <button onClick={() => setShowAnalysis(true)}
            style={{ ...ss, cursor: "pointer", color: "#fbbf24", border: "1px solid #fbbf24", padding: "4px 10px", fontSize: 10, fontWeight: 700 }}>
            📊 Analysis
          </button>
          <button onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
            style={{ ...ss, cursor: "pointer", color: "var(--text2)", border: "1px solid var(--line)", padding: "4px 9px", fontSize: 11, fontWeight: 700 }}
            title="Toggle light / dark theme">
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
          <button onClick={exportAll}
            style={{ ...ss, cursor: "pointer", color: "#34d399", border: "1px solid #34d399", padding: "4px 10px", fontSize: 10, fontWeight: 700 }}
            title="Download all CSVs (core TS, deception TS, agents, meta, diagnostics)">
            📦 Export All
          </button>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <div className="side-panel" style={{ width: 280, overflowY: "auto", padding: "7px 12px", flexShrink: 0, borderRight: "1px solid var(--line)", background: "var(--panel)", fontSize: 10 }}>
          <Lbl>§Model — When Deception Spreads</Lbl>
          <div style={{ fontSize: 8, color: "var(--text-dim)", lineHeight: 1.6, marginBottom: 6 }}>
            α_i = C_i (Eq.1): rank signal = true capability<br />
            A: honest baseline · B: deception ON
          </div>
          <Sl l="σ_c (Cap Std)" v={par.capStd} s={v => setPar(p => ({ ...p, capStd: v }))} min={0.00} max={5} step={0.01} />
          <Sl l="σ_u (Mut Str)" v={par.mutS} s={v => setPar(p => ({ ...p, mutS: v }))} min={0.0} max={10} step={0.1} />

          <Lbl style={{ marginTop: 6 }}>§Detection &amp; Selection</Lbl>
          <div style={{ fontSize: 8, color: "var(--text-dim)", lineHeight: 1.5, marginBottom: 4 }}>
            P_det = V₀·ζ(k_d(Δ−θ_d)), V₀=0.5 (Eq.4)<br />
            P(S) ∝ softmax(β·α̂) (Eq.3)
          </div>
          <Sl l="θ_d detect" v={par.thetaD} s={v => setPar(p => ({ ...p, thetaD: v }))} min={0.1} max={5} step={0.1} />
          <Sl l="k_detect (steep)" v={par.kDetect} s={v => setPar(p => ({ ...p, kDetect: v }))} min={0.1} max={5} step={0.1} />
          <Sl l="β softmax (sel)" v={par.softmaxBeta} s={v => setPar(p => ({ ...p, softmaxBeta: v }))} min={0.1} max={10} step={0.1} />

          <Lbl style={{ marginTop: 6 }}>§Evolvable Deception</Lbl>
          <Sl l="η_p⁺ (reinforce)" v={par.etaPplus} s={v => setPar(p => ({ ...p, etaPplus: v }))} min={0.0} max={0.5} step={0.001} />
          <Sl l="η_m⁺ (reinforce)" v={par.etaMplus} s={v => setPar(p => ({ ...p, etaMplus: v }))} min={0.0} max={0.5} step={0.001} />
          <Sl l="η_p⁻ (punish)" v={par.etaPminus} s={v => setPar(p => ({ ...p, etaPminus: v }))} min={0.0} max={0.5} step={0.0001} />
          <Sl l="η_m⁻ (punish)" v={par.etaMminus} s={v => setPar(p => ({ ...p, etaMminus: v }))} min={0.0} max={0.5} step={0.0001} />
          <Sl l="η_p obs (learn)" v={par.etaPobs} s={v => setPar(p => ({ ...p, etaPobs: v }))} min={0.0} max={0.2} step={0.001} />
          <Sl l="η_m obs (learn)" v={par.etaMobs} s={v => setPar(p => ({ ...p, etaMobs: v }))} min={0.0} max={0.2} step={0.001} />
          <Sl l="δ_p (decay)" v={par.deltaP} s={v => setPar(p => ({ ...p, deltaP: v }))} min={0.0} max={0.02} step={0.001} />
          <Sl l="δ_m (decay)" v={par.deltaM} s={v => setPar(p => ({ ...p, deltaM: v }))} min={0.0} max={0.02} step={0.001} />
          <Sl l="M_max" v={par.Mmax} s={v => setPar(p => ({ ...p, Mmax: v }))} min={1} max={30} step={1} />
          <Sl l="γ (Gamma tanh)" v={par.gamma} s={v => setPar(p => ({ ...p, gamma: v }))} min={1} max={2000} step={1} />
          <Sl l="imit scale" v={par.imitScale} s={v => setPar(p => ({ ...p, imitScale: v }))} min={0.1} max={3} step={0.1} />
          <Sl l="child β (hered)" v={par.childBeta} s={v => setPar(p => ({ ...p, childBeta: v }))} min={0.0} max={1.0} step={0.05} />
          <Sl l="σ_mut p" v={par.sigmaMutP} s={v => setPar(p => ({ ...p, sigmaMutP: v }))} min={0.0} max={0.5} step={0.0001} />
          <Sl l="σ_mut m" v={par.sigmaMutM} s={v => setPar(p => ({ ...p, sigmaMutM: v }))} min={0.0} max={0.5} step={0.0001} />

          <Lbl style={{ marginTop: 6 }}>§Ignition (A→B)</Lbl>
          <div style={{ fontSize: 8, color: "var(--text-dim)", lineHeight: 1.5, marginBottom: 4 }}>
            p_i += U(0, 2·ignP), m_i += U(0, 2·ignM)<br />
            Fires once on A→B switch
          </div>
          <Sl l="ign_p (E[Δp])" v={par.ignitionP} s={v => setPar(p => ({ ...p, ignitionP: v }))} min={0.0} max={0.5} step={0.005} />
          <Sl l="ign_m (E[Δm])" v={par.ignitionM} s={v => setPar(p => ({ ...p, ignitionM: v }))} min={0.0} max={3.0} step={0.05} />

          <Lbl style={{ marginTop: 6 }}>p Propensity</Lbl>
          {[
            { l: "p<0.1", c: "#6ee7b7", k: "n" },
            { l: "0.1-0.35", c: "#a3e635", k: "lo" },
            { l: "0.35-0.65", c: "#fbbf24", k: "mi" },
            { l: ">0.65", c: "#ef4444", k: "hi" },
          ].map(({ l, c, k }) => (
            <div key={k} style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 2 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: c }} />
              <span style={{ color: c, fontSize: 9 }}>{l}</span>
              <span style={{ color: "var(--text-faint)", marginLeft: "auto", fontSize: 9 }}>{cnt[k] || 0}</span>
            </div>
          ))}

          {selA && (
            <>
              <Lbl style={{ marginTop: 6 }}>Agent #{selA.id}</Lbl>
              <div style={{ lineHeight: 1.6, color: "var(--text2)", fontSize: 9 }}>
                <div>{selA.gender === "M" ? "♂" : "♀"} · α={selA.alpha.toFixed(2)} · Gen{selA.gen}</div>
                <div>C={selA.capability.toFixed(2)} · ξ={selA.xi.toFixed(3)} · n={selA.n_exp?.toFixed(1)}</div>
                <div style={{ color: "#7dd3fc" }}>α_raw={selA.alpha_raw?.toFixed(3)} · α_norm={selA.alpha_norm?.toFixed(3)}</div>
                <div>p=<b style={{ color: _selPcolor }}>{selA.p.toFixed(3)}</b> · m={selA.m.toFixed(2)}</div>
                <div style={{ color: "#c084fc" }}>Γ={selA.lastGamma?.toFixed(3)} · A*={selA.lastAstar?.toFixed(2)}</div>
                <div>E={selA.energy.toFixed(1)} · Age={selA.age.toFixed(3)}</div>
                <div>Act: {selA.action}</div>
                <div style={{ fontSize: 8, color: "var(--text-dim)" }}>
                  🎤{selA.timeSpeaker} 👂{selA.timeListener} P={selA.coopCount ? (selA.timeSpeaker / selA.coopCount).toFixed(2) : "—"}<br />
                  Coop {selA.coopSuccess}/{selA.coopCount} Lies {selA.lieCount}
                </div>
              </div>
            </>
          )}

          <Lbl style={{ marginTop: 6 }}>p Population (deception propensity)</Lbl>
          <canvas ref={chart} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />

          <Lbl style={{ marginTop: 5 }}>TI (Trophic Incoherence)</Lbl>
          <canvas ref={tiC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>TI={ti.toFixed(4)}</div>

          <Lbl style={{ marginTop: 5 }}>α Distribution (normalized 0–1)</Lbl>
          <canvas ref={alphaDistC} style={{ width: "100%", height: 80, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />

          <Lbl style={{ marginTop: 5 }}>α̂ Reported Distribution (normalized 0–1)</Lbl>
          <canvas ref={alphaRepDistC} style={{ width: "100%", height: 80, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />

          {/* §Diagnostics Panel: raw vs normalized α comparison */}
          <Lbl style={{ marginTop: 6 }}>§Diagnostics (α raw vs norm)</Lbl>
          <div style={{ fontSize: 8, color: "var(--text2)", lineHeight: 1.6, padding: "4px 6px", background: "rgba(0,0,0,0.2)", borderRadius: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Spearman ρ:</span>
              <span style={{ color: _diagSpearCol, fontWeight: 600 }}>
                {diag.spearman?.toFixed(4) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>KL Divergence:</span>
              <span style={{ color: _diagKLcol, fontWeight: 600 }}>
                {diag.KL?.toFixed(6) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Top-5 Overlap:</span>
              <span style={{ color: _diagTop5col }}>
                {diag.overlap?.top5?.toFixed(2) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Top-10 Overlap:</span>
              <span>{diag.overlap?.top10?.toFixed(2) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2, borderTop: "1px solid rgba(110,180,100,0.1)", paddingTop: 2 }}>
              <span>Best Scale:</span>
              <span style={{ color: "#c084fc" }}>{diag.bestScale || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Calibrated KL:</span>
              <span style={{ color: "#c084fc" }}>{diag.bestKL?.toFixed(6) || "—"}</span>
            </div>
            <div style={{ marginTop: 3, fontSize: 7, color: _diagRankCol }}>
              {_diagRankLbl}
            </div>
          </div>

          <Lbl style={{ marginTop: 5 }}>Population N</Lbl>
          <canvas ref={popC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>N={cnt.total}</div>

          <Lbl style={{ marginTop: 5 }}>Mean α (Influence)</Lbl>
          <canvas ref={alphaC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>ᾱ={_avgAlphaStr}</div>

          <Lbl style={{ marginTop: 5 }}>
            <span style={{ color: "#86efac" }}>ξ̄ Experience</span>
          </Lbl>
          <canvas ref={xiC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>ξ̄={cnt.ax?.toFixed(3)}</div>

          <Lbl style={{ marginTop: 5 }}>Mean Energy</Lbl>
          <canvas ref={energyC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>Ē={_avgEstr}</div>

          <Lbl style={{ marginTop: 5 }}>Cooperation Success Rate</Lbl>
          <canvas ref={coopC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>rate={_coopRateStr}%</div>

          <Lbl style={{ marginTop: 5 }}>Mean m̄ (Deception Magnitude)</Lbl>
          <canvas ref={mChartC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>m̄={_avgMstr}</div>

          <Lbl style={{ marginTop: 5 }}>Var(α_true)</Lbl>
          <canvas ref={varAlphaC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>σ²={_varAlphaStr}</div>

          <Lbl style={{ marginTop: 5 }}>Gini(α) — Inequality</Lbl>
          <canvas ref={giniAlphaC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>G={hist.length ? hist[hist.length - 1].giniAlpha?.toFixed(4) : "—"}</div>

          <Lbl style={{ marginTop: 5 }}>Network Density</Lbl>
          <canvas ref={netDensityC} style={{ width: "100%", height: 55, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />
          <div style={{ fontSize: 8, color: "var(--text-dim)" }}>ρ_net={hist.length ? hist[hist.length - 1].netDensity?.toFixed(6) : "—"}</div>

          <Lbl style={{ marginTop: 5 }}>Spearman ρ(α, α̂)</Lbl>
          <canvas ref={spearRepC} style={{ width: "100%", height: 20, borderRadius: 3, border: "1px solid rgba(110,180,100,0.05)" }} />

          {/* §Extended Deception Diagnostics Panel */}
          <Lbl style={{ marginTop: 6 }}>§Deception Diagnostics</Lbl>
          <div style={{ fontSize: 8, color: "var(--text2)", lineHeight: 1.6, padding: "4px 6px", background: "rgba(0,0,0,0.2)", borderRadius: 3 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Var(α̂) reported:</span>
              <span style={{ color: "#fb923c", fontWeight: 600 }}>{diag.varAlphaHat?.toFixed(4) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Var(α) true:</span>
              <span style={{ color: "#38bdf8", fontWeight: 600 }}>{diag.varAlphaTrue?.toFixed(4) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>ρ(α, α̂):</span>
              <span style={{ color: _spearRepCol, fontWeight: 600 }}>
                {diag.spearRawVsRep?.toFixed(4) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Frac {'>'} 95th pctile:</span>
              <span style={{ color: _fracTopCol }}>
                {diag.fracNearTop?.toFixed(3) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>p̄ (mean lying prob):</span>
              <span style={{ color: _meanPcol }}>
                {diag.meanP?.toFixed(4) || "—"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>m̄ (mean lying mag):</span>
              <span style={{ color: "#fb923c" }}>{diag.meanM?.toFixed(4) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Var(p):</span>
              <span style={{ color: "#7dd3fc" }}>{diag.varP?.toFixed(5) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Var(m):</span>
              <span style={{ color: "#7dd3fc" }}>{diag.varM?.toFixed(5) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Γ̄ (mean Gamma):</span>
              <span style={{ color: "#c084fc", fontWeight: 600 }}>{diag.meanGamma?.toFixed(4) || "—"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Ā* (mean adv):</span>
              <span style={{ color: "#c084fc" }}>{diag.meanAstar?.toFixed(3) || "—"}</span>
            </div>
            <div style={{ marginTop: 3, fontSize: 7, color: diag.isPooling ? "#ef4444" : "var(--text-dim)", fontWeight: diag.isPooling ? 700 : 400 }}>
              {diag.isPooling ? "⚠ POOLING — TI collapse phase" : "✓ No pooling detected"}
            </div>
            <div style={{ marginTop: 3, fontSize: 7, color: _regimeCol }}>
              {_regimeText}
            </div>
          </div>

          <div style={{ height: 20 }} />
        </div>

        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--arena-grad)", overflow: "hidden", minWidth: 0 }}>
          <canvas ref={cvs} onClick={handleClick} style={{ maxWidth: "100%", maxHeight: "100%", width: WP, height: HP, borderRadius: 6, border: "1px solid var(--line)", boxShadow: "0 8px 40px var(--shadow)", cursor: "crosshair" }} />
          {/* §Arena legend */}
          <div style={{ position: "absolute", left: 14, bottom: 12, display: "flex", flexDirection: "column", gap: 4, padding: "8px 10px", background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6, fontSize: 9, color: "var(--text2)", backdropFilter: "blur(6px)", pointerEvents: "none" }}>
            <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-dim)", textTransform: "uppercase", marginBottom: 1 }}>Legend</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 9, height: 9, borderRadius: "50%", background: "#6ee7b7", display: "inline-block" }} />Agent — honest</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ display: "inline-flex", gap: 2 }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#a3e635" }} />
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#fbbf24" }} />
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#ef4444" }} />
              </span>lying (p: low→high)
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 9, height: 9, background: "var(--food-swatch)", display: "inline-block", transform: "rotate(45deg)", borderRadius: 1 }} />Food</div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 10, height: 10, borderRadius: "50%", border: "1.5px dashed var(--village-swatch)", display: "inline-block" }} />Village (breeding)</div>
          </div>
        </div>

        <div className="side-panel" style={{ width: 220, overflowY: "auto", padding: "7px 10px", flexShrink: 0, borderLeft: "1px solid var(--line)", background: "var(--panel)" }}>
          {/* §Script log */}
          <Lbl>Script {scriptEnabled ? <span style={{ color: "#6ee7b7" }}>ON</span> : <span style={{ color: "var(--text-dim)" }}>OFF</span>}</Lbl>
          <div style={{ display: "flex", gap: 4, marginBottom: 5 }}>
            <button onClick={() => setScriptEnabled(v => !v)}
              style={{ fontSize: 8, padding: "2px 6px", background: "transparent", border: `1px solid ${scriptEnabled ? "#6ee7b7" : "var(--text-dim)"}`, color: scriptEnabled ? "#6ee7b7" : "var(--text-dim)", borderRadius: 2, cursor: "pointer", fontFamily: "inherit" }}>
              {scriptEnabled ? "⏸ Disable" : "▶ Enable"}
            </button>
            <button onClick={() => setShowScriptEditor(v => !v)}
              style={{ fontSize: 8, padding: "2px 6px", background: "transparent", border: "1px solid #fbbf24", color: "#fbbf24", borderRadius: 2, cursor: "pointer", fontFamily: "inherit" }}>
              ✏ Edit Script
            </button>
          </div>
          {/* Script event list */}
          {script.map(ev => (
            <div key={ev.id} style={{ fontSize: 7.5, lineHeight: 1.4, padding: "2px 4px", marginBottom: 2, borderRadius: 2, background: firedR.current.has(ev.id) ? "rgba(110,231,183,0.08)" : "rgba(251,191,36,0.05)", borderLeft: `2px solid ${firedR.current.has(ev.id) ? "#6ee7b7" : "#fbbf24"}` }}>
              <span style={{ color: firedR.current.has(ev.id) ? "#6ee7b7" : "#fbbf24" }}>t={ev.t}</span>
              {" "}<span style={{ color: "#7a8a74" }}>{ev.label}</span>
            </div>
          ))}
          {/* Script fired log */}
          {scriptLog.length > 0 && (
            <div style={{ marginTop: 4, borderTop: "1px solid rgba(110,180,100,0.1)", paddingTop: 3 }}>
              {scriptLog.slice().reverse().map((l, i) => (
                <div key={i} style={{ fontSize: 7, color: "#6ee7b7", lineHeight: 1.4, padding: "1px 2px" }}>{l.msg}</div>
              ))}
            </div>
          )}
          <Lbl style={{ marginTop: 8 }}>Events</Lbl>
          {evts.slice().reverse().slice(0, 50).map((ev, i) => (
            <div key={i} style={{ fontSize: 8, lineHeight: 1.3, padding: "2px 3px", marginBottom: 1.5, borderRadius: 2, borderLeft: `2px solid ${eCol[ev.type] || "var(--text-faint)"}` }}>
              <span style={{ color: "var(--text-faint)" }}>t{ev.t}</span>
              {" "}<span style={{ color: "#7a8a74" }}>
                {eLbl[ev.type] || ev.type}
                {ev.spk !== undefined && ` 🎤${ev.spk}`}
                {ev.ids && ` [${ev.ids.join(",")}]`}
                {ev.d && ` d=${ev.d}`}
                {ev.a !== undefined && !ev.ids && ` #${ev.a}`}
                {ev.score && ` s=${ev.score}`}
              </span>
            </div>
          ))}
          {evts.length === 0 && <div style={{ fontSize: 10, color: "var(--text-faint)", fontStyle: "italic" }}>Press ▶ ...</div>}
        </div>
      </div>

      {/* §Hidden canvases for auto-export — off-screen, always mounted */}
      <div style={{ position: "absolute", left: -9999, top: -9999, pointerEvents: "none", opacity: 0 }}>
        <canvas ref={hc1} /><canvas ref={hc2} /><canvas ref={hc3} />
        <canvas ref={hc4} /><canvas ref={hc5} /><canvas ref={hc6} />
      </div>

      {/* §Script Editor Modal */}
      {showScriptEditor && <ScriptEditorModal />}
    </div>
  );
}

  // ══════════════════════════════════════════════════════════
  // §ScriptEditorModal — editable list of timed events
  // ══════════════════════════════════════════════════════════
  function ScriptEditorModal() {
    const [draft, setDraft] = useState(script.map(e => ({ ...e })));
    const [newT, setNewT] = useState("");
    const [newAction, setNewAction] = useState("setCond");
    const [newValue, setNewValue] = useState("B");
    const [newLabel, setNewLabel] = useState("");
    const [newParamKey, setNewParamKey] = useState("softmaxBeta");
    const [newParamVal, setNewParamVal] = useState("1.0");

    const actionOptions = [
      { value: "setCond", label: "Switch Condition", valueOpts: ["A","B"] },
      { value: "setParam", label: "Set Parameter (ablation)", valueOpts: null },
      { value: "autoExport", label: "Auto-save all figs + CSVs", valueOpts: null },
      { value: "pause", label: "Pause simulation", valueOpts: null },
    ];
    const ablationParams = [
      { key: "softmaxBeta", label: "β softmax (R3.3)", defaultAblation: "1.0" },
      { key: "etaPobs", label: "η_p obs (R3.1)", defaultAblation: "0" },
      { key: "etaMobs", label: "η_m obs (R3.1)", defaultAblation: "0" },
      { key: "childBeta", label: "child β (R3.2)", defaultAblation: "0" },
      { key: "etaPplus", label: "η_p⁺ reinforce", defaultAblation: "0" },
      { key: "etaMplus", label: "η_m⁺ reinforce", defaultAblation: "0" },
      { key: "imitScale", label: "imit scale", defaultAblation: "0" },
      { key: "thetaD", label: "θ_d detect", defaultAblation: "0.5" },
      { key: "kDetect", label: "k_detect", defaultAblation: "5.0" },
      { key: "gamma", label: "γ (Gamma tanh)", defaultAblation: "200" },
      { key: "ignitionP", label: "ign_p (ignition p)", defaultAblation: "0.05" },
      { key: "ignitionM", label: "ign_m (ignition m)", defaultAblation: "0.5" },
    ];
    const opt = actionOptions.find(o => o.action === newAction) || actionOptions[0];

    const save = () => { setScript(draft); setShowScriptEditor(false); };
    const addEvent = () => {
      const t2 = parseInt(newT);
      if (!t2 || t2 <= 0) return;
      let lbl, val;
      if (newAction === "setParam") {
        val = `${newParamKey}=${newParamVal}`;
        lbl = newLabel || `Set ${newParamKey} → ${newParamVal}`;
      } else {
        val = newValue;
        lbl = newLabel || (newAction === "setCond" ? `Switch → Cond ${newValue}` : newAction === "autoExport" ? "Auto-save all figs + CSVs" : "Pause simulation");
      }
      setDraft(prev => [...prev, { id: `s${Date.now()}`, t: t2, action: newAction, value: val, label: lbl }].sort((a, b) => a.t - b.t));
      setNewT(""); setNewLabel("");
    };
    // §Experiment presets — full condition set of "When Deception Spreads"
    //   Channel encodings: Ch1(self-reinforce) off ⇒ η±_p,η±_m = 0;
    //   Ch2(observation) off ⇒ η^obs_p,η^obs_m = 0;  Ch3(inheritance) off ⇒ β_c = 0.
    const sw = { id: "_sw", t: 40000, action: "setCond", value: "B", label: "Switch → Cond B (Deception)" };
    const exp = { id: "_ex", t: 100000, action: "autoExport", label: "Auto-save all figs + CSVs" };
    const off1 = [ // Ch1 off
      { t: 40000, action: "setParam", value: "etaPplus=0", label: "Ch1 off: η⁺_p=0" },
      { t: 40000, action: "setParam", value: "etaMplus=0", label: "Ch1 off: η⁺_m=0" },
      { t: 40000, action: "setParam", value: "etaPminus=0", label: "Ch1 off: η⁻_p=0" },
      { t: 40000, action: "setParam", value: "etaMminus=0", label: "Ch1 off: η⁻_m=0" },
    ];
    const off2 = [ // Ch2 off
      { t: 40000, action: "setParam", value: "etaPobs=0", label: "Ch2 off: η^obs_p=0" },
      { t: 40000, action: "setParam", value: "etaMobs=0", label: "Ch2 off: η^obs_m=0" },
    ];
    const off3 = [ { t: 40000, action: "setParam", value: "childBeta=0", label: "Ch3 off: β_c=0" } ];
    const mk = (...evs) => evs.flat().map((e, i) => ({ ...e, id: `${e.id || "ab"}${i}` }));
    const ABLATION_PRESETS = {
      // ── Fig.3 phase-switch controls ──
      "C0 (honest)": mk(exp),                              // never switch — stays in Cond A
      "C_null (channels off)": mk(sw, off1, off2, off3, exp),
      "Full": mk(sw, exp),
      // ── Fig.4/5 single-channel sufficiency ──
      "S1 (Ch1 only)": mk(sw, off2, off3, exp),
      "S2 (Ch2 only)": mk(sw, off1, off3, exp),
      "S3 (Ch3 only)": mk(sw, off1, off2, exp),
      // ── Fig.4/5 removal-from-Full necessity ──
      "Full−Ch1": mk(sw, off1, exp),
      "Full−Ch2": mk(sw, off2, exp),
      "Full−Ch3": mk(sw, off3, exp),
      // ── Fig.2 β calibration (Phase A, HONEST throughout) ──
      //   Fig.2 measures F(t) over 0–40k under different selection pressures β, with
      //   deception OFF. β must therefore be set from t=0 (Phase A), NOT at the switch.
      //   No A→B switch is included so the run stays honest, matching the Fig.2 protocol.
      "β=1": mk({ t: 0, action: "setParam", value: "softmaxBeta=1", label: "β=1 (Phase A calib)" }, exp),
      "β=2": mk({ t: 0, action: "setParam", value: "softmaxBeta=2", label: "β=2 (Phase A calib)" }, exp),
      "β=5": mk({ t: 0, action: "setParam", value: "softmaxBeta=5", label: "β=5 (Phase A calib)" }, exp),
      // ── Fig.6 punishment(λ) × inheritance(β_c) 2×2 ──
      //   λ = η⁻/η⁺. Hold η⁺_p=0.10,η⁺_m=0.20 ⇒ λ=0.3 → η⁻_p=0.03,η⁻_m=0.06; λ=1.2 → η⁻_p=0.12,η⁻_m=0.24
      "λ0.3 × β_c0.2": mk(sw, { t: 40000, action: "setParam", value: "etaPminus=0.03", label: "λ=0.3" }, { t: 40000, action: "setParam", value: "etaMminus=0.06", label: "λ=0.3" }, { t: 40000, action: "setParam", value: "childBeta=0.2", label: "β_c=0.2" }, exp),
      "λ1.2 × β_c0.2": mk(sw, { t: 40000, action: "setParam", value: "etaPminus=0.12", label: "λ=1.2" }, { t: 40000, action: "setParam", value: "etaMminus=0.24", label: "λ=1.2" }, { t: 40000, action: "setParam", value: "childBeta=0.2", label: "β_c=0.2" }, exp),
      "λ0.3 × β_c0.9": mk(sw, { t: 40000, action: "setParam", value: "etaPminus=0.03", label: "λ=0.3" }, { t: 40000, action: "setParam", value: "etaMminus=0.06", label: "λ=0.3" }, { t: 40000, action: "setParam", value: "childBeta=0.9", label: "β_c=0.9" }, exp),
      "λ1.2 × β_c0.9": mk(sw, { t: 40000, action: "setParam", value: "etaPminus=0.12", label: "λ=1.2" }, { t: 40000, action: "setParam", value: "etaMminus=0.24", label: "λ=1.2" }, { t: 40000, action: "setParam", value: "childBeta=0.9", label: "β_c=0.9" }, exp),
    };
    const removeEvent = (id) => setDraft(prev => prev.filter(e => e.id !== id));
    const bs2 = { padding: "4px 10px", borderRadius: 4, cursor: "pointer", fontSize: 11, fontFamily: "monospace", border: "1px solid" };

    return (
      <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 2000, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={e => { if (e.target === e.currentTarget) setShowScriptEditor(false); }}>
        <div style={{ background: "var(--panel-solid)", border: "1px solid rgba(110,180,100,0.15)", borderRadius: 8, width: "min(560px,94vw)", maxHeight: "80vh", overflow: "hidden", display: "flex", flexDirection: "column", fontFamily: "monospace" }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(110,180,100,0.1)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#6ee7b7" }}>✏ Experiment Script Editor</div>
              <div style={{ fontSize: 9, color: "var(--text-dim)", marginTop: 2 }}>Events fire automatically at specified ticks. Resets on ↺. Changes take effect on Save.</div>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button onClick={() => setDraft(DEFAULT_SCRIPT.map(e=>({...e})))} style={{ ...bs2, color: "#fbbf24", borderColor: "#fbbf24", background: "transparent" }}>Reset default</button>
              <button onClick={save} style={{ ...bs2, color: "#6ee7b7", borderColor: "#6ee7b7", background: "rgba(110,231,183,0.1)" }}>✓ Save</button>
              <button onClick={() => setShowScriptEditor(false)} style={{ ...bs2, color: "#6b7280", borderColor: "#374151", background: "transparent" }}>✕</button>
            </div>
          </div>
          {/* §Ablation preset buttons */}
          <div style={{ padding: "6px 16px", borderBottom: "1px solid rgba(110,180,100,0.08)", display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 8, fontWeight: 700, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.1em", marginRight: 4 }}>Presets:</span>
            {Object.entries(ABLATION_PRESETS).map(([name, events]) => (
              <button key={name} onClick={() => setDraft(events.map(e => ({ ...e, id: `${e.id}_${Date.now()}` })))}
                style={{ ...bs2, color: "#c084fc", borderColor: "#c084fc", background: "rgba(192,132,252,0.06)", padding: "3px 8px", fontSize: 10 }}>
                {name}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
            {/* Event list */}
            {draft.length === 0 && <div style={{ fontSize: 11, color: "var(--text-dim)", fontStyle: "italic", marginBottom: 8 }}>No events. Add one below.</div>}
            {draft.sort((a,b)=>a.t-b.t).map(ev => (
              <div key={ev.id} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7, padding: "6px 10px", background: "rgba(110,180,100,0.04)", borderRadius: 4, border: "1px solid rgba(110,180,100,0.08)" }}>
                <div style={{ minWidth: 70, fontSize: 12, color: "#fbbf24", fontWeight: 700 }}>t = {ev.t}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: "var(--text)" }}>{ev.label}</div>
                  <div style={{ fontSize: 9, color: "var(--text-dim)" }}>{ev.action}{ev.value ? ` → "${ev.value}"` : ""}</div>
                </div>
                <button onClick={() => removeEvent(ev.id)} style={{ ...bs2, color: "#f87171", borderColor: "#f87171", background: "transparent", padding: "2px 7px", fontSize: 10 }}>✕</button>
              </div>
            ))}

            {/* Add new event */}
            <div style={{ borderTop: "1px solid rgba(110,180,100,0.1)", paddingTop: 10, marginTop: 6 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "var(--text-dim)", marginBottom: 7, textTransform: "uppercase", letterSpacing: "0.1em" }}>Add Event</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontSize: 8, color: "var(--text-dim)" }}>TICK</span>
                  <input value={newT} onChange={e => setNewT(e.target.value)} placeholder="e.g. 30000"
                    style={{ width: 90, padding: "4px 7px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "#fbbf24", fontFamily: "monospace", fontSize: 11 }} />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontSize: 8, color: "var(--text-dim)" }}>ACTION</span>
                  <select value={newAction} onChange={e => setNewAction(e.target.value)}
                    style={{ padding: "4px 6px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "var(--text)", fontFamily: "monospace", fontSize: 11 }}>
                    {actionOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                </div>
                {newAction === "setCond" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    <span style={{ fontSize: 8, color: "var(--text-dim)" }}>COND</span>
                    <select value={newValue} onChange={e => setNewValue(e.target.value)}
                      style={{ padding: "4px 6px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "#c084fc", fontFamily: "monospace", fontSize: 11 }}>
                      {["A","B"].map(c => <option key={c} value={c}>Cond {c}</option>)}
                    </select>
                  </div>
                )}
                {newAction === "setParam" && (
                  <>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 8, color: "var(--text-dim)" }}>PARAM</span>
                      <select value={newParamKey} onChange={e => { setNewParamKey(e.target.value); const ap = ablationParams.find(p => p.key === e.target.value); if (ap) setNewParamVal(ap.defaultAblation); }}
                        style={{ padding: "4px 6px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "#fbbf24", fontFamily: "monospace", fontSize: 11 }}>
                        {ablationParams.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}
                      </select>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 8, color: "var(--text-dim)" }}>VALUE</span>
                      <input value={newParamVal} onChange={e => setNewParamVal(e.target.value)} placeholder="0"
                        style={{ width: 60, padding: "4px 7px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "#fbbf24", fontFamily: "monospace", fontSize: 11 }} />
                    </div>
                  </>
                )}
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontSize: 8, color: "var(--text-dim)" }}>LABEL (optional)</span>
                  <input value={newLabel} onChange={e => setNewLabel(e.target.value)} placeholder="description"
                    style={{ width: 140, padding: "4px 7px", background: "var(--input)", border: "1px solid rgba(110,180,100,0.2)", borderRadius: 3, color: "var(--text)", fontFamily: "monospace", fontSize: 11 }} />
                </div>
                <button onClick={addEvent} style={{ ...bs2, marginTop: 14, color: "#6ee7b7", borderColor: "#6ee7b7", background: "rgba(110,231,183,0.08)", fontSize: 12 }}>+ Add</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
function Lbl({ children, style }) { return <div style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--text-dim)", marginBottom: 4, ...style }}>{children}</div>; }
function Sl({ l, v, s, min, max, step }) {
  return (<div style={{ marginBottom: 5 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9 }}>
      <span style={{ color: "var(--text-dim)" }}>{l}</span>
      <span style={{ color: "var(--text2)", fontWeight: 600 }}>{v < 1 ? v.toFixed(3) : v.toFixed(1)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={v} onChange={e => s(+e.target.value)}
      style={{ width: "100%", height: 3, appearance: "none", background: "var(--line)", borderRadius: 2, outline: "none", cursor: "pointer", accentColor: "var(--accent)" }} />
  </div>);
}
function Btn({ children, c, onClick }) { return <button onClick={onClick} style={{ background: "transparent", border: `1px solid ${c}`, color: c, padding: "3px 8px", borderRadius: 3, fontSize: 11, cursor: "pointer", fontFamily: "inherit", fontWeight: 600 }}>{children}</button>; }
const ss = { background: "var(--input)", border: "1px solid var(--line)", color: "var(--text2)", padding: "3px 5px", borderRadius: 3, fontSize: 10, fontFamily: "inherit" };
