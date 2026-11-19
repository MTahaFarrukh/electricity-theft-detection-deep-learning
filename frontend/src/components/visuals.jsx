import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useApi } from "../hooks";
import { NAV } from "../nav";
import { CountUp, useMounted, useReducedMotion } from "../motion";
import { Icon } from "./ui";

/* ====================================================================
   Hero: a consumption curve that suddenly drops (the pattern the model looks for),
   with the anomaly pulsing, next to the fleet split as a donut.
   ==================================================================== */
const W = 1200, H = 220, STEP_X = 820;
const logistic = (x) => 1 / (1 + Math.exp(-(x - STEP_X) / 7));
const curve = (x, k = 1, phase = 0) => {
  const s = logistic(x);
  const level = 100 + 56 * s;
  const amp = (46 - 30 * s) * k;
  return level + amp * 0.55 * Math.sin(x / 38 + phase) + amp * 0.35 * Math.sin(x / 11 + 1 + phase) + amp * 0.2 * Math.sin(x / 5.5 + phase * 2);
};
const pathFor = (fn) => {
  let d = "";
  for (let x = 0; x <= W; x += 6) d += `${x ? "L" : "M"}${x} ${fn(x).toFixed(1)}`;
  return d;
};
const MAIN_PATH = pathFor((x) => curve(x));
const GHOST_A = pathFor((x) => curve(x, 0.8, 1.7) - 26);
const GHOST_B = pathFor((x) => curve(x, 0.6, 3.1) + 24);
const ANOMALY = { x: STEP_X, y: curve(STEP_X) };

export function Donut({ suspicious, normal }) {
  const mounted = useMounted(250);
  const total = suspicious + normal;
  const sp = total ? (suspicious / total) * 100 : 0;
  const np = total ? 100 - sp : 0;
  const R = 52;
  const ring = { cx: 70, cy: 70, r: R, fill: "none", strokeWidth: 13, pathLength: 100, transform: "rotate(-90 70 70)" };
  return (
    <div className="donut" role="img" aria-label={`${suspicious} suspicious and ${normal} normal meters`}>
      <svg viewBox="0 0 140 140">
        <circle {...ring} className="donut-track" strokeDasharray="100 100" />
        <circle {...ring} className="donut-normal" strokeDasharray={`${mounted ? Math.max(0, np - 1) : 0} 100`} />
        <circle {...ring} className="donut-suspicious" strokeDasharray={`${mounted ? sp : 0} 100`}
          strokeDashoffset={-np} />
      </svg>
      <div className="donut-centre">
        <strong>{total ? <CountUp value={sp} decimals={1} suffix="%" /> : "—"}</strong>
        <span>suspicious</span>
      </div>
    </div>
  );
}

export function Hero({ data, aside }) {
  return (
    <section className="hero">
      <svg className="hero-waves" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMax slice" aria-hidden="true">
        <defs>
          <filter id="glow" x="-10%" y="-40%" width="120%" height="180%">
            <feGaussianBlur stdDeviation="3.5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <linearGradient id="wave-grad" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0" stopColor="#6aa6ee" /><stop offset="0.66" stopColor="#8cc0ff" /><stop offset="0.7" stopColor="#ff9a70" /><stop offset="1" stopColor="#ff7a45" />
          </linearGradient>
        </defs>
        <path d={GHOST_A} className="wave wave-ghost" />
        <path d={GHOST_B} className="wave wave-ghost wave-ghost-2" />
        <path d={MAIN_PATH} className="wave wave-main" pathLength="1" filter="url(#glow)" stroke="url(#wave-grad)" />
        <path d={MAIN_PATH} className="wave wave-flow" pathLength="100" />
        <g className="anomaly" transform={`translate(${ANOMALY.x} ${ANOMALY.y})`}>
          <circle r="7" className="anomaly-ring" />
          <circle r="7" className="anomaly-ring anomaly-ring-2" />
          <circle r="4.5" className="anomaly-dot" />
        </g>
      </svg>

      <div className="hero-body">
        <div className="hero-copy">
          <span className="live live-light"><span className="dot" aria-hidden="true" />Live · refreshes every 5 s</span>
          <h1>Which meters should we inspect first?</h1>
          <p>
            The model scores every meter’s full consumption history. Scores at or above the validated threshold
            raise an alert: investigation recommended, not proof of theft.
          </p>
          <div className="hero-stats">
            <div className="hero-stat"><span>Meters monitored</span><strong><CountUp value={data.n_meters} /></strong></div>
            <div className="hero-stat hot"><span>Suspicious</span><strong><CountUp value={data.n_suspicious} /></strong></div>
            <div className="hero-stat"><span>Normal</span><strong><CountUp value={data.n_normal} /></strong></div>
            <div className="hero-stat"><span>Open alerts</span><strong><CountUp value={data.alerts_by_status.OPEN} /></strong></div>
          </div>
        </div>
        <div className="hero-side">
          <Donut suspicious={data.n_suspicious} normal={data.n_normal} />
          <div className="donut-legend">
            <span><i className="sw sw-normal" />Normal</span>
            <span><i className="sw sw-sus" />Suspicious</span>
          </div>
          {aside}
        </div>
      </div>
    </section>
  );
}

/* ====================================================================
   RiskDial: half-circle gauge; the needle swings to the score and the
   decision threshold is marked on the arc.
   ==================================================================== */
const CX = 120, CY = 122, RAD = 92;
const pt = (s, r = RAD) => [CX - r * Math.cos(Math.PI * s), CY - r * Math.sin(Math.PI * s)];
const arc = (a, b) => {
  const [x1, y1] = pt(a), [x2, y2] = pt(b);
  return `M${x1.toFixed(2)} ${y1.toFixed(2)}A${RAD} ${RAD} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
};

export function RiskDial({ score, threshold }) {
  const mounted = useMounted(200);
  if (score == null) return <p className="muted">This meter has not been scored.</p>;
  const over = threshold != null && score >= threshold;
  const s = mounted ? Math.min(1, Math.max(0, score)) : 0;
  const [tx1, ty1] = threshold != null ? pt(threshold, RAD - 14) : [0, 0];
  const [tx2, ty2] = threshold != null ? pt(threshold, RAD + 14) : [0, 0];
  return (
    <div className={`dial ${over ? "over" : ""}`} role="img"
      aria-label={`Risk score ${score.toFixed(3)}${threshold != null ? `, alert threshold ${threshold.toFixed(3)}` : ""}`}>
      <svg viewBox="0 0 240 150">
        <path d={arc(0, 1)} className="dial-track" />
        {threshold != null && <path d={arc(0, threshold)} className="dial-zone dial-zone-ok" />}
        {threshold != null && <path d={arc(threshold, 1)} className="dial-zone dial-zone-hot" />}
        <path d={arc(0, 1)} className="dial-value" pathLength="1" strokeDasharray={`${s} 1`} />
        {threshold != null && <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} className="dial-tick" />}
        <g className="dial-needle" style={{ transform: `rotate(${s * 180}deg)` }}>
          <path d={`M${CX - RAD + 22} ${CY}L${CX - 6} ${CY - 3}L${CX - 6} ${CY + 3}Z`} />
        </g>
        <circle cx={CX} cy={CY} r="7" className="dial-hub" />
        <text x={CX - RAD} y={CY + 18} textAnchor="middle" className="dial-label">0</text>
        <text x={CX + RAD} y={CY + 18} textAnchor="middle" className="dial-label">1</text>
      </svg>
      <div className="dial-readout">
        <strong><CountUp value={score} decimals={3} duration={1500} /></strong>
        {threshold != null && <span>Alert threshold {threshold.toFixed(3)}</span>}
      </div>
    </div>
  );
}

/* ====================================================================
   Splash: a one-off intro, once per browser session.
   ==================================================================== */
export function Splash() {
  const reduced = useReducedMotion();
  const [phase, setPhase] = useState(() => {
    try { return sessionStorage.getItem("etd-splash") ? "done" : "show"; } catch { return "show"; }
  });
  useEffect(() => {
    if (phase !== "show") return undefined;
    if (reduced) { setPhase("done"); return undefined; }
    try { sessionStorage.setItem("etd-splash", "1"); } catch { /* ignore */ }
    const a = setTimeout(() => setPhase("out"), 1700);
    const b = setTimeout(() => setPhase("done"), 2300);
    return () => { clearTimeout(a); clearTimeout(b); };
  }, [phase, reduced]);
  if (phase === "done") return null;
  return (
    <div className={`splash ${phase === "out" ? "out" : ""}`} aria-hidden="true">
      <svg viewBox="0 0 120 120" className="splash-dial">
        <circle cx="60" cy="60" r="48" className="splash-ring" pathLength="1" />
        {Array.from({ length: 11 }, (_, i) => {
          const a = (-210 + i * 24) * (Math.PI / 180);
          return <line key={i} x1={60 + 40 * Math.cos(a)} y1={60 + 40 * Math.sin(a)} x2={60 + 46 * Math.cos(a)} y2={60 + 46 * Math.sin(a)}
            className="splash-tick" style={{ animationDelay: `${0.5 + i * 0.05}s` }} />;
        })}
        <g className="splash-needle"><path d="M60 60L96 36" /></g>
        <circle cx="60" cy="60" r="5" fill="#fff" />
      </svg>
      <div className="splash-title">Theft Risk Monitor</div>
      <div className="splash-sub">Inspection prioritisation</div>
    </div>
  );
}

/* ====================================================================
   Theme toggle (system -> light -> dark) and command palette (Ctrl/Cmd+K)
   ==================================================================== */
export function ThemeToggle({ mode, onCycle }) {
  const icon = { system: "auto", light: "sun", dark: "moon" }[mode];
  return (
    <button className="side-btn" onClick={onCycle} aria-label={`Theme: ${mode}. Click to change.`}>
      <Icon name={icon} /><span>Theme: {mode[0].toUpperCase() + mode.slice(1)}</span>
    </button>
  );
}

export function CommandPalette({ open, onClose }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const input = useRef(null);
  const { data: meters } = useApi(() => (open ? api.meters() : Promise.resolve([])), [open]);

  useEffect(() => {
    if (open) { setQuery(""); setIndex(0); setTimeout(() => input.current?.focus(), 30); }
  }, [open]);
  useEffect(() => setIndex(0), [query]);

  const q = query.trim().toLowerCase();
  const items = [
    ...NAV.filter(([, label]) => label.toLowerCase().includes(q)).map(([to, label, icon]) => ({ to, label, icon, hint: "Page" })),
    ...(meters || []).filter((m) => m.meter_id.includes(q)).slice(0, 8).map((m) => ({
      to: `/meters/${encodeURIComponent(m.meter_id)}`, label: `Meter ${m.meter_id}`, icon: "meters",
      hint: m.latest_prediction === "SUSPICIOUS" ? "Suspicious" : "Normal" })),
  ];
  const go = (item) => { if (item) { navigate(item.to); onClose(); } };

  if (!open) return null;
  const onKey = (e) => {
    if (e.key === "Escape") onClose();
    else if (e.key === "ArrowDown") { e.preventDefault(); setIndex((i) => Math.min(items.length - 1, i + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setIndex((i) => Math.max(0, i - 1)); }
    else if (e.key === "Enter") go(items[index]);
  };
  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div className="palette" role="dialog" aria-label="Quick search" onMouseDown={(e) => e.stopPropagation()}>
        <div className="palette-input">
          <Icon name="search" />
          <input ref={input} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={onKey}
            placeholder="Jump to a page or type a meter ID" aria-label="Quick search" />
          <kbd>Esc</kbd>
        </div>
        <ul className="palette-list">
          {items.length === 0 && <li className="palette-empty">Nothing matches “{query}”.</li>}
          {items.map((item, i) => (
            <li key={item.to}>
              <button className={i === index ? "active" : ""} onMouseEnter={() => setIndex(i)} onClick={() => go(item)}>
                <Icon name={item.icon} /><span>{item.label}</span><em>{item.hint}</em>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
