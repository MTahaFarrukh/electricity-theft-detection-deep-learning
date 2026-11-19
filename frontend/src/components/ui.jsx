import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { API_BASE, api } from "../api";
import { CountUp, useMounted } from "../motion";

/* ---------- icons (24px, stroke) ---------- */
const ICONS = {
  overview: <><rect x="3" y="3" width="7.5" height="7.5" rx="1.5" /><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" /><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" /><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" /></>,
  meters: <><circle cx="12" cy="12" r="9" /><path d="M12 12l4.2-4.2" /><path d="M7 16.5h.01M5.2 12h.01M7 7.5h.01M12 5.2v.01" /></>,
  alerts: <><path d="M6 9.5a6 6 0 1 1 12 0c0 5 2 6.5 2 6.5H4s2-1.5 2-6.5" /><path d="M10 19.5a2 2 0 0 0 4 0" /></>,
  test: <><path d="M14.5 2v6.5l4.5 4.5M9.5 2v6.5L4 16c-1.5 2.5 0 6 3 6h10c3 0 4.5-3.5 3-6l-5.5-7.5V2h-5z" /><path d="M7 16h10" /><path d="M11 11h2" /></>,
  model: <path d="M5 20V11M12 20V4M19 20v-6" />,
  about: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5.5M12 7.5v.01" /></>,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="M16 16l4.5 4.5" /></>,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" /></>,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  auto: <><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>,
};

export function Icon({ name }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICONS[name]}</svg>
  );
}

export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" strokeWidth="2.2" opacity=".55" />
      <g className="needle"><path d="M16 16l7-7" stroke="var(--series-2)" strokeWidth="2.8" strokeLinecap="round" /></g>
      <circle cx="16" cy="16" r="2.6" fill="currentColor" />
    </svg>
  );
}

/* ---------- status chips ---------- */
// Status is never colour-alone: icon + label + colour.
export function RiskBadge({ prediction }) {
  if (!prediction) return <span className="badge badge-none">Not scored</span>;
  const suspicious = prediction === "SUSPICIOUS";
  return (
    <span className={`badge ${suspicious ? "badge-suspicious" : "badge-normal"}`}>
      <span aria-hidden="true">{suspicious ? "⚠" : "✓"}</span>
      {suspicious ? "Suspicious" : "Normal"}
    </span>
  );
}

export function AlertStatus({ status }) {
  const icon = { OPEN: "●", REVIEWED: "✓", DISMISSED: "–" }[status];
  return (
    <span className={`badge badge-${status.toLowerCase()}`}>
      <span aria-hidden="true">{icon}</span>
      {status[0] + status.slice(1).toLowerCase()}
    </span>
  );
}

/* ---------- risk score meter: the score on a 0-1 track with the decision threshold marked ---------- */
export function RiskMeter({ score, threshold, size = "md" }) {
  if (score == null) return <span className="muted">—</span>;
  const over = threshold != null && score >= threshold;
  const label = `Risk score ${score.toFixed(3)}${threshold != null ? `, alert threshold ${threshold.toFixed(3)}` : ""}`;
  return (
    <div className={`riskmeter riskmeter-${size} ${over ? "over" : ""}`} role="img" aria-label={label}>
      <div className="riskmeter-track">
        <div className="riskmeter-fill" style={{ width: `${Math.min(100, score * 100)}%` }} />
        {threshold != null && <div className="riskmeter-tick" style={{ left: `${threshold * 100}%` }} />}
      </div>
      <span className="riskmeter-value">{score.toFixed(3)}</span>
    </div>
  );
}

/* ---------- layout pieces ---------- */
export function PageHeader({ title, sub, aside }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {aside}
    </div>
  );
}

export function StatTile({ label, value, note, tone, decimals }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {typeof value === "number" ? <CountUp value={value} decimals={decimals} /> : (value ?? "—")}
      </div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function StatStrip({ children }) {
  return <div className="stat-strip">{children}</div>;
}

export function Card({ title, subtitle, children, actions, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p className="muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Empty({ title, children }) {
  return (
    <div className="empty">
      <svg className="empty-art" viewBox="0 0 120 40" aria-hidden="true">
        <path d="M2 20h30l8-14 12 28 10-22 8 8h48" fill="none" stroke="currentColor" strokeWidth="2.2"
          strokeLinecap="round" strokeLinejoin="round" pathLength="1" />
      </svg>
      <strong>{title}</strong>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Skeleton({ lines = 3, height = 14 }) {
  return (
    <div className="skeleton-wrap" aria-busy="true" aria-label="Loading">
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton" style={{ height, width: `${100 - i * 12}%`, animationDelay: `${i * 120}ms` }} />
      ))}
    </div>
  );
}

export function Status({ loading, error, lines }) {
  if (error) {
    return (
      <div className="notice notice-error" role="alert">
        <strong>Can’t reach the API at {API_BASE}.</strong>{" "}
        {error.message}. Start the backend from the project root with <code>uvicorn backend.app.main:app</code>.
      </div>
    );
  }
  if (loading) return <Skeleton lines={lines} />;
  return null;
}

export function MeterLink({ id }) {
  return <Link className="meter-link" to={`/meters/${encodeURIComponent(id)}`}>{id}</Link>;
}

export function SimulationBanner({ onRefresh }) {
  const [simState, setSimState] = useState({ running: false, current: 0, total: 0, alerts_raised: 0 });
  const [busy, setBusy] = useState(false);
  const [limit, setLimit] = useState(30);

  const checkStatus = async () => {
    try {
      const s = await api.simulationStatus();
      setSimState(s);
      if (s.running && onRefresh) {
        onRefresh();
      }
    } catch (e) {}
  };

  useEffect(() => {
    checkStatus();
    const timer = setInterval(checkStatus, simState.running ? 1500 : 4000);
    return () => clearInterval(timer);
  }, [simState.running]);

  const handleStart = async () => {
    setBusy(true);
    try {
      const s = await api.startSimulation(limit, 1.0);
      setSimState(s);
      if (onRefresh) onRefresh();
    } catch (e) {
      alert("Failed to start inspection: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      const s = await api.stopSimulation();
      setSimState(s);
      if (onRefresh) onRefresh();
    } catch (e) {
      console.warn(e);
    } finally {
      setBusy(false);
    }
  };

  const handleReset = async () => {
    if (!window.confirm("Reset and clear all inspected meter data?")) return;
    setBusy(true);
    try {
      const s = await api.resetSimulation();
      setSimState(s);
      if (onRefresh) onRefresh();
    } catch (e) {
      alert("Failed to reset: " + e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="notice"
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 14,
        padding: "14px 18px",
      }}
    >
      <div style={{ flex: "1 1 280px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <strong style={{ fontSize: 14.5 }}>Simulated Real-Time Inspection</strong>
          {simState.running ? (
            <span className="badge badge-suspicious" style={{ padding: "2px 8px", fontSize: 11 }}>
              STREAMING ({simState.current}/{simState.total}) · {simState.alerts_raised} Alerts
            </span>
          ) : (
            <span className="badge badge-none" style={{ padding: "2px 8px", fontSize: 11 }}>
              READY
            </span>
          )}
        </div>
        <div className="muted" style={{ fontSize: 13 }}>
          {simState.running
            ? simState.message || `Inspecting meter ${simState.last_meter_id || ""}...`
            : "Stream smart-meter sequences into the deployed model and observe real-time risk classification & alerts."}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        {!simState.running ? (
          <>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              disabled={busy}
              style={{
                font: "inherit",
                fontSize: 12.5,
                background: "var(--glass)",
                color: "var(--text-1)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: "5px 8px",
                cursor: "pointer",
              }}
            >
              <option value={15}>15 meters</option>
              <option value={30}>30 meters</option>
              <option value={50}>50 meters</option>
              <option value={100}>100 meters</option>
            </select>
            <button
              disabled={busy}
              onClick={handleStart}
              style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              <span>▶</span> Start Inspection
            </button>
            <Link
              to="/test"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                padding: "5px 11px",
                borderRadius: 7,
                fontSize: 13,
                fontWeight: 500,
                textDecoration: "none",
                background: "var(--glass)",
                border: "1px solid var(--border)",
                color: "var(--text-1)",
              }}
              title="Test custom or preset meter readings"
            >
              <span>🧪</span> Test Meter
            </Link>
            <button className="ghost" disabled={busy} onClick={handleReset} title="Clear database and start fresh">
              🔄 Reset Demo
            </button>
          </>
        ) : (
          <button
            onClick={handleStop}
            disabled={busy}
            style={{
              background: "var(--series-2)",
              borderColor: "var(--series-2)",
              color: "#fff",
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>⏹</span> Stop Inspection
          </button>
        )}
      </div>
    </div>
  );
}

export function ChartTooltip({ active, payload, label, labelFormatter, valueFormatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="muted">{labelFormatter ? labelFormatter(label, payload) : label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="tooltip-row">
          <span className="swatch" style={{ background: p.color || p.fill }} />
          <span>{p.name}</span>
          <strong>{valueFormatter ? valueFormatter(p.value) : p.value}</strong>
        </div>
      ))}
    </div>
  );
}

/* ---------- histogram of latest risk scores, with the alert threshold marked ----------
   Bars grow from the baseline on load and glide to new heights when live data changes. */
export function RiskHistogram({ bins, threshold }) {
  const mounted = useMounted();
  const W = 560, H = 250, L = 34, R = 10, T = 22, B = 28;
  const plotH = H - T - B;
  const max = Math.max(1, ...bins.map((b) => b.count));
  const step = max <= 4 ? 1 : max <= 10 ? 2 : max <= 20 ? 5 : max <= 50 ? 10 : max <= 100 ? 20 : 50;
  const top = Math.ceil(max / step) * step;
  const sx = (v) => L + v * (W - L - R);
  const sy = (c) => T + (1 - c / top) * plotH;
  const yTicks = Array.from({ length: top / step + 1 }, (_, i) => i * step);
  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];

  return (
    <svg className="histogram" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label="Histogram of latest risk scores, one bar per 0.1 interval">
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="h-grid" />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end" className="h-tick">{t}</text>
        </g>
      ))}
      {threshold != null && (
        <rect x={sx(threshold)} y={T} width={sx(1) - sx(threshold)} height={plotH} className="h-zone" />
      )}
      {bins.map((b, i) => {
        const frac = mounted ? b.count / top : 0;
        const delay = `${200 + i * 70}ms`;
        return (
          <g key={b.from}>
            <rect x={sx(b.from) + 2} y={T} width={sx(b.to) - sx(b.from) - 4} height={plotH} rx="3" className="h-bar"
              style={{ transform: `scaleY(${frac})`, transitionDelay: delay }}>
              <title>{`Risk ${b.from.toFixed(1)}–${b.to.toFixed(1)}: ${b.count} meter${b.count === 1 ? "" : "s"}`}</title>
            </rect>
            {b.count > 0 && (
              <text x={(sx(b.from) + sx(b.to)) / 2} y={sy(0) - 6} textAnchor="middle" className="h-count"
                style={{ transform: `translateY(${-frac * plotH}px)`, opacity: mounted ? 1 : 0, transitionDelay: delay }}>
                {b.count}
              </text>
            )}
          </g>
        );
      })}
      {xTicks.map((t) => (
        <text key={t} x={sx(t)} y={H - 8} textAnchor="middle" className="h-tick">{t.toFixed(1)}</text>
      ))}
      {threshold != null && (
        <g className="h-threshold-group">
          <line x1={sx(threshold)} x2={sx(threshold)} y1={T - 6} y2={H - B} className="h-threshold" />
          <text x={sx(threshold) - 6} y={T - 8} textAnchor="end" className="h-threshold-label">
            Alert threshold {threshold.toFixed(3)}
          </text>
        </g>
      )}
    </svg>
  );
}
