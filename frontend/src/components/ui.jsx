import { Link } from "react-router-dom";

// Status is never colour-alone: icon + label + colour.
export function RiskBadge({ prediction }) {
  if (!prediction) return <span className="badge badge-none">Not scored</span>;
  const suspicious = prediction === "SUSPICIOUS";
  return (
    <span className={`badge ${suspicious ? "badge-critical" : "badge-good"}`}>
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

export function StatTile({ label, value, note }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value ?? "—"}</div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

export function Card({ title, subtitle, children, actions }) {
  return (
    <section className="card">
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

export function Status({ loading, error }) {
  if (error) return <div className="notice notice-error">Could not reach the API: {error.message}</div>;
  if (loading) return <div className="muted">Loading…</div>;
  return null;
}

export function MeterLink({ id }) {
  return <Link to={`/meters/${encodeURIComponent(id)}`}>{id}</Link>;
}

export function SimulationBanner() {
  return (
    <div className="notice">
      <strong>Simulated Real-Time Monitoring.</strong> Data is replayed from the held-out test set by{" "}
      <code>scripts/simulate_monitoring.py</code>; this system is not connected to real meters. A suspicious result is a
      risk indicator for prioritising inspections, not proof of theft.
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
