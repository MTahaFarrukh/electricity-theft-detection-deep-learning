import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { AlertStatus, Card, Empty, MeterLink, RiskHistogram, RiskMeter, SimulationBanner, Status } from "../components/ui";
import { Hero } from "../components/visuals";
import { fmtScore, fmtTime, useApi } from "../hooks";

export default function Dashboard() {
  const { data, loading, error } = useApi(api.summary, [], 5000); // live while the simulator runs
  const { data: meters } = useApi(() => api.meters(), [], 5000);
  const threshold = data?.model?.model_loaded ? data.model.threshold : null;
  const top = (meters || []).slice(0, 8);

  // Highlight alerts that arrive while the page is open (the first load is not "new").
  const seen = useRef(null);
  const [fresh, setFresh] = useState(() => new Set());
  useEffect(() => {
    if (!data) return;
    const ids = data.recent_alerts.map((a) => a.id);
    if (seen.current === null) { seen.current = new Set(ids); return; }
    const added = ids.filter((id) => !seen.current.has(id));
    if (added.length) {
      added.forEach((id) => seen.current.add(id));
      setFresh(new Set(added));
      setTimeout(() => setFresh(new Set()), 4500);
    }
  }, [data]);

  return (
    <>
      <Status loading={loading && !data} error={error} lines={4} />
      {data && (
        <>
          <Hero data={data} />
          <SimulationBanner />

          <div className="dash-grid">
            <Card title="Risk score distribution"
              subtitle="Latest score per meter. Meters in the shaded zone, at or above the threshold, raise an alert.">
              {data.n_meters === 0 ? (
                <Empty title="No meters scored yet">
                  Start the simulator to replay held-out test meters: <code>python scripts/simulate_monitoring.py</code>
                </Empty>
              ) : (
                <RiskHistogram bins={data.risk_histogram} threshold={threshold} />
              )}
            </Card>

            <Card title="Highest-risk meters" subtitle="Top of the inspection queue. The tick marks the alert threshold.">
              {top.length === 0 ? <Empty title="Nothing to rank yet" /> : (
                <table>
                  <thead><tr><th>Meter</th><th>Risk score</th><th className="num">Open alerts</th></tr></thead>
                  <tbody>
                    {top.map((m, i) => (
                      <tr key={m.meter_id} className="row" style={{ "--i": i }}>
                        <td><MeterLink id={m.meter_id} /></td>
                        <td><RiskMeter score={m.latest_risk_score} threshold={threshold} /></td>
                        <td className="num">{m.open_alerts}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>

          <Card title="Recent alerts">
            {data.recent_alerts.length === 0 ? (
              <Empty title="No alerts yet">An alert appears when a meter’s risk score reaches the threshold.</Empty>
            ) : (
              <table>
                <thead><tr><th>Meter</th><th>Risk score</th><th>Status</th><th>Raised</th></tr></thead>
                <tbody>
                  {data.recent_alerts.map((a, i) => (
                    <tr key={a.id} className={`row ${fresh.has(a.id) ? "fresh" : ""}`} style={{ "--i": i }}>
                      <td><MeterLink id={a.meter_id} /></td>
                      <td><RiskMeter score={a.risk_score} threshold={threshold} /></td>
                      <td><AlertStatus status={a.status} /></td>
                      <td className="muted">{fmtTime(a.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
          <p className="muted small">
            Model: {data.model?.model_loaded ? `${data.model.model} · threshold ${fmtScore(data.model.threshold)}` : "not loaded"}
          </p>
        </>
      )}
    </>
  );
}
