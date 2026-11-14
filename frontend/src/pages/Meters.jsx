import { useState } from "react";
import { api } from "../api";
import { Card, MeterLink, RiskBadge, Status } from "../components/ui";
import { fmtScore, fmtTime, useApi } from "../hooks";

const FILTERS = [["", "All"], ["SUSPICIOUS", "Suspicious"], ["NORMAL", "Normal"]];

export default function Meters() {
  const [status, setStatus] = useState("");
  const { data, loading, error } = useApi(() => api.meters(status), [status], 10000);

  return (
    <>
      <h1>Meters</h1>
      <Card
        title="Monitored meters"
        subtitle="Sorted by latest risk score, highest first."
        actions={
          <div className="segmented" role="group" aria-label="Filter by prediction">
            {FILTERS.map(([value, label]) => (
              <button key={label} className={status === value ? "active" : ""} onClick={() => setStatus(value)}>
                {label}
              </button>
            ))}
          </div>
        }
      >
        <Status loading={loading && !data} error={error} />
        {data && data.length === 0 && <p className="muted">No meters match.</p>}
        {data && data.length > 0 && (
          <table>
            <thead>
              <tr><th>Meter</th><th>Latest result</th><th className="num">Risk score</th><th className="num">Open alerts</th><th>Last scored</th></tr>
            </thead>
            <tbody>
              {data.map((m) => (
                <tr key={m.meter_id}>
                  <td><MeterLink id={m.meter_id} /></td>
                  <td><RiskBadge prediction={m.latest_prediction} /></td>
                  <td className="num">{fmtScore(m.latest_risk_score)}</td>
                  <td className="num">{m.open_alerts}</td>
                  <td className="muted">{fmtTime(m.last_scored_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}
