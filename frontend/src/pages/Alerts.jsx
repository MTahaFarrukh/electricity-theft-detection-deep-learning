import { useState } from "react";
import { api } from "../api";
import { AlertStatus, Card, MeterLink, Status } from "../components/ui";
import { fmtScore, fmtTime, useApi } from "../hooks";

const FILTERS = [["OPEN", "Open"], ["REVIEWED", "Reviewed"], ["DISMISSED", "Dismissed"], ["", "All"]];

export default function Alerts() {
  const [status, setStatus] = useState("OPEN");
  const [busy, setBusy] = useState(null);
  const { data, loading, error, reload } = useApi(() => api.alerts(status), [status], 10000);

  const update = async (id, next) => {
    setBusy(id);
    try {
      await api.updateAlert(id, next);
      reload();
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <h1>Alerts</h1>
      <Card
        title="Alert history"
        subtitle="An alert is raised once per submitted series when the risk score reaches the validated threshold. It means investigation is recommended."
        actions={
          <div className="segmented" role="group" aria-label="Filter by status">
            {FILTERS.map(([value, label]) => (
              <button key={label} className={status === value ? "active" : ""} onClick={() => setStatus(value)}>
                {label}
              </button>
            ))}
          </div>
        }
      >
        <Status loading={loading && !data} error={error} />
        {data && data.length === 0 && <p className="muted">No alerts.</p>}
        {data && data.length > 0 && (
          <table>
            <thead><tr><th>Raised</th><th>Meter</th><th className="num">Risk score</th><th>Status</th><th>Action</th></tr></thead>
            <tbody>
              {data.map((a) => (
                <tr key={a.id}>
                  <td className="muted">{fmtTime(a.created_at)}</td>
                  <td><MeterLink id={a.meter_id} /></td>
                  <td className="num">{fmtScore(a.risk_score)}</td>
                  <td><AlertStatus status={a.status} /></td>
                  <td className="actions">
                    {a.status !== "REVIEWED" && (
                      <button disabled={busy === a.id} onClick={() => update(a.id, "REVIEWED")}>Mark reviewed</button>
                    )}
                    {a.status !== "DISMISSED" && (
                      <button className="ghost" disabled={busy === a.id} onClick={() => update(a.id, "DISMISSED")}>Dismiss</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}
