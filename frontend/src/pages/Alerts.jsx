import { useState } from "react";
import { api } from "../api";
import { AlertStatus, Card, Empty, MeterLink, PageHeader, RiskMeter, Status } from "../components/ui";
import { fmtTime, useApi } from "../hooks";
import { useModel } from "../model";

const FILTERS = [["OPEN", "Open"], ["REVIEWED", "Reviewed"], ["DISMISSED", "Dismissed"], ["", "All"]];

export default function Alerts() {
  const [status, setStatus] = useState("OPEN");
  const [busy, setBusy] = useState(null);
  const [failure, setFailure] = useState(null);
  const [leaving, setLeaving] = useState(null);
  const { threshold } = useModel();
  const { data, loading, error, reload } = useApi(() => api.alerts(status), [status], 10000);
  const { data: summary, reload: reloadSummary } = useApi(api.summary, [], 10000);
  const counts = summary?.alerts_by_status;
  const total = counts ? counts.OPEN + counts.REVIEWED + counts.DISMISSED : null;

  const update = async (id, next) => {
    setBusy(id);
    setFailure(null);
    try {
      await api.updateAlert(id, next);
      reloadSummary();
      if (status !== "" && status !== next) {      // the row leaves this filtered view: slide it out first
        setLeaving(id);
        await new Promise((resolve) => setTimeout(resolve, 380));
        setLeaving(null);
      }
      reload();
    } catch (e) {
      setFailure(`Could not update alert ${id}: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };
  const countFor = (value) => (counts ? (value === "" ? total : counts[value]) : null);

  return (
    <>
      <PageHeader title="Alerts"
        sub="An alert is raised once per submitted series when its risk score reaches the validated threshold. It means investigation is recommended, not that theft has occurred." />
      <Card
        title="Alert history"
        actions={
          <div className="segmented" role="group" aria-label="Filter by status">
            {FILTERS.map(([value, label]) => (
              <button key={label} className={status === value ? "active" : ""} aria-pressed={status === value}
                onClick={() => setStatus(value)}>
                {label}{countFor(value) != null && <span className="count">{countFor(value)}</span>}
              </button>
            ))}
          </div>
        }
      >
        {failure && <div className="notice notice-error" role="alert">{failure}</div>}
        <Status loading={loading && !data} error={error} lines={5} />
        {data && data.length === 0 && (
          <Empty title={status === "OPEN" ? "No open alerts" : "No alerts in this view"}>
            {status === "OPEN" ? "Everything raised so far has been reviewed or dismissed." : "Pick another status to see more."}
          </Empty>
        )}
        {data && data.length > 0 && (
          <table>
            <thead><tr><th>Raised</th><th>Meter</th><th>Risk score</th><th>Status</th><th>Action</th></tr></thead>
            <tbody>
              {data.map((a, i) => (
                <tr key={a.id} className={`row ${leaving === a.id ? "leaving" : ""}`} style={{ "--i": Math.min(i, 14) }}>
                  <td className="muted">{fmtTime(a.created_at)}</td>
                  <td><MeterLink id={a.meter_id} /></td>
                  <td><RiskMeter score={a.risk_score} threshold={threshold} /></td>
                  <td><AlertStatus status={a.status} /></td>
                  <td>
                    <div className="actions">
                      {a.status !== "REVIEWED" && (
                        <button disabled={busy === a.id} onClick={() => update(a.id, "REVIEWED")}>Mark reviewed</button>
                      )}
                      {a.status !== "DISMISSED" && (
                        <button className="ghost" disabled={busy === a.id} onClick={() => update(a.id, "DISMISSED")}>Dismiss</button>
                      )}
                    </div>
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
