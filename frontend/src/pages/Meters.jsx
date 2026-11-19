import { useMemo, useState } from "react";
import { api } from "../api";
import { Card, Empty, MeterLink, PageHeader, RiskBadge, RiskMeter, Status } from "../components/ui";
import { fmtTime, useApi } from "../hooks";
import { useModel } from "../model";

const FILTERS = [["", "All"], ["SUSPICIOUS", "Suspicious"], ["NORMAL", "Normal"]];

export default function Meters() {
  const [status, setStatus] = useState("");
  const [query, setQuery] = useState("");
  const { threshold } = useModel();
  const { data, loading, error } = useApi(() => api.meters(status), [status], 10000);
  const rows = useMemo(() => (data || []).filter((m) => m.meter_id.includes(query.trim())), [data, query]);

  return (
    <>
      <PageHeader title="Meters" sub="Every scored meter, highest latest risk score first." />
      <Card
        title="Monitored meters"
        subtitle={data ? `Showing ${rows.length.toLocaleString()} of ${data.length.toLocaleString()}. The tick on each bar marks the alert threshold.` : undefined}
        actions={
          <div className="toolbar">
            <input type="search" value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="Search meter ID" aria-label="Search meter ID" />
            <div className="segmented" role="group" aria-label="Filter by prediction">
              {FILTERS.map(([value, label]) => (
                <button key={label} className={status === value ? "active" : ""} aria-pressed={status === value}
                  onClick={() => setStatus(value)}>{label}</button>
              ))}
            </div>
          </div>
        }
      >
        <Status loading={loading && !data} error={error} lines={5} />
        {data && rows.length === 0 && (
          <Empty title="No meters match">
            {data.length === 0 ? "No meters have been scored yet. Start the simulator to replay test meters." : "Try another meter ID or filter."}
          </Empty>
        )}
        {rows.length > 0 && (
          <table>
            <thead>
              <tr><th>Meter</th><th>Latest result</th><th>Risk score</th><th className="num">Open alerts</th><th>Last scored</th></tr>
            </thead>
            <tbody>
              {rows.map((m, i) => (
                <tr key={m.meter_id} className="row" style={{ "--i": Math.min(i, 14) }}>
                  <td><MeterLink id={m.meter_id} /></td>
                  <td><RiskBadge prediction={m.latest_prediction} /></td>
                  <td><RiskMeter score={m.latest_risk_score} threshold={threshold} /></td>
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
