import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { AlertStatus, Card, ChartTooltip, RiskBadge, StatTile, Status } from "../components/ui";
import { fmtScore, fmtTime, useApi } from "../hooks";

function addDays(isoDay, n) {
  const d = new Date(`${isoDay}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

export default function MeterDetail() {
  const { meterId } = useParams();
  const { data, loading, error } = useApi(() => api.meter(meterId), [meterId]);

  const series = useMemo(() => {
    if (!data?.first_day) return [];
    return data.readings.map((kwh, i) => ({ day: addDays(data.first_day, i), kwh }));
  }, [data]);
  const missing = series.filter((p) => p.kwh == null).length;

  return (
    <>
      <p><Link to="/meters">← All meters</Link></p>
      <h1>Meter {meterId}</h1>
      <Status loading={loading} error={error} />
      {data && (
        <>
          <div className="tiles">
            <StatTile label="Latest result" value={<RiskBadge prediction={data.latest_prediction} />}
              note={data.latest_prediction === "SUSPICIOUS" ? "Investigation recommended" : undefined} />
            <StatTile label="Latest risk score" value={fmtScore(data.latest_risk_score)} />
            <StatTile label="Open alerts" value={data.open_alerts} />
            <StatTile label="Missing days" value={`${missing} / ${series.length}`} />
          </div>

          <Card title="Daily consumption (kWh)" subtitle="As submitted. Gaps are missing readings.">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={series} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="day" minTickGap={60} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                <Tooltip cursor={{ stroke: "var(--axis)" }}
                  content={<ChartTooltip valueFormatter={(v) => (v == null ? "missing" : `${v.toFixed(2)} kWh`)} />} />
                <Line dataKey="kwh" name="Consumption" stroke="var(--series-1)" strokeWidth={1.5} dot={false}
                  connectNulls={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <div className="grid-2">
            <Card title="Prediction history">
              <table>
                <thead><tr><th>Scored</th><th>Result</th><th className="num">Risk</th><th className="num">Threshold</th><th>Model</th></tr></thead>
                <tbody>
                  {[...data.predictions].reverse().map((p) => (
                    <tr key={p.id}>
                      <td className="muted">{fmtTime(p.created_at)}</td>
                      <td><RiskBadge prediction={p.prediction} /></td>
                      <td className="num">{fmtScore(p.risk_score)}</td>
                      <td className="num">{fmtScore(p.threshold)}</td>
                      <td className="muted">{p.model_run}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="Alerts">
              {data.alerts.length === 0 ? <p className="muted">No alerts for this meter.</p> : (
                <table>
                  <thead><tr><th>Raised</th><th className="num">Risk</th><th>Status</th></tr></thead>
                  <tbody>
                    {data.alerts.map((a) => (
                      <tr key={a.id}>
                        <td className="muted">{fmtTime(a.created_at)}</td>
                        <td className="num">{fmtScore(a.risk_score)}</td>
                        <td><AlertStatus status={a.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </div>
        </>
      )}
    </>
  );
}
