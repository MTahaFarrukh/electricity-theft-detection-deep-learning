import { useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { Brush, CartesianGrid, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { AlertStatus, Card, ChartTooltip, Empty, PageHeader, RiskBadge, StatTile, Status } from "../components/ui";
import { RiskDial } from "../components/visuals";
import { fmtScore, fmtTime, useApi } from "../hooks";

function addDays(isoDay, n) {
  const d = new Date(`${isoDay}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

// Consecutive missing days -> [{from, to}] so gaps can be shaded.
function missingRuns(series) {
  const runs = [];
  let start = null;
  series.forEach((p, i) => {
    if (p.kwh == null && start === null) start = i;
    if (p.kwh != null && start !== null) { runs.push({ from: series[start].day, to: series[i - 1].day }); start = null; }
  });
  if (start !== null) runs.push({ from: series[start].day, to: series[series.length - 1].day });
  return runs;
}

export default function MeterDetail() {
  const { meterId } = useParams();
  const { data, loading, error } = useApi(() => api.meter(meterId), [meterId]);

  const series = useMemo(() => {
    if (!data?.first_day) return [];
    return data.readings.map((kwh, i) => ({ day: addDays(data.first_day, i), kwh }));
  }, [data]);
  const runs = useMemo(() => missingRuns(series), [series]);
  const missing = series.filter((p) => p.kwh == null).length;
  const last = data?.predictions?.[data.predictions.length - 1];
  const threshold = last?.threshold ?? null;
  const suspicious = data?.latest_prediction === "SUSPICIOUS";

  return (
    <>
      <p className="small"><Link to="/meters">← All meters</Link></p>
      <PageHeader title={`Meter ${meterId}`} aside={data && <RiskBadge prediction={data.latest_prediction} />} />
      <Status loading={loading} error={error} lines={4} />
      {data && (
        <>
          <div className="detail-top">
            <Card title="Latest risk score" className={suspicious ? "card-hot" : ""}>
              <RiskDial score={data.latest_risk_score} threshold={threshold} />
              <p className="dial-note">
                {suspicious ? "Investigation recommended. This is a risk indicator, not proof of theft."
                  : data.latest_prediction ? "Below the alert threshold." : ""}
              </p>
            </Card>
            <div className="detail-stats">
              <StatTile label="Open alerts" value={data.open_alerts} />
              <StatTile label="Missing days" value={missing} note={series.length ? `of ${series.length} (${((missing / series.length) * 100).toFixed(1)}%)` : undefined} />
              <StatTile label="Series start" value={data.first_day ?? "—"} />
              <StatTile label="Times scored" value={data.predictions.length} />
            </div>
          </div>

          <Card title="Daily consumption (kWh)" subtitle="As submitted. Shaded bands are runs of missing readings; drag the handles below to zoom.">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={series} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="day" minTickGap={70} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" width={40} />
                <Tooltip cursor={{ stroke: "var(--axis)" }}
                  content={<ChartTooltip valueFormatter={(v) => (v == null ? "missing" : `${v.toFixed(2)} kWh`)} />} />
                {runs.map((r) => (
                  <ReferenceArea key={r.from} x1={r.from} x2={r.to} fill="var(--muted)" fillOpacity={0.22} stroke="none" />
                ))}
                <Line dataKey="kwh" name="Consumption" stroke="var(--series-1)" strokeWidth={1.6} dot={false}
                  connectNulls={false} isAnimationActive animationDuration={1800} animationEasing="ease-out" />
                <Brush dataKey="day" height={26} stroke="var(--axis)" fill="var(--surface-2)" travellerWidth={9} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <div className="grid-2">
            <Card title="Prediction history">
              {data.predictions.length === 0 ? <Empty title="No predictions yet" /> : (
                <table>
                  <thead><tr><th>Scored</th><th>Result</th><th className="num">Risk</th><th className="num">Threshold</th><th>Model</th></tr></thead>
                  <tbody>
                    {[...data.predictions].reverse().map((p, i) => (
                      <tr key={p.id} className="row" style={{ "--i": i }}>
                        <td className="muted">{fmtTime(p.created_at)}</td>
                        <td><RiskBadge prediction={p.prediction} /></td>
                        <td className="num">{fmtScore(p.risk_score)}</td>
                        <td className="num">{fmtScore(p.threshold)}</td>
                        <td className="muted">{p.model_run}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
            <Card title="Alerts">
              {data.alerts.length === 0 ? <Empty title="No alerts for this meter" /> : (
                <table>
                  <thead><tr><th>Raised</th><th className="num">Risk</th><th>Status</th></tr></thead>
                  <tbody>
                    {data.alerts.map((a, i) => (
                      <tr key={a.id} className="row" style={{ "--i": i }}>
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
