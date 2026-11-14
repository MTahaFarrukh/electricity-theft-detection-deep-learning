import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { AlertStatus, Card, ChartTooltip, MeterLink, SimulationBanner, StatTile, Status } from "../components/ui";
import { fmtScore, fmtTime, useApi } from "../hooks";

export default function Dashboard() {
  const { data, loading, error } = useApi(api.summary, [], 5000); // live while the simulator runs
  const model = data?.model;
  const histogram = (data?.risk_histogram || []).map((b) => ({ ...b, bin: `${b.from.toFixed(1)}–${b.to.toFixed(1)}` }));

  return (
    <>
      <h1>Dashboard</h1>
      <SimulationBanner />
      <Status loading={loading && !data} error={error} />
      {data && (
        <>
          <div className="tiles">
            <StatTile label="Meters monitored" value={data.n_meters.toLocaleString()} />
            <StatTile label="Suspicious (latest score)" value={data.n_suspicious.toLocaleString()}
              note="Investigation recommended" />
            <StatTile label="Normal (latest score)" value={data.n_normal.toLocaleString()} />
            <StatTile label="Open alerts" value={data.alerts_by_status.OPEN.toLocaleString()}
              note={`${data.alerts_by_status.REVIEWED} reviewed · ${data.alerts_by_status.DISMISSED} dismissed`} />
          </div>
          <p className="muted small">
            Model: {model?.model_loaded ? `${model.model} · threshold ${fmtScore(model.threshold)}` : "not loaded"} ·
            refreshes every 5 s
          </p>

          <div className="grid-2">
            <Card title="Risk score distribution" subtitle="Latest score per meter. Scores at or above the threshold raise an alert.">
              {data.n_meters === 0 ? (
                <p className="muted">No meters scored yet. Start the simulator to replay test meters.</p>
              ) : (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={histogram} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                    <CartesianGrid vertical={false} stroke="var(--grid)" />
                    <XAxis dataKey="bin" tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                    <YAxis allowDecimals={false} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
                    <Tooltip cursor={{ fill: "var(--hover)" }}
                      content={<ChartTooltip labelFormatter={(l) => `Risk score ${l}`} />} />
                    <Bar dataKey="count" name="Meters" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={40} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Card>

            <Card title="Recent alerts">
              {data.recent_alerts.length === 0 ? (
                <p className="muted">No alerts yet.</p>
              ) : (
                <table>
                  <thead><tr><th>Meter</th><th className="num">Risk</th><th>Status</th><th>Raised</th></tr></thead>
                  <tbody>
                    {data.recent_alerts.map((a) => (
                      <tr key={a.id}>
                        <td><MeterLink id={a.meter_id} /></td>
                        <td className="num">{fmtScore(a.risk_score)}</td>
                        <td><AlertStatus status={a.status} /></td>
                        <td className="muted">{fmtTime(a.created_at)}</td>
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
