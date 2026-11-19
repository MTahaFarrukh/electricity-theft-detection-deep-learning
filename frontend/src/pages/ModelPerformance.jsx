import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ChartTooltip, PageHeader, StatStrip, StatTile, Status } from "../components/ui";
import { CountUp } from "../motion";
import { fmtScore, useApi } from "../hooks";

const NAMES = {
  missingness_only_lr: "Missingness-only LR", logistic_regression: "Logistic Regression",
  random_forest: "Random Forest", cnn: "1D-CNN", lstm: "LSTM", cnn_lstm: "CNN-LSTM",
};
const ORDER = Object.keys(NAMES);
const COLS = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"],
  ["pr_auc", "PR-AUC"], ["roc_auc", "ROC-AUC"], ["fpr", "FPR"]];
const LOWER_IS_BETTER = new Set(["fpr"]);

// Bold the best value in each column (lowest for FPR, highest otherwise).
function bestByColumn(rows) {
  const best = {};
  COLS.forEach(([k]) => {
    const vals = rows.map((r) => r.m[k]).filter((v) => v != null);
    if (vals.length) best[k] = LOWER_IS_BETTER.has(k) ? Math.min(...vals) : Math.max(...vals);
  });
  return best;
}

// Rows are shares of each true class; blue = correct, orange = wrong.
function ConfusionMatrix({ cm }) {
  const cells = [
    { key: "tn", label: "Normal-labelled", pred: "predicted normal", ok: true, v: cm.tn, row: cm.tn + cm.fp },
    { key: "fp", label: "Normal-labelled", pred: "predicted suspicious", ok: false, v: cm.fp, row: cm.tn + cm.fp },
    { key: "fn", label: "Theft-labelled", pred: "predicted normal", ok: false, v: cm.fn, row: cm.fn + cm.tp },
    { key: "tp", label: "Theft-labelled", pred: "predicted suspicious", ok: true, v: cm.tp, row: cm.fn + cm.tp },
  ];
  return (
    <div className="cm-grid" role="table" aria-label="Confusion matrix">
      <span className="cm-corner" />
      <span className="cm-head">Predicted normal</span>
      <span className="cm-head">Predicted suspicious</span>
      {[cells.slice(0, 2), cells.slice(2)].map((row) => (
        <div key={row[0].key} className="cm-row" role="row">
          <span className="cm-side">{row[0].label}</span>
          {row.map((c, i) => (
            <div key={c.key} role="cell" className={`cm-cell ${c.ok ? "ok" : "bad"}`}
              style={{ "--heat": c.row ? c.v / c.row : 0, animationDelay: `${300 + i * 120}ms` }}>
              <strong><CountUp value={c.v} /></strong>
              <span>{c.row ? `${((c.v / c.row) * 100).toFixed(1)}% of row` : ""}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function MetricsTable({ rows }) {
  const best = bestByColumn(rows);
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Model</th>{COLS.map(([k, l]) => <th key={k} className="num">{l}</th>)}</tr></thead>
        <tbody>
          {rows.map(({ name, m, highlight }, i) => (
            <tr key={name} className={`row ${highlight ? "highlight" : ""}`} style={{ "--i": i }}>
              <td>{name}{highlight && <span className="tag">Deployed</span>}</td>
              {COLS.map(([k]) => <td key={k} className={`num ${m[k] === best[k] ? "best" : ""}`}>{fmtScore(m[k])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPerformance() {
  const { data, loading, error } = useApi(api.modelMetrics);
  if (!data) return (<><PageHeader title="Model performance" /><Status loading={loading} error={error} lines={5} /></>);

  const deployed = data.deployed_model;
  const val = { ...(data.validation_baselines?.models || {}), ...(data.validation_deep?.models || {}) };
  const test = data.test;
  const deployedArch = deployed?.model;
  const deployedTest = test && deployed ? test.runs[deployed.run]?.test : null;

  const valRows = ORDER.filter((k) => val[k]).map((k) => ({ name: NAMES[k], m: val[k], highlight: k === deployedArch }));
  const testRows = test ? ORDER.filter((k) => test.runs[k]).map((k) => ({
    name: NAMES[k], m: test.runs[k].test, highlight: k === deployedArch })) : [];
  const chart = ORDER.filter((k) => val[k]).map((k) => ({
    model: NAMES[k], validation: val[k].pr_auc, test: test?.runs[k]?.test.pr_auc ?? null }));
  const cm = test && deployed ? test.runs[deployed.run].test.confusion_matrix : null;

  return (
    <>
      <PageHeader title="Model performance"
        sub="All numbers are stored evaluation results. Thresholds and model selection use the validation split only; the test split was used once, for the final evaluation." />

      {deployed ? (
        <StatStrip>
          <StatTile label="Deployed model" value={NAMES[deployed.model]} note={`run ${deployed.run}`} />
          <StatTile label="Decision threshold" value={<CountUp value={deployed.threshold} decimals={3} />} note={deployed.threshold_rule} />
          <StatTile label="Test PR-AUC" value={deployedTest ? <CountUp value={deployedTest.pr_auc} decimals={3} /> : "—"}
            note={test ? `random classifier ≈ ${fmtScore(test.test_positive_rate)}` : "not evaluated yet"} />
          <StatTile label="Test recall / FPR" value={deployedTest ? <><CountUp value={deployedTest.recall} decimals={3} /> / <CountUp value={deployedTest.fpr} decimals={3} /></> : "—"} />
        </StatStrip>
      ) : (
        <div className="notice">No deployed model yet: run <code>scripts/evaluate_test.py</code>.</div>
      )}

      <Card title="PR-AUC by model"
        subtitle="The primary metric for imbalanced data (8.5% theft-labelled). Higher is better; the dashed line is what a random classifier scores.">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chart} margin={{ top: 12, right: 8, bottom: 8, left: 0 }} barGap={3}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="model" tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" interval={0} />
            <YAxis domain={[0, 1]} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" width={36} />
            <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip valueFormatter={fmtScore} />} />
            <Legend wrapperStyle={{ color: "var(--text-2)", fontSize: 12.5 }} />
            {test && (
              <ReferenceLine y={test.test_positive_rate} stroke="var(--muted)" strokeDasharray="5 4" />
            )}
            <Bar dataKey="validation" name="Validation" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={30} animationDuration={1300} />
            {test && <Bar dataKey="test" name="Test" fill="var(--series-2)" radius={[4, 4, 0, 0]} maxBarSize={30} animationDuration={1300} animationBegin={250} />}
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {test && (
        <Card title="Test results (final, one-time evaluation)"
          subtitle={`${test.n_test.toLocaleString()} held-out customers, ${(test.test_positive_rate * 100).toFixed(1)}% theft-labelled. Main run (seed 42) per model; the deployed model is highlighted and the best value in each column is bold.`}>
          <MetricsTable rows={testRows} />
        </Card>
      )}

      {test && (
        <div className="dash-grid">
          <Card title="Stability across seeds (test)" subtitle="Mean ± standard deviation over training runs.">
            <table>
              <thead><tr><th>Model</th><th className="num">Runs</th><th className="num">PR-AUC</th><th className="num">ROC-AUC</th><th className="num">F1</th></tr></thead>
              <tbody>
                {ORDER.filter((k) => test.by_architecture.some((r) => r.arch === k)).map((k) => {
                  const get = (metric) => test.by_architecture.find((r) => r.arch === k && r.metric === metric);
                  const show = (r) => `${fmtScore(r.mean)}${r.std != null ? ` ± ${r.std.toFixed(3)}` : ""}`;
                  return (
                    <tr key={k}><td>{NAMES[k]}</td><td className="num">{get("pr_auc").n_runs}</td>
                      <td className="num">{show(get("pr_auc"))}</td><td className="num">{show(get("roc_auc"))}</td>
                      <td className="num">{show(get("f1"))}</td></tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
          {cm && (
            <Card title={`Confusion matrix: ${NAMES[deployedArch]} (test)`}>
              <ConfusionMatrix cm={cm} />
              <p className="muted small" style={{ marginTop: 12 }}>
                Low-missingness subset (≤ {test.low_missing_subset.max_missing_fraction * 100}% missing,
                n = {test.low_missing_subset.n.toLocaleString()}): PR-AUC{" "}
                {fmtScore(test.runs[deployed.run].test_low_missing.pr_auc)}
                {" "}(random ≈ {fmtScore(test.low_missing_subset.positive_rate)}).
              </p>
            </Card>
          )}
        </div>
      )}

      <Card title="Validation results (model development)" subtitle="Used for hyper-parameters, early stopping and thresholds.">
        <MetricsTable rows={valRows} />
      </Card>
    </>
  );
}
