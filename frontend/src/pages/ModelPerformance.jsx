import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ChartTooltip, StatTile, Status } from "../components/ui";
import { fmtScore, useApi } from "../hooks";

const NAMES = {
  missingness_only_lr: "Missingness-only LR", logistic_regression: "Logistic Regression",
  random_forest: "Random Forest", cnn: "1D-CNN", lstm: "LSTM", cnn_lstm: "CNN-LSTM",
};
const ORDER = Object.keys(NAMES);
const COLS = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"],
  ["pr_auc", "PR-AUC"], ["roc_auc", "ROC-AUC"], ["fpr", "FPR"]];

function MetricsTable({ rows }) {
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Model</th>{COLS.map(([k, l]) => <th key={k} className="num">{l}</th>)}</tr></thead>
        <tbody>
          {rows.map(({ name, m, highlight }) => (
            <tr key={name} className={highlight ? "highlight" : ""}>
              <td>{name}</td>
              {COLS.map(([k]) => <td key={k} className="num">{fmtScore(m[k])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPerformance() {
  const { data, loading, error } = useApi(api.modelMetrics);
  if (!data) return (<><h1>Model performance</h1><Status loading={loading} error={error} /></>);

  const deployed = data.deployed_model;
  const val = { ...(data.validation_baselines?.models || {}), ...(data.validation_deep?.models || {}) };
  const test = data.test;
  const deployedArch = deployed?.model;

  const valRows = ORDER.filter((k) => val[k]).map((k) => ({ name: NAMES[k], m: val[k], highlight: k === deployedArch }));
  const testRows = test ? ORDER.filter((k) => test.runs[k]).map((k) => ({
    name: NAMES[k], m: test.runs[k].test, highlight: k === deployedArch })) : [];
  const chart = ORDER.filter((k) => val[k]).map((k) => ({
    model: NAMES[k], validation: val[k].pr_auc, test: test?.runs[k]?.test.pr_auc ?? null }));
  const cm = test && deployed ? test.runs[deployed.run].test.confusion_matrix : null;

  return (
    <>
      <h1>Model performance</h1>
      <p className="muted">
        All numbers are stored evaluation results. Thresholds and model selection use the validation split only;
        the test split was used once, for the final evaluation.
      </p>

      {deployed ? (
        <div className="tiles">
          <StatTile label="Deployed model" value={NAMES[deployed.model]} note={`run ${deployed.run}`} />
          <StatTile label="Decision threshold" value={fmtScore(deployed.threshold)} note={deployed.threshold_rule} />
          <StatTile label="Test PR-AUC" value={test ? fmtScore(test.runs[deployed.run].test.pr_auc) : "—"}
            note={test ? `random classifier ≈ ${fmtScore(test.test_positive_rate)}` : "not evaluated yet"} />
          <StatTile label="Test recall / FPR" value={test ? `${fmtScore(test.runs[deployed.run].test.recall)} / ${fmtScore(test.runs[deployed.run].test.fpr)}` : "—"} />
        </div>
      ) : (
        <div className="notice">No deployed model yet: run <code>scripts/evaluate_test.py</code>.</div>
      )}

      <Card title="PR-AUC by model" subtitle="The primary metric for imbalanced data (8.5% theft-labelled). Higher is better.">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart} margin={{ top: 8, right: 8, bottom: 8, left: 0 }} barGap={2}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="model" tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" interval={0} />
            <YAxis domain={[0, 1]} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--axis)" />
            <Tooltip cursor={{ fill: "var(--hover)" }} content={<ChartTooltip valueFormatter={fmtScore} />} />
            <Legend wrapperStyle={{ color: "var(--text-2)", fontSize: 12 }} />
            <Bar dataKey="validation" name="Validation" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={28} />
            {test && <Bar dataKey="test" name="Test" fill="var(--series-2)" radius={[4, 4, 0, 0]} maxBarSize={28} />}
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {test && (
        <Card title="Test results (final, one-time evaluation)"
          subtitle={`${test.n_test.toLocaleString()} held-out customers, ${(test.test_positive_rate * 100).toFixed(1)}% theft-labelled. Main run (seed 42) per model; the deployed model is highlighted.`}>
          <MetricsTable rows={testRows} />
        </Card>
      )}

      {test && (
        <div className="grid-2">
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
              <table className="cm">
                <thead><tr><th /><th className="num">Predicted normal</th><th className="num">Predicted suspicious</th></tr></thead>
                <tbody>
                  <tr><th>Normal-labelled</th><td className="num">{cm.tn.toLocaleString()}</td><td className="num">{cm.fp.toLocaleString()}</td></tr>
                  <tr><th>Theft-labelled</th><td className="num">{cm.fn.toLocaleString()}</td><td className="num">{cm.tp.toLocaleString()}</td></tr>
                </tbody>
              </table>
              <p className="muted small">
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
