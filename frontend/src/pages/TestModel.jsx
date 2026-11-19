import { useState, useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import { Card, ChartTooltip, PageHeader, RiskBadge, RiskMeter, StatTile, Status } from "../components/ui";
import { RiskDial } from "../components/visuals";
import { fmtScore } from "../hooks";
import { useModel } from "../model";

const N_DAYS = 1035;

// Generate realistic 1,035-day sequences for quick preset testing
function generatePreset(type) {
  const readings = [];
  const rng = Math.random;

  for (let i = 0; i < N_DAYS; i++) {
    // Normal weekly cycle (weekend slightly higher)
    const dayOfWeek = i % 7;
    const weekendBoost = dayOfWeek >= 5 ? 1.05 : 1.0;
    const season = Math.sin((i / 365) * 2 * Math.PI) * 1.2;

    let val = 0;
    let isMissing = false;

    if (type === "normal_household") {
      val = (5.5 + season + (rng() * 2 - 1) * 1.2) * weekendBoost;
      isMissing = rng() < 0.02;
    } else if (type === "theft_bypass") {
      if (i < 550) {
        val = (7.8 + season + (rng() * 2 - 1) * 1.5) * weekendBoost;
      } else {
        // Sudden 85% drop after day 550
        val = (1.1 + (rng() * 2 - 1) * 0.4) * weekendBoost;
      }
      isMissing = rng() < 0.04;
    } else if (type === "theft_tampering") {
      if (i < 450) {
        val = (6.5 + season + (rng() * 2 - 1) * 1.5) * weekendBoost;
        isMissing = rng() < 0.03;
      } else {
        // High variance, frequent drop to near-zero and lots of missing data
        val = rng() < 0.3 ? 0.2 : (2.5 + (rng() * 2 - 1) * 2.0) * weekendBoost;
        isMissing = rng() < 0.25;
      }
    } else if (type === "commercial_normal") {
      val = (22.0 + season * 2.5 + (rng() * 2 - 1) * 3.5) * weekendBoost;
      isMissing = rng() < 0.015;
    }

    readings.push(isMissing ? null : Math.max(0.05, Math.round(val * 100) / 100));
  }
  return readings;
}

export default function TestModel() {
  const { threshold: defaultThreshold } = useModel();
  const [tab, setTab] = useState("presets"); // "presets" | "custom" | "manual"
  const [presetKey, setPresetKey] = useState("theft_bypass");

  // Custom slider generator state
  const [baseLoad, setBaseLoad] = useState(6.5);
  const [hasDrop, setHasDrop] = useState(true);
  const [dropDay, setDropDay] = useState(550);
  const [dropSeverity, setDropSeverity] = useState(80); // percent drop
  const [missingPct, setMissingPct] = useState(3); // percent missing

  // Manual raw input state
  const [rawText, setRawText] = useState("");
  const [meterId, setMeterId] = useState("TEST-METER-01");
  const [saveToDb, setSaveToDb] = useState(true);

  // Current active readings
  const [readings, setReadings] = useState(() => generatePreset("theft_bypass"));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Switch preset
  const handleSelectPreset = (key) => {
    setPresetKey(key);
    const newReadings = generatePreset(key);
    setReadings(newReadings);
    setResult(null);
  };

  // Build readings from custom sliders
  const handleApplySliders = () => {
    const arr = [];
    const rng = Math.random;
    for (let i = 0; i < N_DAYS; i++) {
      const season = Math.sin((i / 365) * 2 * Math.PI) * (baseLoad * 0.15);
      let val = baseLoad + season + (rng() * 2 - 1) * (baseLoad * 0.2);
      if (hasDrop && i >= dropDay) {
        val *= 1 - dropSeverity / 100;
      }
      const isMiss = rng() < missingPct / 100;
      arr.push(isMiss ? null : Math.max(0.05, Math.round(val * 100) / 100));
    }
    setReadings(arr);
    setResult(null);
  };

  // Parse manual raw text input
  const handleApplyManual = () => {
    try {
      const clean = rawText
        .replace(/[[\]]/g, "")
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => (s.toLowerCase() === "null" || s.toLowerCase() === "nan" ? null : parseFloat(s)));

      if (clean.length === 0) throw new Error("Please enter numbers separated by comma or space.");
      if (clean.some((n) => n !== null && isNaN(n))) throw new Error("Contains invalid non-numeric values.");

      setReadings(clean);
      setResult(null);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  // Run model inference
  const handlePredict = async () => {
    if (!readings || readings.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      let finalReadings = [...readings];
      if (finalReadings.length < N_DAYS) {
        while (finalReadings.length < N_DAYS) {
          finalReadings = finalReadings.concat(readings);
        }
      }
      finalReadings = finalReadings.slice(0, N_DAYS);

      const payloadId = saveToDb && meterId.trim() ? meterId.trim() : null;
      const res = await api.predict(finalReadings, payloadId);
      setResult(res);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  // Chart data formatting (sample 200 points for chart performance if long)
  const chartData = useMemo(() => {
    if (!readings) return [];
    return readings.map((kwh, i) => ({
      day: i + 1,
      kwh: kwh,
    }));
  }, [readings]);

  // Statistics
  const stats = useMemo(() => {
    if (!readings) return { validCount: 0, missingCount: 0, avg: 0, preAvg: 0, postAvg: 0 };
    const valid = readings.filter((r) => r !== null);
    const missing = readings.length - valid.length;
    const avg = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;

    const mid = Math.floor(readings.length * 0.6);
    const preValid = readings.slice(0, mid).filter((r) => r !== null);
    const postValid = readings.slice(mid).filter((r) => r !== null);
    const preAvg = preValid.length ? preValid.reduce((a, b) => a + b, 0) / preValid.length : 0;
    const postAvg = postValid.length ? postValid.reduce((a, b) => a + b, 0) / postValid.length : 0;

    return {
      validCount: valid.length,
      missingCount: missing,
      missingRate: (missing / readings.length) * 100,
      avg: Math.round(avg * 100) / 100,
      preAvg: Math.round(preAvg * 100) / 100,
      postAvg: Math.round(postAvg * 100) / 100,
    };
  }, [readings]);

  const activeThreshold = result?.threshold ?? defaultThreshold ?? 0.764;
  const isSuspicious = result?.prediction === "SUSPICIOUS";

  return (
    <>
      <PageHeader
        title="Test & Inspect Model"
        sub="Simulate and test smart-meter consumption sequences against the deployed CNN-LSTM model in real time."
      />

      <div className="dash-grid" style={{ gridTemplateColumns: "1fr 1.15fr", alignItems: "start", gap: 20 }}>
        {/* Left Column: Input Builder */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card title="Input Readings Setup">
            {/* Tab navigation */}
            <div className="segmented" style={{ marginBottom: 14 }}>
              <button className={tab === "presets" ? "active" : ""} onClick={() => setTab("presets")}>
                ⭐ Presets
              </button>
              <button className={tab === "custom" ? "active" : ""} onClick={() => setTab("custom")}>
                ⚙️ Custom Sliders
              </button>
              <button className={tab === "manual" ? "active" : ""} onClick={() => setTab("manual")}>
                📝 Manual / Raw
              </button>
            </div>

            {/* Presets Tab */}
            {tab === "presets" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 8,
                    background: presetKey === "theft_bypass" ? "var(--hover)" : "transparent",
                    border: "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="preset"
                    checked={presetKey === "theft_bypass"}
                    onChange={() => handleSelectPreset("theft_bypass")}
                    style={{ marginTop: 3 }}
                  />
                  <div>
                    <strong style={{ color: "var(--series-2)" }}>⚡ Theft: Sudden Meter Bypass</strong>
                    <p className="muted small" style={{ margin: "2px 0 0" }}>
                      Steady ~7.8 kWh/day usage followed by an abrupt 85% drop at day 550.
                    </p>
                  </div>
                </label>

                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 8,
                    background: presetKey === "theft_tampering" ? "var(--hover)" : "transparent",
                    border: "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="preset"
                    checked={presetKey === "theft_tampering"}
                    onChange={() => handleSelectPreset("theft_tampering")}
                    style={{ marginTop: 3 }}
                  />
                  <div>
                    <strong style={{ color: "var(--series-2)" }}>⚠️ Theft: Erratic Tampering & Missingness</strong>
                    <p className="muted small" style={{ margin: "2px 0 0" }}>
                      Erratic fluctuations with 25%+ missing readings from meter disconnections.
                    </p>
                  </div>
                </label>

                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 8,
                    background: presetKey === "normal_household" ? "var(--hover)" : "transparent",
                    border: "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="preset"
                    checked={presetKey === "normal_household"}
                    onChange={() => handleSelectPreset("normal_household")}
                    style={{ marginTop: 3 }}
                  />
                  <div>
                    <strong style={{ color: "var(--accent)" }}>🏠 Normal Household</strong>
                    <p className="muted small" style={{ margin: "2px 0 0" }}>
                      Normal residential consumption ~5.5 kWh/day with natural weekend peaks.
                    </p>
                  </div>
                </label>

                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                    padding: "10px 12px",
                    borderRadius: 8,
                    background: presetKey === "commercial_normal" ? "var(--hover)" : "transparent",
                    border: "1px solid var(--border)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="preset"
                    checked={presetKey === "commercial_normal"}
                    onChange={() => handleSelectPreset("commercial_normal")}
                    style={{ marginTop: 3 }}
                  />
                  <div>
                    <strong style={{ color: "var(--accent)" }}>🏢 Commercial High Load (Normal)</strong>
                    <p className="muted small" style={{ margin: "2px 0 0" }}>
                      High steady consumption ~22 kWh/day. High volume but clean normal cyclical profile.
                    </p>
                  </div>
                </label>
              </div>
            )}

            {/* Custom Sliders Tab */}
            {tab === "custom" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span className="small">Base Consumption:</span>
                    <strong>{baseLoad} kWh/day</strong>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="25"
                    step="0.5"
                    value={baseLoad}
                    onChange={(e) => setBaseLoad(parseFloat(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>

                <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", marginBottom: 8 }}>
                    <input type="checkbox" checked={hasDrop} onChange={(e) => setHasDrop(e.target.checked)} />
                    <strong>Inject Consumption Drop (Theft Simulation)</strong>
                  </label>

                  {hasDrop && (
                    <>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span className="small">Drop Severity:</span>
                        <strong style={{ color: "var(--series-2)" }}>-{dropSeverity}%</strong>
                      </div>
                      <input
                        type="range"
                        min="20"
                        max="95"
                        value={dropSeverity}
                        onChange={(e) => setDropSeverity(parseInt(e.target.value))}
                        style={{ width: "100%", marginBottom: 10 }}
                      />

                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span className="small">Drop Occurs At Day:</span>
                        <strong>Day {dropDay}</strong>
                      </div>
                      <input
                        type="range"
                        min="100"
                        max="900"
                        step="25"
                        value={dropDay}
                        onChange={(e) => setDropDay(parseInt(e.target.value))}
                        style={{ width: "100%" }}
                      />
                    </>
                  )}
                </div>

                <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span className="small">Missing Readings:</span>
                    <strong>{missingPct}%</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="35"
                    value={missingPct}
                    onChange={(e) => setMissingPct(parseInt(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>

                <button onClick={handleApplySliders} style={{ marginTop: 6 }}>
                  Apply Parameters to Readings
                </button>
              </div>
            )}

            {/* Manual Raw Input Tab */}
            {tab === "manual" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <p className="muted small">
                  Paste comma or space-separated values (e.g. <code>5.4, 6.1, 4.9, 1.2, 0.8</code>). Shorter sequences
                  are seamlessly auto-tiled to match model window.
                </p>
                <textarea
                  rows={5}
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  placeholder="e.g. 5.2, 4.8, 6.1, 5.9, null, 1.2, 0.9..."
                  style={{
                    fontFamily: "monospace",
                    fontSize: 12,
                    background: "var(--glass)",
                    color: "var(--text-1)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    padding: 8,
                    resize: "vertical",
                  }}
                />
                <button onClick={handleApplyManual}>Load Input Values</button>
              </div>
            )}

            {/* Meter Registration & Predict Action */}
            <div style={{ borderTop: "1px solid var(--border)", marginTop: 16, paddingTop: 14 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", marginBottom: 10 }}>
                <input type="checkbox" checked={saveToDb} onChange={(e) => setSaveToDb(e.target.checked)} />
                <span className="small">Save result to Monitored Meters database</span>
              </label>

              {saveToDb && (
                <div style={{ marginBottom: 12 }}>
                  <input
                    type="text"
                    value={meterId}
                    onChange={(e) => setMeterId(e.target.value)}
                    placeholder="Enter Meter ID"
                    style={{
                      width: "100%",
                      padding: "6px 10px",
                      background: "var(--glass)",
                      color: "var(--text-1)",
                      border: "1px solid var(--border)",
                      borderRadius: 6,
                      fontSize: 13,
                    }}
                  />
                </div>
              )}

              <button
                disabled={loading || !readings.length}
                onClick={handlePredict}
                style={{
                  width: "100%",
                  padding: "10px",
                  fontSize: 14,
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                }}
              >
                {loading ? "Analyzing..." : "⚡ Run Model Prediction"}
              </button>
            </div>
          </Card>
        </div>

        {/* Right Column: Prediction Results & Consumption Profile */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {error && <div className="notice notice-error">{error}</div>}

          {/* Model Prediction Outcome */}
          {result ? (
            <Card
              title="Prediction Result"
              className={isSuspicious ? "card-hot" : ""}
              actions={<RiskBadge prediction={result.prediction} />}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap", marginBottom: 12 }}>
                <div style={{ flex: "0 0 160px" }}>
                  <RiskDial score={result.risk_score} threshold={activeThreshold} />
                </div>
                <div style={{ flex: "1 1 200px" }}>
                  <div style={{ fontSize: 13, color: "var(--text-2)", marginBottom: 4 }}>Model Decision:</div>
                  <h3
                    style={{
                      margin: 0,
                      fontSize: 22,
                      color: isSuspicious ? "var(--series-2)" : "var(--accent)",
                    }}
                  >
                    {result.prediction}
                  </h3>
                  <p className="small muted" style={{ marginTop: 6, lineHeight: 1.4 }}>
                    {result.recommendation}
                  </p>
                  {result.alert_raised && (
                    <span className="badge badge-suspicious" style={{ marginTop: 6 }}>
                      🚨 Alert Raised & Logged
                    </span>
                  )}
                </div>
              </div>

              <div className="detail-stats" style={{ marginTop: 12 }}>
                <StatTile label="Risk Score" value={fmtScore(result.risk_score)} />
                <StatTile label="Alert Threshold" value={fmtScore(activeThreshold)} />
                <StatTile label="Pre vs Post Avg" value={`${stats.preAvg} → ${stats.postAvg}`} note="kWh/day" />
                <StatTile label="Missing Readings" value={`${stats.missingRate.toFixed(1)}%`} note={`${stats.missingCount} days`} />
              </div>
            </Card>
          ) : (
            <Card title="Model Ready">
              <div className="empty" style={{ padding: "28px 16px" }}>
                <strong>No prediction yet</strong>
                <p>Select a preset or adjust the sliders, then click "⚡ Run Model Prediction" above.</p>
              </div>
            </Card>
          )}

          {/* Consumption Time Series Chart */}
          <Card
            title="Consumption Profile"
            subtitle={`${readings.length} daily readings. Visualizing consumption behavior across time.`}
          >
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 4, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--grid)" />
                <XAxis
                  dataKey="day"
                  tick={{ fill: "var(--muted)", fontSize: 11 }}
                  stroke="var(--axis)"
                  tickFormatter={(d) => `Day ${d}`}
                />
                <YAxis
                  domain={[0, "auto"]}
                  tick={{ fill: "var(--muted)", fontSize: 11 }}
                  stroke="var(--axis)"
                  unit=" kWh"
                />
                <Tooltip content={<ChartTooltip valueFormatter={(v) => `${v} kWh`} />} />
                <Line
                  type="monotone"
                  dataKey="kwh"
                  stroke={isSuspicious ? "var(--series-2)" : "var(--accent)"}
                  strokeWidth={1.8}
                  dot={false}
                  name="Consumption"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </div>
    </>
  );
}
