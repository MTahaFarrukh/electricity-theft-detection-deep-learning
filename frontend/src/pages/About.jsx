import { Card } from "../components/ui";

export default function About() {
  return (
    <>
      <h1>About &amp; methodology</h1>
      <Card title="What this system does">
        <p>
          It analyses a smart meter's daily consumption and gives a <strong>risk score</strong> between 0 and 1.
          Meters whose score reaches the validated threshold are marked <strong>Suspicious: investigation
          recommended</strong>. This helps a utility prioritise inspections. <strong>A prediction is not proof of
          theft.</strong>
        </p>
      </Card>
      <Card title="Data">
        <ul>
          <li>SGCC (State Grid Corporation of China) dataset: 42,372 customers, daily kWh from 2014-01-01 to 2016-10-31, 8.5% theft-labelled.</li>
          <li>Labels come from the utility's own inspections, so they can be noisy and undetected theft may be labelled normal.</li>
          <li>Customers are split 70/15/15 into train / validation / test <em>before</em> any preprocessing is fitted. No customer appears in two splits.</li>
        </ul>
      </Card>
      <Card title="Pipeline">
        <ol>
          <li>Readings are capped at the training 99.9th percentile. Gaps of 7 days or less are interpolated within each customer's own series. Values are log-scaled and standardised with training statistics.</li>
          <li>A missingness mask is kept as a second input channel.</li>
          <li>Models: Logistic Regression and Random Forest on hand-crafted features, then 1D-CNN, LSTM and the proposed hybrid CNN-LSTM on the full 1,035-day series.</li>
          <li>Class weights handle the imbalance. Early stopping, model selection and the decision threshold all use the validation split. The test split was used once.</li>
        </ol>
      </Card>
      <Card title="Simulated Real-Time Monitoring">
        <p>
          The live views are fed by a script that replays <strong>held-out test customers</strong> to the API at a fixed
          interval. The system is not connected to real meters. Each submitted series is scored once, and an alert is
          raised at most once per series.
        </p>
      </Card>
      <Card title="Limitations">
        <ul>
          <li>Missing-data patterns are associated with the label. Part of a model's signal can reflect data collection rather than behaviour. This is measured on the Model performance page (low-missingness subset).</li>
          <li>A model trained on one utility and region may not generalise to others.</li>
          <li>The risk score supports a human decision; it never replaces an inspection.</li>
        </ul>
      </Card>
    </>
  );
}
