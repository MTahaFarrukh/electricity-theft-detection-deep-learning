# Data

Datasets are **not committed** to Git because of their size and licence terms. Download them yourself and place them here.

```
data/
├── raw/         # original, unmodified downloaded files (read-only by convention)
└── processed/   # outputs of the preprocessing pipeline (regenerable)
```

## Intended dataset: SGCC electricity theft dataset

- **Source:** State Grid Corporation of China (SGCC). It was released alongside
  Zheng et al., *"Wide and Deep Convolutional Neural Networks for Electricity-Theft
  Detection to Secure Smart Grids"*, IEEE Transactions on Industrial Informatics, 2018.
- **Where to get it:** the paper's authors' GitHub repository (`henryRDlab/ElectricityTheftDetection`),
  which distributes it as a split zip archive, or one of the Kaggle mirrors (search Kaggle for
  "SGCC electricity theft"). Check the licence/terms on the page you download from and cite the
  paper in your report.
- **Put it at:** `data/raw/` (for example `data/raw/data.csv`). If the archive is split into parts,
  extract all parts together.

### What the literature says about it (NOT VERIFIED YET)

Papers that use the dataset describe it as about 42,000 customers with daily consumption from
2014-01-01 to 2016-10-31 (about 1,035 days). Each row is one customer, with a customer-ID column,
a binary label column (1 = theft-labelled) and one column per date. Roughly 8–9% of customers are
labelled as theft, and there are many missing values.

**None of this is assumed by the code.** The first step is to run the audit:

```bash
python scripts/audit_dataset.py data/raw/<file>.csv
```

That writes `reports/audit/dataset_audit.md` and `.json` with the actual structure. Preprocessing
decisions are made only after that report has been reviewed.

## Alternatives if SGCC cannot be obtained

| Dataset | Labels? | Notes |
|---|---|---|
| **Irish CER Smart Metering Trial** (ISSDA) | No theft labels | Real half-hourly data; requires an application to ISSDA. Theft labels would have to be *synthetically injected* (e.g. using the six attack functions of Jokar et al., 2016), and that must be stated clearly in the report. |
| **Kaggle "Electricity Theft Detection" mirrors of SGCC** | Yes | Same data as SGCC; check that the file matches the original. |
| **London Smart Meters (Low Carbon London)** | No theft labels | Same caveat as CER: labels would be synthetic. |

If only an unlabelled dataset is available, the project methodology changes: the injected attacks
have to be documented as synthetic, and the results are not comparable to papers that use real labels.
