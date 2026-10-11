# Data provenance and licensing

## Dataset

| | |
|---|---|
| **Name** | IBM Telco Customer Churn (sample dataset) |
| **File in this repository** | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Original publisher** | IBM (first distributed through the IBM Watson Analytics community) |
| **Where to get it** | `https://github.com/IBM/telco-customer-churn-on-icp4d` (file `data/Telco-Customer-Churn.csv`) |
| **File sha256** | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |
| **Retrieval date** | Not recorded in this repository |

The file sha256 above is the checksum of the bytes on disk. The pipeline separately records a hash of the
cleaned, canonical content as `hashes.input_data` in `reports/metrics.json` (see `src.data.load.data_hash`).

## What the MIT License covers

The `LICENSE` file (MIT) covers the source code, templates, tests, and documentation written for this
project. It does **not** cover the dataset or the files derived from it.

## Licensing status of the data

No explicit license for the dataset itself was found in IBM's published material at the time of writing.
The Apache-2.0 license that appears in IBM's repositories is stated for the code patterns in them, not for
the data. This project does not claim any license over the data. Before redistributing this repository or
the data files, check the terms with IBM or replace the data with your own.

## Files derived from the dataset

| File | How it is derived |
|---|---|
| `data/sample.csv` | Seeded, label-stratified subsample of the raw file (`src/data/sample.py`) |
| `tests/fixtures/sample_leaky.csv` | `data/sample.csv` plus a synthetic `leaky_feature` column (`src/features/leaky_fixture.py`) |
| `tests/fixtures/broken/*.csv` | Small slices of the data with deliberate defects, used to test validation |

Files under `reports/` and `docs/model_card.md` contain only aggregate statistics computed from the data,
not customer rows.

## Using your own data

Point `paths.raw` in `params.yaml` to a CSV with the same schema. See `docs/reference.md` for the data format.
