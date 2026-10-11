"""Build and execute notebooks/01_data.ipynb.

Execution uses nbclient (a developer tool; not in requirements). The P2 tests verify the
notebook without nbclient: the stored outputs must equal the result of re-running its code cells.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_data.ipynb"

MD = nbformat.v4.new_markdown_cell
CODE = nbformat.v4.new_code_cell

CELLS = [
    MD(
        "# 01. Data\n\n"
        "This notebook inspects the raw data before it is used: structure, missing values, label, "
        "leakage risk, and sample consistency. All statistics are computed directly from the data; "
        "none are typed by hand."
    ),
    CODE(
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'params.yaml').exists())\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))"
    ),
    CODE(
        "import re\n\n"
        "import pandas as pd\n\n"
        "from src.config import load_params\n"
        "from src.data.load import data_hash, load_raw, to_csv_text\n"
        "from src.data.sample import make_sample\n"
        "from src.data.validate import banner, validate\n\n"
        "params = load_params()\n"
        "print('data_source:', params['data_source'])\n"
        "print('synthetic banner:', banner(params))"
    ),
    MD(
        "## Problem (excerpt from the README)\n\n"
        "The block below is read directly from `README.md`, not rewritten here."
    ),
    CODE(
        "readme = (ROOT / 'README.md').read_text(encoding='utf-8')\n"
        "block = re.search(r'## Problem\\n(.*?)(?=\\n## )', readme, re.S).group(1).strip()\n"
        "print(block)"
    ),
    MD("## Load and validate"),
    CODE(
        "raw = load_raw('raw', params)\n"
        "report = validate(raw, params)\n"
        "print('rows x columns:', raw.shape)\n"
        "print('data_hash:', data_hash(raw)[:16])\n"
        "print('validation ok:', report.ok)\n"
        "assert report.ok, report.errors"
    ),
    MD("## Column structure"),
    CODE(
        "def kind(s):\n"
        "    if pd.api.types.is_integer_dtype(s):\n"
        "        return 'integer'\n"
        "    if pd.api.types.is_float_dtype(s):\n"
        "        return 'float'\n"
        "    return 'text'\n\n\n"
        "info = raw.apply(kind).to_frame('type')\n"
        "info['missing'] = raw.isna().sum()\n"
        "print(info.to_string())"
    ),
    MD(
        "## Missing TotalCharges\n\n"
        "Validator rule: `TotalCharges` may be missing only when `tenure` = 0 "
        "(a new customer with no bills yet). "
        "The cell below verifies that the real data follows this rule."
    ),
    CODE(
        "nulls = raw[raw['TotalCharges'].isna()]\n"
        "print('rows with missing TotalCharges:', len(nulls))\n"
        "print('tenure values on those rows:', sorted(nulls['tenure'].unique().tolist()))\n"
        "print('label on those rows:')\n"
        "print(nulls['Churn'].value_counts().to_string())\n"
        "assert len(nulls) > 0 and (nulls['tenure'] == 0).all()"
    ),
    MD("## Label"),
    CODE(
        "sc = params['schema']\n"
        "y = (raw[sc['label_column']] == sc['positive_label']).astype(int)\n"
        "print(raw[sc['label_column']].value_counts().to_string())\n"
        "print('churn rate:', round(y.mean(), 4))\n"
        "assert raw[sc['id_column']].is_unique"
    ),
    MD("## Churn by segment"),
    CODE(
        "for col in ['Contract', 'InternetService', 'PaymentMethod']:\n"
        "    seg = raw.groupby(col)[sc['label_column']].agg(\n"
        "        n='size', churn_rate=lambda s: round((s == sc['positive_label']).mean(), 3)\n"
        "    )\n"
        "    print(seg.to_string())\n"
        "    print()"
    ),
    MD(
        "## Leakage risk: tenure and TotalCharges\n\n"
        "Both columns accumulate until churn is observed. The difference in their distributions "
        "between labels below shows why both are flagged in `params.yaml`."
    ),
    CODE(
        "flagged = params['leakage']['flagged_columns']\n"
        "print('flagged columns:', flagged)\n"
        "print('policy:', params['leakage']['policy'])\n"
        "for col in flagged:\n"
        "    print()\n"
        "    print(raw.groupby(sc['label_column'])[col].describe().round(2).to_string())"
    ),
    MD("## Sample versus raw data"),
    CODE(
        "sample = load_raw('sample', params)\n"
        "print('sample rows:', len(sample))\n"
        "print('sample validation ok:', validate(sample, params).ok)\n"
        "y_s = (sample[sc['label_column']] == sc['positive_label']).astype(int)\n"
        "print('churn rate difference sample - raw:', round(y_s.mean() - y.mean(), 4))\n"
        "committed = (ROOT / params['paths']['sample']).read_text(encoding='utf-8')\n"
        "same = committed == to_csv_text(make_sample(raw, params))\n"
        "print('sample.csv matches the generator:', same)\n"
        "assert same"
    ),
    MD(
        "## Findings for the next parts\n\n"
        "- One row per customer, unique ids, no dates: the non-temporal variant applies.\n"
        "- `TotalCharges` is missing only for new customers (`tenure` = 0); "
        "the validator enforces this rule.\n"
        "- The label maps `Churn == Yes` to 1. The positive class is the minority, "
        "so the primary metric is PR-based, not accuracy.\n"
        "- `tenure` and `TotalCharges` differ sharply between labels because they accumulate until churn; "
        "the default policy excludes them from the features.\n"
        "- The stratified sample keeps the churn rate close to that of the raw data."
    ),
]


def main(execute: bool = True) -> None:
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if execute:
        NotebookClient(
            nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}
        ).execute()
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"written {OUT}")


if __name__ == "__main__":
    main(execute="--no-exec" not in sys.argv)
