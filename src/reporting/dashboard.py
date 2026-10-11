"""Dashboard: model_hash, contact list, synthetic-data banner, business value summary.

It also scores an uploaded CSV or Excel file with the saved model and offers the contact list as a
CSV or Excel download. Uploaded rows are scored in memory only; nothing is written to disk.

Run: streamlit run src/reporting/dashboard.py   (or: make dashboard)
The project root can be overridden with CHURN_ROOT (used by tests). All numbers are read from
artifacts; there are no business numbers in this file.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import ROOT, business_status, load_all
from src.data.load import DataLoadError, clean, load_raw
from src.data.validate import banner
from src.features.build import FeatureError
from src.models.predict import PredictError, predict
from src.models.targeting import ContactPlan, TargetingError, plan_contacts
from src.models.train import load_bundle


class DashboardError(RuntimeError):
    """An artifact is missing or out of sync. `code` is stable for tests."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class View:
    """Everything the dashboard shows, loaded once."""

    banner: str | None
    business_status: str
    model_hash: str
    model_name: str
    metrics: dict
    roi: dict
    p_star: float
    k: int
    plan: ContactPlan
    n_scored: int


# --- Loading ----------------------------------------------------------------------------------


def project_root() -> Path:
    return Path(os.environ.get("CHURN_ROOT") or ROOT)


def _require_artifacts(root: Path, paths: dict) -> dict[str, Path]:
    """Return the artifact paths, or raise E_DASH_MISSING for the first one that does not exist."""
    needed = {
        "metrics": root / paths["metrics"],
        "roi": root / paths["roi"],
        "model": root / paths["artifacts_dir"] / "model.joblib",
    }
    for name, path in needed.items():
        if not path.is_file():
            raise DashboardError(
                "E_DASH_MISSING", f"{name} not found at {path.relative_to(root)}; run make all"
            )
    return needed


def load_view(root: Path | None = None) -> View:
    root = Path(root) if root is not None else project_root()
    params, assumptions = load_all(root / "params.yaml", root / "assumptions.yaml")
    paths = params["paths"]

    files = _require_artifacts(root, paths)
    metrics = json.loads(files["metrics"].read_text(encoding="utf-8"))
    roi = json.loads(files["roi"].read_text(encoding="utf-8"))
    bundle = load_bundle(files["model"])

    # The saved model must be the one that produced metrics.json, otherwise the page would mix runs.
    if bundle.model.model_hash != metrics["hashes"]["model"]:
        raise DashboardError("E_DASH_STALE", "artifact model_hash differs from metrics.json; run make train")

    # Score the same data the metrics were computed on, then plan contacts.
    data_key = metrics.get("data", "sample")
    df = load_raw(root / paths["sample" if data_key == "sample" else "raw"], params)
    pred = predict(df, bundle, params)
    plan = plan_contacts(pred, assumptions)

    return View(
        banner=banner({"data_source": metrics["data_source"]}) or banner(params),
        business_status=business_status(assumptions),
        model_hash=bundle.model.model_hash,
        model_name=metrics["selected"],
        metrics=metrics,
        roi=roi,
        p_star=plan.p_star,
        k=plan.k,
        plan=plan,
        n_scored=len(pred),
    )


def contact_table(view: View) -> pd.DataFrame:
    """The contact list as a table, with customerID as a regular column."""
    return plan_table(view.plan)


def policy_table(view: View) -> pd.DataFrame:
    """One row per compared policy (no contact, model, random)."""
    return pd.DataFrame(
        [
            {
                "policy": name,
                "contacts": p["n_contacts"],
                "churners_contacted": p["n_churners_contacted"],
                "net_value": p["net_value"],
            }
            for name, p in view.roi["policies"].items()
        ]
    )


# --- Upload and download ------------------------------------------------------------------------

UPLOAD_SUFFIXES = (".csv", ".xlsx")
UPLOAD_ERRORS = (DashboardError, DataLoadError, PredictError, FeatureError, TargetingError)


def read_upload(name: str, data: bytes, params: dict) -> pd.DataFrame:
    """Read an uploaded CSV or XLSX (first sheet) as text, then `clean` it like the pipeline does.

    Raises DashboardError E_UPLOAD_TYPE for other file types and E_UPLOAD_READ for unreadable files.
    """
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise DashboardError("E_UPLOAD_TYPE", f"{name}: use a .csv or .xlsx file")
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False, engine="openpyxl")
    except Exception as e:  # parser errors differ per engine; report them all the same way
        raise DashboardError("E_UPLOAD_READ", f"{name}: cannot read the file ({e})") from e
    if df.empty:
        raise DashboardError("E_UPLOAD_READ", f"{name}: the file has no rows")
    return clean(df, params, name=name)


def load_scoring(root: Path | None = None):
    """The saved model bundle with the configuration needed to score new rows."""
    root = Path(root) if root is not None else project_root()
    params, assumptions = load_all(root / "params.yaml", root / "assumptions.yaml")
    files = _require_artifacts(root, params["paths"])
    return load_bundle(files["model"]), params, assumptions


def score_upload(df: pd.DataFrame, bundle, params: dict, assumptions: dict) -> ContactPlan:
    """Score uploaded rows and plan contacts with the same rules as the main contact list."""
    return plan_contacts(predict(df, bundle, params), assumptions)


def plan_table(plan: ContactPlan) -> pd.DataFrame:
    """A contact plan as a table, with customerID as a regular column."""
    contacts = plan.contacts[["rank", "p_churn", "out_of_domain"]].copy()
    contacts.index.name = "customerID"
    return contacts.reset_index()


# A text cell starting with one of these is run as a formula by Excel and LibreOffice. customerID comes
# from an uploaded file, so exports must not pass it on as a formula (CSV/formula injection).
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralize(value):
    """Prefix a text value that a spreadsheet would run as a formula with an apostrophe."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv_bytes(table: pd.DataFrame) -> bytes:
    safe = table.copy()
    for col in safe.columns:
        if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(_neutralize)
    return safe.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(table: pd.DataFrame) -> bytes:
    """Excel export; text that looks like a formula is stored as text, with the value unchanged."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="contacts")
        for row in writer.sheets["contacts"].iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"
    return buf.getvalue()


# --- Page -------------------------------------------------------------------------------------


def _show_banners(st, v: View) -> None:
    if v.banner:
        st.error(v.banner)
    if v.business_status == "conditional":
        st.warning("Conditional value: save_rate is assumed, so all business values hold only conditionally.")


def _show_model(st, v: View) -> None:
    st.subheader("Model")
    st.code(v.model_hash, language="text")
    st.caption(f"model_hash, selected model: {v.model_name}")
    st.code(v.metrics["hashes"]["input_data"], language="text")
    st.caption("input data sha256 (from metrics.json)")

    t = v.metrics["test"][v.model_name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC (test)", f"{t['pr_auc']:.3f}")
    c2.metric(f"Lift@{t['k']} (test)", f"{t['lift_at_k']:.2f}")
    c3.metric("p* (break-even)", f"{v.p_star:.4f}")
    c4.metric("Capacity k", f"{v.k}")


def _download_buttons(st, table: pd.DataFrame, stem: str, key: str) -> None:
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("Download CSV", to_csv_bytes(table), f"{stem}.csv", "text/csv", key=f"{key}_csv")
    c2.download_button(
        "Download Excel",
        to_xlsx_bytes(table),
        f"{stem}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key}_xlsx",
    )


def _show_upload(st) -> None:
    st.subheader("Score your own file")
    st.caption(
        "Upload a .csv or .xlsx with the same columns as the training data "
        "(the Churn column is not needed). Rows are scored in memory and not saved."
    )
    f = st.file_uploader("Customer file", type=[s.lstrip(".") for s in UPLOAD_SUFFIXES])
    if f is None:
        return
    try:
        bundle, params, assumptions = load_scoring()
        df = read_upload(f.name, f.getvalue(), params)
        plan = score_upload(df, bundle, params, assumptions)
    except UPLOAD_ERRORS as e:
        st.error(str(e))
        return
    st.success(
        f"Scored {len(df)} customers; {plan.n_contacts} on the contact list "
        f"(p* = {plan.p_star:.4f}, k = {plan.k})."
    )
    table = plan_table(plan)
    st.dataframe(table, hide_index=True)
    _download_buttons(st, table, f"contact_list_{Path(f.name).stem}", key="upload")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Telco Churn", layout="wide")
    st.title("Telco Churn: contact list")

    try:
        v = load_view()
    except DashboardError as e:
        st.error(f"{e.code}: {e.message}")
        return

    _show_banners(st, v)
    _show_model(st, v)

    st.subheader(f"Contact list ({v.plan.n_contacts} of {v.n_scored} customers)")
    st.dataframe(contact_table(v), hide_index=True)

    _download_buttons(st, contact_table(v), "contact_list", key="main")

    st.subheader("Business value (test set)")
    st.dataframe(policy_table(v), hide_index=True)

    _show_upload(st)


if __name__ == "__main__":
    main()
