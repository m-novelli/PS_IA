import argparse, sys, json
import pandas as pd
from pathlib import Path
from src.monitoring.drift import detect_drift

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    return pd.read_csv(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--cols", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="artifacts/drift_report.csv")
    ap.add_argument("--out_json", type=str, default="artifacts/drift_summary.json")
    ap.add_argument("--fail_on_drift_rate", type=float, default=0.30)
    args = ap.parse_args()

    df_ref  = _load(args.ref)
    df_prod = _load(args.prod)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()] or None

    out = detect_drift(df_ref, df_prod, alpha=args.alpha, columns=cols)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "n_features": int(len(out)),
        "n_drifted": int(out["drift"].sum()) if len(out) else 0,
        "drift_rate": float(out.attrs.get("drift_rate", 0.0))
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(2 if summary["drift_rate"] >= args.fail_on_drift_rate else 0)

if __name__ == "__main__":
    main()
