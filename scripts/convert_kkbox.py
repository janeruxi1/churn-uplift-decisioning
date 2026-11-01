"""Convert the downloaded KKBox CSV files to Parquet, in pieces small enough for a modest laptop.

Download from the Kaggle competition "WSDM - KKBox's Churn Prediction Challenge" (accept its rules
first): train.csv, train_v2.csv, members_v3.csv, transactions.csv, transactions_v2.csv and
user_logs_v2.csv, and extract them into data/raw/kkbox/. Then:

    python scripts/convert_kkbox.py             # converts everything not yet converted
    python scripts/convert_kkbox.py --seconds 140   # stop after ~140 s; run again to resume

Large files are read in ~300 MB byte ranges and written as numbered Parquet parts, so the work
can be split across several short runs; progress is kept in data/raw/kkbox/parquet/.progress.json.
After it finishes, the .csv and .7z files can be deleted (about 5 GB).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kkbox"
OUT = RAW / "parquet"
SINGLE = ["train", "train_v2", "members_v3", "transactions_v2"]
PARTED = ["transactions", "user_logs_v2"]
CHUNK = 300 * 2 ** 20


def find_csv(name: str) -> Path | None:
    hits = sorted(RAW.rglob(f"{name}.csv"), key=lambda p: -p.stat().st_size if p.is_file() else 0)
    return next((p for p in hits if p.is_file()), None)


def read_bytes(buf: bytes, names: list[str]) -> pa.Table:
    return pc.read_csv(io.BytesIO(buf), read_options=pc.ReadOptions(column_names=names),
                       convert_options=pc.ConvertOptions(column_types={"msno": pa.string()}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=float("inf"))
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    state_f = OUT / ".progress.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    t0 = time.time()
    for name in SINGLE + PARTED:
        src = find_csv(name)
        done = ((OUT / f"{name}.parquet").exists() or state.get(name, {}).get("done")
                or (name not in state and (OUT / name).is_dir()))   # converted by an earlier version
        if done:
            continue
        if src is None:
            print(f"{name}.csv not found under {RAW}; skipping")
            continue
        with open(src, "rb") as fh:
            names = fh.readline().decode().strip().split(",")
            st = state.setdefault(name, {"offset": fh.tell(), "part": 0, "rows": 0})
            while time.time() - t0 < args.seconds:
                fh.seek(st["offset"])
                buf = fh.read(CHUNK if name in PARTED else -1)
                if not buf:
                    st["done"] = True
                    break
                if name in PARTED and len(buf) == CHUNK:
                    buf = buf[: buf.rfind(b"\n") + 1]
                tbl = read_bytes(buf, names)
                dst = OUT / name / f"part-{st['part']:03d}.parquet" if name in PARTED else OUT / f"{name}.parquet"
                dst.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(tbl, f"{dst}.tmp", compression="zstd")
                os.replace(f"{dst}.tmp", dst)
                st["offset"] += len(buf)
                st["part"] += 1
                st["rows"] += tbl.num_rows
                state_f.write_text(json.dumps(state, indent=1))
                print(f"{name}: {st['rows']:,} rows", flush=True)
                if name not in PARTED:
                    st["done"] = True
                    break
        state_f.write_text(json.dumps(state, indent=1))
        if time.time() - t0 >= args.seconds:
            print("time limit reached; run again to continue")
            return 0
    print("all files converted" if all(state.get(n, {}).get("done") or (OUT / f"{n}.parquet").exists()
                                       for n in SINGLE + PARTED) else "some files are still missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
