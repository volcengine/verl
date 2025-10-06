import json
import os
from pathlib import Path

import datasets


def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    jsonl_path = Path(jsonl_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Sanitize empty structs that parquet cannot write (e.g., {})
            extra = rec.get("extra_info", {})
            tools_kwargs = extra.get("tools_kwargs", None)
            if isinstance(tools_kwargs, dict) and len(tools_kwargs) == 0:
                extra["tools_kwargs"] = {"placeholder": None}
                rec["extra_info"] = extra
            records.append(rec)

    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(str(parquet_path))
    print(f"Wrote {len(ds)} rows to {parquet_path}")


if __name__ == "__main__":
    here = Path(__file__).parent
    jsonl = here / "frozen_lake_dataset.jsonl"
    out = here / "frozen_lake_dataset.parquet"
    jsonl_to_parquet(str(jsonl), str(out))

