"""Isolated Parquet row-group counter/sampler for preprocessing_fast.py."""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SEED = 1
TRUE_VALUES = ["t", "true", "1", "1.0"]
FLAG_COLS = [
    "is_opened", "is_clicked", "is_unsubscribed",
    "is_complained", "is_purchased",
]


def flag(values):
    return (
        values.astype(str).str.strip().str.lower()
        .isin(TRUE_VALUES).astype(np.int8)
    )


def main():
    mode = sys.argv[1]
    source = sys.argv[2]
    row_group_start = int(sys.argv[3])
    row_group_end = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    output = sys.argv[6] if mode == "sample" else None
    neg_ratio = float(sys.argv[7]) if mode == "sample" else None
    bad_groups = {
        int(value) for value in (sys.argv[8] if mode == "sample" else sys.argv[6]).split(",")
        if value.strip()
    }

    parquet_file = pq.ParquetFile(source)
    total_neg = selected_pos = selected_neg = 0
    writer = None
    try:
        columns = ["is_purchased"] if mode == "count" else None
        for row_group in range(row_group_start, row_group_end):
            if row_group in bad_groups:
                continue
            rng = np.random.RandomState(SEED + row_group * 1_000_003)
            for batch in parquet_file.iter_batches(
                row_groups=[row_group], columns=columns,
                batch_size=batch_size, use_threads=False,
            ):
                purchase_idx = batch.schema.get_field_index("is_purchased")
                if purchase_idx < 0:
                    raise KeyError("Required column 'is_purchased' is missing.")
                purchase = flag(batch.column(purchase_idx).to_pandas()).to_numpy()
                if mode == "count":
                    total_neg += int((purchase == 0).sum())
                    continue

                keep = purchase == 1
                selected_pos += int(keep.sum())
                neg_idx = np.flatnonzero(purchase == 0)
                if len(neg_idx):
                    chosen = neg_idx[rng.random_sample(len(neg_idx)) < neg_ratio]
                    keep[chosen] = True
                    selected_neg += len(chosen)
                if keep.any():
                    selected = pa.Table.from_batches([batch.filter(pa.array(keep))]).to_pandas()
                    for col in FLAG_COLS:
                        selected[col] = flag(selected[col]) if col in selected.columns else 0
                    table = pa.Table.from_pandas(selected, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output, table.schema, compression="snappy")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if mode == "count":
        print(f"TAU_WORKER_COUNT={total_neg}")
    elif mode == "sample":
        print(f"TAU_WORKER_SAMPLE={selected_pos},{selected_neg}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
