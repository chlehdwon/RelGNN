import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from relbench.tasks import get_task


def _type_name(value):
    if value is None:
        return "NoneType"
    return type(value).__name__


def summarize_time_column(series, max_type_examples=5):
    dtype_str = str(series.dtype)
    type_counts = series.map(_type_name).value_counts(dropna=False)
    unique_types = type_counts.index.tolist()
    summary = {
        "dtype": dtype_str,
        "num_unique_types": len(unique_types),
        "type_counts": type_counts,
    }

    if is_datetime64_any_dtype(series):
        summary["min"] = series.min()
        summary["max"] = series.max()
    elif np.issubdtype(series.dtype, np.number):
        summary["min"] = series.min()
        summary["max"] = series.max()
    else:
        summary["min"] = None
        summary["max"] = None

    if max_type_examples > 0:
        examples = {}
        for t in unique_types[:max_type_examples]:
            examples[t] = series[series.map(_type_name) == t].head(3).tolist()
        summary["examples"] = examples

    return summary


def count_unique_time_values(series):
    unique_values = series.dropna().unique().tolist()
    num_unique_values = len(unique_values)

    if is_datetime64_any_dtype(series) or looks_like_datetime_col(series):
        dt_series = pd.to_datetime(series, errors="coerce")
        unique_dates = dt_series.dropna().dt.date.unique().tolist()
        num_unique_dates = len(unique_dates)
    else:
        unique_dates = []
        num_unique_dates = 0

    return num_unique_values, num_unique_dates, unique_values, unique_dates


def looks_like_datetime_col(series, sample_size=50):
    if is_datetime64_any_dtype(series):
        return True
    if series.dtype != "object":
        return False
    sample = series.dropna().head(sample_size).tolist()
    for value in sample:
        if isinstance(value, (pd.Timestamp, np.datetime64, datetime)):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Inspect timestamp types in relbench task tables."
    )
    parser.add_argument("--dataset", type=str, default="rel-amazon")
    parser.add_argument("--task", type=str, default="user-churn")
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    task = get_task(args.dataset, args.task, download=True)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Time column: {task.time_col}")
    print("")

    global_type_counts = Counter()
    global_unique_values = set()
    global_unique_dates = set()

    for split in split_names:
        table = task.get_table(split, mask_input_cols=False)
        df = table.df
        time_series = df[task.time_col]

        summary = summarize_time_column(time_series)
        global_type_counts.update(summary["type_counts"].to_dict())
        (
            num_unique_values,
            num_unique_dates,
            unique_values,
            unique_dates,
        ) = count_unique_time_values(time_series)
        global_unique_values.update(unique_values)
        global_unique_dates.update(unique_dates)

        print(f"Split: {split}")
        print(f"- dtype: {summary['dtype']}")
        print(f"- num unique value types: {summary['num_unique_types']}")
        print(f"- num unique time values: {num_unique_values}")
        print(f"- num unique dates: {num_unique_dates}")
        print(f"- type counts: {summary['type_counts'].to_dict()}")
        if summary["min"] is not None or summary["max"] is not None:
            print(f"- min: {summary['min']}")
            print(f"- max: {summary['max']}")
        if "examples" in summary:
            print(f"- examples by type: {summary['examples']}")

        timestamp_cols = [
            col for col in df.columns if looks_like_datetime_col(df[col])
        ]
        print(f"- datetime-like columns in table: {timestamp_cols}")
        print("")

    print("Combined time_col value types across splits:")
    print(f"- type counts: {dict(global_type_counts)}")
    print(f"- num unique types: {len(global_type_counts)}")
    print("Combined unique time values across splits:")
    print(f"- num unique time values: {len(global_unique_values)}")
    print(f"- num unique dates: {len(global_unique_dates)}")


if __name__ == "__main__":
    main()
