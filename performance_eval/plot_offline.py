#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PATTERN = re.compile(
    r'^(?P<model>.+?)_in(?P<input_len>\d+)_seq(?P<max_num_seqs>\d+)\.json$'
)

def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        m = PATTERN.match(path.name)
        if not m:
            print(f"Skipping unrecognized filename: {path.name}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        rows.append({
            "file": path.name,
            "model": m.group("model"),
            "input_len": int(m.group("input_len")),
            "max_num_seqs": int(m.group("max_num_seqs")),
            "elapsed_time": data.get("elapsed_time"),
            "num_requests": data.get("num_requests"),
            "total_num_tokens": data.get("total_num_tokens"),
            "requests_per_second": data.get("requests_per_second"),
            "tokens_per_second": data.get("tokens_per_second"),
        })

    if not rows:
        raise ValueError(f"No matching JSON files found in {results_dir}")

    return pd.DataFrame(rows).sort_values(["model", "input_len", "max_num_seqs"]).reset_index(drop=True)


def safe_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def plot_metric(df: pd.DataFrame, metric: str, out_path: str, title_prefix: str) -> None:
    plt.figure(figsize=(8, 5))
    for input_len in sorted(df["input_len"].unique()):
        sub = df[df["input_len"] == input_len].sort_values("max_num_seqs")
        plt.plot(
            sub["max_num_seqs"],
            sub[metric],
            marker="o",
            label=f"input_len={input_len}",
        )

    plt.xlabel("max_num_seqs")
    plt.ylabel(metric)
    plt.title(f"{title_prefix}: {metric} vs max_num_seqs")
    plt.legend(title="input_len")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--prefix", default="offline")
    args = parser.parse_args()

    df = load_results(args.results_dir)

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        model_tag = safe_name(model)
        base = f"{args.prefix}_{model_tag}"

        sub.to_csv(f"{base}_summary.csv", index=False)
        print(f"Saved {base}_summary.csv")

        plot_metric(sub, "tokens_per_second", f"{base}_tokens_per_second.png", model)
        plot_metric(sub, "requests_per_second", f"{base}_requests_per_second.png", model)
        plot_metric(sub, "elapsed_time", f"{base}_elapsed_time.png", model)


if __name__ == "__main__":
    main()