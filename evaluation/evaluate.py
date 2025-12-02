import argparse
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from math import sqrt

def load_workouts(workout_json):
    with open(workout_json, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_ground_truth(gt_dir):
    gt_files = Path(gt_dir).glob("gt_*.json")
    rows = []
    for p in gt_files:
        d = json.load(open(p))
        sess = d.get("session_file")
        for mark in d.get("gt", []):
            rows.append({
                "session_file": sess,
                "frame_index": mark.get("frame_index"),
                "timestamp": mark.get("timestamp")
            })
    return pd.DataFrame(rows)

def compute_metrics(work_df, gt_df):
    ps_df = work_df.rename(columns={'reps': 'detected'}).copy()
    ps_df['gt'] = np.nan
    ps_df['error'] = np.nan
    ps_df['abs_error'] = np.nan
    return ps_df

def summary_metrics(ps_df):
    m = {}
    if 'abs_error' in ps_df:
        m['MAE'] = ps_df['abs_error'].mean()
    return m

def plots(ps_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure()
    plt.hist(ps_df['detected'], bins=10)
    fig.savefig(os.path.join(out_dir, "hist_detected.png"))
    plt.close(fig)

def main(args):
    work_df = load_workouts(args.workout)
    if args.gt_dir:
        gt_df = load_ground_truth(args.gt_dir)
    else:
        gt_df = pd.DataFrame()
    ps_df = compute_metrics(work_df, gt_df)
    metrics = summary_metrics(ps_df)
    print("Summary metrics:", metrics)
    plots(ps_df, args.out_dir)
    ps_df.to_csv(os.path.join(args.out_dir, "per_session_metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workout", required=True)
    parser.add_argument("--gt_dir", default="evaluation/data")
    parser.add_argument("--out_dir", default="evaluation/reports")
    args = parser.parse_args()
    main(args)
