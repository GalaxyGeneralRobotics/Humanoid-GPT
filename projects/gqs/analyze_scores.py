import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# ================= Configuration =================
SCORE_JSON = "storage/gqs_score/amass.json"
CLASS_JSON = "storage/configs/amass_n20.json"

# Calibrated weights (must match the scoring formula in physics_filter.py)
WEIGHTS = {
    "foot_sliding": 1.70,
    "velocity_violation": 44.22,
    "self_collision": 0.17,
    "jerk": 0.28,
    "penetration": 216.62,
    "floating_frames_ratio": 24.19
}
# ===========================================

def load_data():
    print(f"Loading scores from {SCORE_JSON}...")
    with open(SCORE_JSON, 'r') as f:
        score_data = json.load(f)
    details = score_data.get("details", {})

    print(f"Loading classes from {CLASS_JSON}...")
    with open(CLASS_JSON, 'r') as f:
        class_data = json.load(f)

    # Build filename -> class_id mapping
    file_to_class = {}
    for cid, paths in class_data.items():
        for p in paths:
            fname = os.path.basename(p)
            file_to_class[fname] = cid

    return details, file_to_class, class_data

def calculate_deductions(metrics, weights):
    """Compute per-metric score deductions."""
    deductions = {}
    total_deduction = 0.0
    for k, w in weights.items():
        val = metrics.get(k, 0.0)
        # Handle invalid values: treat None or non-finite as 0
        if val is None or not np.isfinite(val):
            val = 0.0
        score_loss = val * w
        deductions[k] = score_loss
        total_deduction += score_loss
    return deductions, total_deduction

def analyze_group(group_name, metrics_list):
    """Compute summary statistics for a group of samples."""
    if not metrics_list:
        return None

    stats = {}
    n = len(metrics_list)

    # Convert list of dicts to dict of lists for easy numpy computation
    # data_by_key: {'foot_sliding': [0.1, 0.2, ...], ...}
    data_by_key = defaultdict(list)
    total_deductions = []

    for m in metrics_list:
        deds, tot = calculate_deductions(m, WEIGHTS)
        total_deductions.append(tot)
        for k, v in deds.items():
            data_by_key[k].append(v)

    # Mean total deduction for this group, used to compute contribution ratios
    avg_total_loss = np.mean(total_deductions)
    if avg_total_loss < 1e-6: avg_total_loss = 1e-6 # Avoid division by zero

    summary = []
    for k in WEIGHTS.keys():
        arr = np.array(data_by_key[k])

        # 1. Penalty rate: fraction of samples with deduction > 0.001
        penalty_rate = np.mean(arr > 0.001) * 100

        # 2. Mean deduction
        mean_deduction = np.mean(arr)

        # 3. Max deduction
        max_deduction = np.max(arr)

        # 4. Contribution ratio (this metric's mean deduction / total mean deduction)
        contribution = (mean_deduction / avg_total_loss) * 100

        summary.append({
            "Metric": k,
            "Penalty Rate (%)": f"{penalty_rate:.1f}%",
            "Mean Deduction": f"{mean_deduction:.2f}",
            "Max Deduction": f"{max_deduction:.2f}",
            "Contrib (%)": f"{contribution:.1f}%"
        })

    return pd.DataFrame(summary)

def main():
    if not os.path.exists(SCORE_JSON):
        print("Score file not found.")
        return

    details, file_to_class, class_data = load_data()

    # 1. Prepare data containers
    all_metrics = []
    class_metrics = defaultdict(list)

    valid_count = 0
    missing_class_count = 0

    for fname, mets in details.items():
        all_metrics.append(mets)

        if fname in file_to_class:
            cid = file_to_class[fname]
            class_metrics[cid].append(mets)
        else:
            missing_class_count += 1

    print(f"Total files in score json: {len(details)}")
    print(f"Files matched to classes: {len(details) - missing_class_count}")
    print("-" * 60)

    # 2. Overall analysis
    print("\n=== [Overall Dataset Statistics] ===")
    df_all = analyze_group("Overall", all_metrics)
    print(df_all.to_string(index=False))

    # 3. Per-class analysis (sorted by Class ID)
    sorted_cids = sorted(class_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # To keep the output compact, print each class's "Top Contributor"
    # (the largest-deduction metric) along with brief information.

    print("\n\n=== [Per-Class Breakdown] ===")
    print(f"{'Class ID':<10} | {'Files':<6} | {'Avg Score Loss':<15} | {'Main Penalty Source (Contrib %)'}")
    print("-" * 80)

    for cid in sorted_cids:
        mets = class_metrics[cid]
        deds_list = []
        for m in mets:
            _, t = calculate_deductions(m, WEIGHTS)
            deds_list.append(t)

        avg_loss = np.mean(deds_list)

        # Analyze this class
        df = analyze_group(cid, mets)
        # Find the metric with the largest contribution by recomputing the
        # largest mean deduction directly (simpler than parsing the string table)
        sums = {}
        for k in WEIGHTS.keys():
            vals = [calculate_deductions(m, WEIGHTS)[0][k] for m in mets]
            sums[k] = np.mean(vals)

        main_source = max(sums, key=sums.get)
        main_contrib = (sums[main_source] / avg_loss * 100) if avg_loss > 1e-6 else 0.0

        print(f"{cid:<10} | {len(mets):<6} | {avg_loss:<15.2f} | {main_source} ({main_contrib:.1f}%)")

    # To inspect the detailed table for a specific class, enable the lines below
    # print("\nDetailed Stats for Class '0':")
    # print(analyze_group("0", class_metrics['0']).to_string(index=False))

if __name__ == "__main__":
    main()