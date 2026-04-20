# -*- coding: utf-8 -*-
"""
Build Correlation CSVs from Master Arrays
==========================================
For each feature (theta, alpha, aperiodic_exponent), computes the
Spearman correlation between that feature and setsize across trials,
for every subject × channel combination.

Channel names are loaded from chanlocs_55.mat (55 channels, EEGLAB format).

Output — three CSV files:
    correlation_csvs/correlation_theta.csv
    correlation_csvs/correlation_alpha.csv
    correlation_csvs/correlation_aperiodic_exponent.csv

Each CSV columns:
    Subject | Channel | Band | Correlation | P_Value

@author: Rahul Venugopal
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_DATA_PATH    = r"master_arrays/master_data.npy"
MASTER_SETSIZE_PATH = r"master_arrays/master_setsize.npy"
TRIAL_COUNTS_PATH   = r"master_arrays/trial_counts.npy"
CHANLOCS_PATH       = r"chanlocs_55.mat"
OUT_DIR             = r"correlation_csvs"

# Feature axis mapping inside master_data (axis=1)
FEATURES = {
    "Theta"              : 0,
    "Alpha"              : 1,
    "Aperiodic_Exponent" : 2,
}

# Minimum non-NaN trial pairs required to compute a valid correlation
MIN_VALID_TRIALS = 20

# =============================================================================
# LOAD CHANNEL NAMES FROM CHANLOCS
# =============================================================================

def load_ch_names(chanlocs_path: str) -> list[str]:
    """
    Load channel labels from an EEGLAB-format chanlocs .mat file.
    Returns a list of 55 label strings in electrode order.
    """
    mat = sio.loadmat(chanlocs_path, squeeze_me=True)

    # Find the chanlocs struct (first non-dunder key)
    key = next(k for k in mat if not k.startswith('_'))
    cl  = mat[key]

    labels = [str(cl['labels'][i]) for i in range(len(cl))]
    print(f"  Loaded {len(labels)} channel labels from {chanlocs_path}")
    print(f"  Labels: {labels}")
    return labels

# =============================================================================
# CORRELATION BUILDER
# =============================================================================

def build_correlation_df(
    feat_data   : np.ndarray,    # (n_subjects, n_channels, max_trials)
    setsize     : np.ndarray,    # (n_subjects, max_trials)
    trial_counts: np.ndarray,    # (n_subjects,)
    ch_names    : list[str],
    band_label  : str,
) -> pd.DataFrame:
    """
    Spearman r between feature values and setsize across trials,
    per subject × channel.  NaN for channels with < MIN_VALID_TRIALS pairs.
    """
    n_subjects, n_channels, _ = feat_data.shape
    records = []

    for s in range(n_subjects):
        n_trials = int(trial_counts[s])
        if n_trials == 0:
            continue

        sz = setsize[s, :n_trials]               # (n_trials,)

        for ch in range(n_channels):
            feat = feat_data[s, ch, :n_trials]   # (n_trials,)

            # Valid pairs only (non-NaN in both feature and setsize)
            valid   = ~np.isnan(feat) & ~np.isnan(sz)
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_TRIALS:
                r, p = np.nan, np.nan
            else:
                r, p = spearmanr(feat[valid], sz[valid])

            records.append({
                "Subject"     : s + 1,           # 1-indexed to match convention
                "Channel"     : ch_names[ch],
                "Band"        : band_label,
                "Correlation" : r,
                "P_Value"     : p,
            })

    return pd.DataFrame(records)

# =============================================================================
# MAIN
# =============================================================================

def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading inputs...")
    master_data    = np.load(MASTER_DATA_PATH)      # (36, 3, 55, 300)
    master_setsize = np.load(MASTER_SETSIZE_PATH)   # (36, 300)
    trial_counts   = np.load(TRIAL_COUNTS_PATH)     # (36,)
    ch_names       = load_ch_names(CHANLOCS_PATH)   # 55 labels

    print(f"\n  master_data    : {master_data.shape}")
    print(f"  master_setsize : {master_setsize.shape}")
    print(f"  Subjects       : {len(trial_counts)}")
    print(f"  Channels       : {len(ch_names)}\n")

    for band_label, feat_idx in FEATURES.items():
        print(f"Processing: {band_label}  (feature axis {feat_idx})")

        feat_data = master_data[:, feat_idx, :, :]  # (36, 55, 300)

        df = build_correlation_df(
            feat_data    = feat_data,
            setsize      = master_setsize,
            trial_counts = trial_counts,
            ch_names     = ch_names,
            band_label   = band_label,
        )

        valid_corrs = df["Correlation"].dropna()
        sig_count   = (df["P_Value"] < 0.05).sum()

        print(f"  Rows        : {len(df)}")
        print(f"  NaN rows    : {df['Correlation'].isna().sum()}")
        print(f"  p < 0.05    : {sig_count} / {len(df)}")
        print(f"  Corr range  : {valid_corrs.min():.3f} to {valid_corrs.max():.3f}")

        safe_label = band_label.lower()
        out_path   = os.path.join(OUT_DIR, f"correlation_{safe_label}.csv")
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"  Saved → {out_path}\n")

    print("All done.")

if __name__ == "__main__":
    run()
