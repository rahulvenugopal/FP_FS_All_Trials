# -*- coding: utf-8 -*-
"""
Build Master Parameter Array Across Subjects
=============================================
Loads per-subject specparam outputs and assembles a single master array.

mat file structure (wmload_sequence.mat):
    capacities : (36,) object array
                 each entry is a 1-D uint8 array of length n_trials_for_that_subject
                 containing the setsize (capacity) value for each trial.

Master array shapes:
    master_data    : (n_subjects, n_features, n_channels, max_trials)  float64, NaN-padded
    master_setsize : (n_subjects, max_trials)                           float64, NaN-padded

Features (axis=1):
    0 → theta_oscillatory_power
    1 → alpha_oscillatory_power
    2 → aperiodic_exponent

Zeros in the numpy arrays → NaN  (failed specparam fits).

@author: Rahul Venugopal
"""

import os
import glob
import numpy as np
import scipy.io as sio

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR  = r"spectral_params_trial_avg"   # folder with per-subject subdirs
MAT_PATH  = r"wmload_sequence.mat"         # single .mat file for all subjects
OUT_DIR   = r"master_arrays"

N_SUBJECTS  = 36
N_CHANNELS  = 55
MAX_TRIALS  = 300

FEATURE_FILES = [
    "theta_oscillatory_power.npy",
    "alpha_oscillatory_power.npy",
    "aperiodic_exponent.npy",
]
FEATURE_NAMES = ["theta_oscillatory_power",
                 "alpha_oscillatory_power",
                 "aperiodic_exponent"]

# =============================================================================
# HELPERS
# =============================================================================

def load_capacities(mat_path: str) -> list:
    """
    Load wmload_sequence.mat and return a list of 36 1-D float arrays,
    one per subject, each of length n_trials for that subject.
    """
    mat  = sio.loadmat(mat_path, squeeze_me=True)
    caps = mat["capacities"]                         # (36,) object array
    return [caps[i].astype(float).ravel() for i in range(len(caps))]


def load_npy_as_channels_x_trials(path: str, n_channels: int) -> np.ndarray:
    """
    Load a .npy file, ensure shape (n_channels, n_trials),
    and replace zeros with NaN.

    The pipeline saves (n_trials, n_channels) so we transpose.
    """
    arr = np.load(path).astype(float)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape} in {path}")

    if arr.shape[1] == n_channels and arr.shape[0] != n_channels:
        arr = arr.T                                  # (n_trials, n_ch) → (n_ch, n_trials)
    elif arr.shape[0] == n_channels:
        pass                                         # already (n_ch, n_trials)
    else:
        raise ValueError(
            f"Neither dimension matches n_channels={n_channels} "
            f"in {path} with shape {arr.shape}."
        )

    arr[arr == 0] = np.nan                           # zeros = failed specparam fits
    return arr                                       # (n_channels, n_trials)


def pad_1d(arr: np.ndarray, length: int) -> np.ndarray:
    out = np.full(length, np.nan)
    n   = min(len(arr), length)
    out[:n] = arr[:n]
    return out


def pad_2d(arr: np.ndarray, max_trials: int) -> np.ndarray:
    """Pad (n_channels, n_trials) → (n_channels, max_trials)."""
    n_ch, n_tr = arr.shape
    out = np.full((n_ch, max_trials), np.nan)
    n   = min(n_tr, max_trials)
    out[:, :n] = arr[:, :n]
    return out

# =============================================================================
# MAIN
# =============================================================================

def build_master_array():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load setsize sequences for all 36 subjects ───────────────────────────
    print(f"Loading setsize sequences from: {MAT_PATH}")
    all_capacities   = load_capacities(MAT_PATH)
    trial_counts_mat = [len(c) for c in all_capacities]
    print(f"  Subjects in .mat : {len(all_capacities)}")
    print(f"  Trial counts     — min={min(trial_counts_mat)}, "
          f"max={max(trial_counts_mat)}, mean={np.mean(trial_counts_mat):.0f}\n")

    # ── Discover subject folders (sorted = same order as .mat) ───────────────
    sub_dirs = sorted([
        d for d in glob.glob(os.path.join(ROOT_DIR, "*"))
        if os.path.isdir(d)
    ])
    n_found = len(sub_dirs)
    print(f"Found {n_found} subject folder(s) in {ROOT_DIR}.\n")

    # ── Pre-allocate ──────────────────────────────────────────────────────────
    n_features     = len(FEATURE_FILES)
    master_data    = np.full((n_found, n_features, N_CHANNELS, MAX_TRIALS), np.nan)
    master_setsize = np.full((n_found, MAX_TRIALS), np.nan)
    subject_ids    = []
    trial_counts   = []
    failed         = []

    for sub_idx, sub_dir in enumerate(sub_dirs):
        sub_id = os.path.basename(sub_dir)
        subject_ids.append(sub_id)
        print(f"[{sub_idx+1:>3}/{n_found}]  {sub_id}")

        # ── Setsize (matched by sort order to .mat rows) ──────────────────────
        if sub_idx < len(all_capacities):
            caps            = all_capacities[sub_idx]
            master_setsize[sub_idx] = pad_1d(caps, MAX_TRIALS)
            n_trials_mat    = len(caps)
        else:
            print(f"  WARNING: no .mat entry for index {sub_idx}")
            n_trials_mat = None

        # ── Feature arrays ────────────────────────────────────────────────────
        sub_ok       = True
        n_trials_npy = None

        for feat_idx, fname in enumerate(FEATURE_FILES):
            fpath = os.path.join(sub_dir, fname)
            if not os.path.exists(fpath):
                print(f"  MISSING: {fname} — skipping subject")
                sub_ok = False
                break
            try:
                arr = load_npy_as_channels_x_trials(fpath, N_CHANNELS)
                if feat_idx == 0:
                    n_trials_npy = arr.shape[1]
                master_data[sub_idx, feat_idx] = pad_2d(arr, MAX_TRIALS)
            except Exception as e:
                print(f"  ERROR loading {fname}: {e}")
                sub_ok = False
                break

        if not sub_ok:
            failed.append(sub_id)
            trial_counts.append(0)
            continue

        trial_counts.append(n_trials_npy)

        # ── Cross-check ───────────────────────────────────────────────────────
        if n_trials_mat is not None and n_trials_npy != n_trials_mat:
            print(f"  WARNING: npy trials ({n_trials_npy}) != mat trials "
                  f"({n_trials_mat}) — possible subject order mismatch!")
        else:
            print(f"  Trials: {n_trials_npy}  [OK]")

        # Warn on high NaN rates
        for fi, name in enumerate(FEATURE_NAMES):
            chunk = master_data[sub_idx, fi, :, :n_trials_npy]
            pct   = np.isnan(chunk).mean() * 100
            if pct > 50:
                print(f"  WARNING: {name} — {pct:.0f}% NaN")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "master_data.npy"),    master_data)
    np.save(os.path.join(OUT_DIR, "master_setsize.npy"), master_setsize)
    np.save(os.path.join(OUT_DIR, "trial_counts.npy"),   np.array(trial_counts))
    np.savetxt(os.path.join(OUT_DIR, "subject_ids.txt"), subject_ids, fmt="%s")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_counts = [t for t in trial_counts if t]
    print(f"\n{'='*62}")
    print(f"  master_data    : {master_data.shape}")
    print(f"    axis 0  subjects  ({n_found})")
    print(f"    axis 1  features  {FEATURE_NAMES}")
    print(f"    axis 2  channels  ({N_CHANNELS})")
    print(f"    axis 3  trials    ({MAX_TRIALS}, NaN-padded)")
    print(f"  master_setsize : {master_setsize.shape}")
    print(f"    setsize range : {int(np.nanmin(master_setsize))} – "
          f"{int(np.nanmax(master_setsize))}")
    print(f"\n  Overall NaN%   : {np.isnan(master_data).mean()*100:.1f}%")
    print(f"  Trial counts   : min={min(valid_counts)}, "
          f"max={max(valid_counts)}, mean={np.mean(valid_counts):.0f}")
    if failed:
        print(f"\n  Failed ({len(failed)}): {failed}")
    print(f"\n  Saved to ./{OUT_DIR}/")
    print(f"{'='*62}")

    return master_data, master_setsize, subject_ids, trial_counts


# =============================================================================
# QUICK ACCESS RECIPE
# =============================================================================
"""
import numpy as np

master_data    = np.load("master_arrays/master_data.npy")
master_setsize = np.load("master_arrays/master_setsize.npy")
trial_counts   = np.load("master_arrays/trial_counts.npy")
subject_ids    = open("master_arrays/subject_ids.txt").read().splitlines()

# Slice real trials for subject 0 (drop NaN padding)
s = 0
n = trial_counts[s]

theta    = master_data[s, 0, :, :n]   # (55, n_trials)
alpha    = master_data[s, 1, :, :n]   # (55, n_trials)
exponent = master_data[s, 2, :, :n]   # (55, n_trials)
setsize  = master_setsize[s, :n]      # (n_trials,)

# Example: theta on channel 10 for setsize==4 trials only
theta_sz4 == theta[10, setsize == 4]
"""

if __name__ == "__main__":
    build_master_array()
