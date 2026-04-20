# -*- coding: utf-8 -*-
"""
Aperiodic-Adjusted Theta & Alpha Power — Per-Trial, Window-Averaged
=====================================================================
Adapted from Bender et al. (2025, PNAS).

Key difference from the sliding-window version:
    Instead of fitting specparam to each of the ~19 windows separately,
    we average all windows into ONE representative PSD per trial per channel,
    then fit specparam once to that averaged spectrum.

    Gain  : 19× fewer specparam fits → much faster; better SNR per fit
    Trade : no time-within-trial resolution (encoding vs. delay period)
    Use when : your analysis is at the trial level (condition comparison,
               RT correlation, single-trial classification, etc.)

Output per trial per channel — scalars, not time-series:
    theta_oscillatory_power  : (n_trials, n_channels)  — 4–8  Hz AUC
    alpha_oscillatory_power  : (n_trials, n_channels)  — 8–12 Hz AUC
    aperiodic_exponent       : (n_trials, n_channels)  — 1/f slope

@author: Rahul Venugopal
"""

import numpy as np
from scipy.integrate import trapezoid
from fooof import FOOOF as SpectralModel
import mne
import os
import glob
from joblib import Parallel, delayed

# =============================================================================
# USER CONFIGURATION
# =============================================================================

SET_FILE_PATH = r"E:\Fooofing WM\Data"
OUT_DIR       = "spectral_params_trial_avg"
N_JOBS        = 28    # cores; leave 2 free for OS

# Spectral settings
WINDOW_SEC  = 1.0     # window length for multitaper estimation
STEP_SEC    = 0.1     # step between windows (only affects how many PSDs
                      # are averaged — does not affect final resolution
                      # since we collapse across windows anyway)
FREQ_MIN    = 2.0
FREQ_MAX    = 50.0
N_FREQ_BINS = 128

THETA_BAND = (4.0,  8.0)
ALPHA_BAND = (8.0, 12.0)

FOOOF_SETTINGS = dict(
    peak_width_limits = (2, 8),
    max_n_peaks       = 4,
    min_peak_height   = 0,
    peak_threshold    = 2.0,
    aperiodic_mode    = "fixed",
)

# =============================================================================
# LOAD
# =============================================================================

def load_set_file(path: str) -> mne.Epochs:
    print(f"\nLoading: {path}")
    epochs   = mne.io.read_epochs_eeglab(path, verbose=False)
    sfreq    = epochs.info["sfreq"]
    duration = epochs.times[-1] - epochs.times[0]
    print(f"  Channels={len(epochs.ch_names)} | Trials={len(epochs)} | "
          f"Duration={duration:.2f}s | sfreq={sfreq} Hz")
    if duration < WINDOW_SEC:
        raise ValueError(f"Epoch ({duration:.2f}s) shorter than window ({WINDOW_SEC}s).")
    return epochs

# =============================================================================
# STEP 1 — Average multitaper PSD across all windows  (one trial, all channels)
# =============================================================================

def compute_mean_psd(
    trial_data : np.ndarray,   # (n_channels, n_times)
    sfreq      : float,
    freqs      : np.ndarray,   # (n_freqs,)  target frequency axis
) -> np.ndarray:
    """
    Slide a multitaper window across the trial, compute PSD at each position,
    then return the MEAN across all windows.

    Averaging before fitting specparam:
        - increases SNR (random noise averages out across windows)
        - makes the 1/f slope and any peaks more stable and representative
        - reduces specparam fits from n_windows → 1 per channel per trial

    Returns
    -------
    mean_psd : ndarray (n_channels, n_freqs)  — trial-representative PSD
    n_windows_used : int  — number of windows that went into the average
    """
    n_channels, n_times = trial_data.shape
    win_samples  = int(WINDOW_SEC  * sfreq)
    step_samples = max(1, int(STEP_SEC * sfreq))
    starts       = np.arange(0, n_times - win_samples + 1, step_samples)

    # Accumulate PSDs across windows
    psd_accumulator = np.zeros((n_channels, len(freqs)))

    for start in starts:
        seg = trial_data[:, start : start + win_samples]
        psd_native, freqs_native = mne.time_frequency.psd_array_multitaper(
            seg, sfreq=sfreq, fmin=FREQ_MIN, fmax=FREQ_MAX,
            bandwidth=2.0, adaptive=False, low_bias=True,
            normalization="full", verbose=False,
        )
        for ch in range(n_channels):
            psd_accumulator[ch] += np.interp(freqs, freqs_native, psd_native[ch])

    mean_psd = psd_accumulator / len(starts)   # (n_channels, n_freqs)
    mean_psd = mean_psd * 1e12
    return mean_psd, len(starts)

# =============================================================================
# STEP 2 — Fit specparam once per channel to the averaged PSD
# =============================================================================

def parameterize_mean_psd(
    mean_psd : np.ndarray,
    freqs    : np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_channels = mean_psd.shape[0]
    aperiodic_exponent = np.full(n_channels, np.nan)
    residual_psd       = np.zeros_like(mean_psd)
    n_failed           = 0

    fm = SpectralModel(**FOOOF_SETTINGS, verbose=False)

    for ch in range(n_channels):
        try:
            fm.fit(freqs, mean_psd[ch], freq_range=[FREQ_MIN, FREQ_MAX])
            offset, exponent       = fm.aperiodic_params_
            aperiodic_exponent[ch] = exponent

            # Subtract in log space — where specparam actually fits
            log_psd = np.log10(np.maximum(mean_psd[ch], 1e-30))
            log_ap  = offset - exponent * np.log10(freqs)
            residual_psd[ch] = np.clip(log_psd - log_ap, 0, None)

        except Exception as e:
            n_failed += 1

    if n_failed:
        print(f"  WARNING: specparam failed on {n_failed}/{n_channels} channels "
              f"(residual left as zeros for those channels)")

    return aperiodic_exponent, residual_psd

# =============================================================================
# STEP 3 — Band AUC on residual PSD  (now 1-D per channel, not time-series)
# =============================================================================

def compute_band_auc(
    residual_psd : np.ndarray,          # (n_channels, n_freqs)
    freqs        : np.ndarray,
    band         : tuple[float, float],
) -> np.ndarray:
    """
    AUC of aperiodic-adjusted PSD in band (linear scale).

    Returns
    -------
    band_power : (n_channels,)  — one scalar per channel per trial
    """
    lo, hi = band
    mask   = (freqs >= lo) & (freqs <= hi)
    return trapezoid(residual_psd[:, mask], x=freqs[mask], axis=-1)

# =============================================================================
# Per-trial worker  (called in parallel across trials)
# =============================================================================

def process_one_trial(
    trial_data : np.ndarray,   # (n_channels, n_times)
    sfreq      : float,
    freqs      : np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline for one trial.

    Returns
    -------
    theta : (n_channels,)
    alpha : (n_channels,)
    exp   : (n_channels,)
    """
    # Step 1 — average PSDs across windows
    mean_psd, n_win = compute_mean_psd(trial_data, sfreq, freqs)

    # Step 2 — one specparam fit per channel
    ap_exp, resid = parameterize_mean_psd(mean_psd, freqs)

    # Step 3 — band AUC
    theta = compute_band_auc(resid, freqs, THETA_BAND)
    alpha = compute_band_auc(resid, freqs, ALPHA_BAND)

    return theta, alpha, ap_exp

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(epochs: mne.Epochs) -> dict:
    sfreq      = epochs.info["sfreq"]
    n_channels = len(epochs.ch_names)
    data       = epochs.get_data()       # (n_trials, n_ch, n_times)
    n_trials   = data.shape[0]
    freqs      = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ_BINS)

    # Quick check: how many windows will be averaged?
    win_samples  = int(WINDOW_SEC  * sfreq)
    step_samples = max(1, int(STEP_SEC * sfreq))
    n_win        = len(np.arange(0, data.shape[2] - win_samples + 1, step_samples))

    print(f"  {n_trials} trials | {n_channels} channels | "
          f"{n_win} windows averaged per trial | {N_JOBS} cores")
    print(f"  specparam fits: {n_trials * n_channels} total "
          f"(vs {n_trials * n_channels * n_win} in sliding-window version)")

    # Parallel across trials
    results_list = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(process_one_trial)(data[t], sfreq, freqs)
        for t in range(n_trials)
    )

    # Stack → (n_trials, n_channels)
    all_theta = np.stack([r[0] for r in results_list], axis=0)
    all_alpha = np.stack([r[1] for r in results_list], axis=0)
    all_exp   = np.stack([r[2] for r in results_list], axis=0)

    print(f"  Done. Output shape: {all_theta.shape}  "
          f"[n_trials × n_channels]")

    return {
        "theta_oscillatory_power" : all_theta,   # (n_trials, n_channels)
        "alpha_oscillatory_power" : all_alpha,   # (n_trials, n_channels)
        "aperiodic_exponent"      : all_exp,     # (n_trials, n_channels)
        "freqs"                   : freqs,
        "ch_names"                : epochs.ch_names,
        "epoch_times"             : epochs.times,
    }

# =============================================================================
# BATCH ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    set_files = sorted(glob.glob(os.path.join(SET_FILE_PATH, "*.set")))
    if not set_files:
        raise FileNotFoundError(f"No .set files found in: {SET_FILE_PATH}")

    print(f"Found {len(set_files)} .set files | {N_JOBS} cores\n")

    failed = []

    for sub_idx, set_path in enumerate(set_files):

        sub_id      = os.path.splitext(os.path.basename(set_path))[0]
        sub_out_dir = os.path.join(OUT_DIR, sub_id)
        os.makedirs(sub_out_dir, exist_ok=True)

        print(f"\n{'='*62}")
        print(f"  Subject {sub_idx + 1}/{len(set_files)} : {sub_id}")
        print(f"{'='*62}")

        marker = os.path.join(sub_out_dir, "alpha_oscillatory_power.npy")
        if os.path.exists(marker):
            print("  Already processed — skipping.")
            continue

        try:
            epochs  = load_set_file(set_path)
            results = run_pipeline(epochs)

            save_map = {
                "theta_oscillatory_power" : results["theta_oscillatory_power"],
                "alpha_oscillatory_power" : results["alpha_oscillatory_power"],
                "aperiodic_exponent"      : results["aperiodic_exponent"],
                "freqs"                   : results["freqs"],
            }
            for fname, arr in save_map.items():
                np.save(os.path.join(sub_out_dir, f"{fname}.npy"), arr)
                print(f"  Saved {fname}.npy  {arr.shape}")

            with open(os.path.join(sub_out_dir, "ch_names.txt"), "w") as f:
                f.write("\n".join(results["ch_names"]))

        except Exception as e:
            print(f"\n  ERROR on {sub_id}: {e}")
            failed.append((sub_id, str(e)))
            continue

    print(f"\n{'='*62}")
    print(f"  Finished. "
          f"{len(set_files) - len(failed)}/{len(set_files)} subjects saved "
          f"to ./{OUT_DIR}/")
    if failed:
        print(f"\n  Failed subjects:")
        for sub_id, err in failed:
            print(f"    {sub_id}: {err}")