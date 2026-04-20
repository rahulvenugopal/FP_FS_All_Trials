# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:55:14 2026

@author: Admin
"""

# ── Paste this and run on ONE trial to see the real error ──────────────────
import numpy as np
import mne
from fooof import FOOOF as SpectralModel

epochs = mne.io.read_epochs_eeglab(r"E:\Fooofing WM\Data\abanti.set",
                                    verbose=False)
data  = epochs.get_data()
sfreq = epochs.info["sfreq"]
freqs = np.linspace(2.0, 50.0, 128)

# Grab trial 0, channel 0
trial = data[0]                   # (n_ch, n_times)
win   = int(1.0 * sfreq)
seg   = trial[:, :win]

psd_native, freqs_native = mne.time_frequency.psd_array_multitaper(
    seg, sfreq=sfreq, fmin=2.0, fmax=50.0,
    bandwidth=2.0, adaptive=False, low_bias=True,
    normalization="full", verbose=False,
)
psd_ch0 = np.interp(freqs, freqs_native, psd_native[1]) * 1e12   # → µV²/Hz

print("PSD range:", psd_ch0.min(), "to", psd_ch0.max())
print("Any NaN?", np.isnan(psd_ch0).any())
print("Any zero/negative?", (psd_ch0 <= 0).any())

# Now try fitting directly — NO try/except so the real error surfaces
fm = SpectralModel(peak_width_limits=(2,8), max_n_peaks=4,
                   min_peak_height=0, peak_threshold=2.0,
                   aperiodic_mode="fixed", verbose=True)
fm.fit(freqs, psd_ch0, freq_range=[2.0, 50.0])
fm.print_results()