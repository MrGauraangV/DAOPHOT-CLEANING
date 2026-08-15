import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import os

# Load photometry (tab-separated)
df = pd.read_csv("star_files_3_csv/Final_1501_Photometry.csv", sep='\t')
df = df.dropna().sort_values("JD")

# Use differential magnitude directly
t = np.asarray(df["JD"], dtype=float)
mag = np.asarray(df["Differential_Magnitude"], dtype=float)

# Optional: Bin data to ~3000 points to reduce RAM use
def bin_time_series(t, y, n_bins=3000):
    if len(t) <= n_bins:
        return t, y
    bins = np.linspace(t.min(), t.max(), n_bins + 1)
    digitized = np.digitize(t, bins) - 1
    t_bin, y_bin = [], []
    for i in range(n_bins):
        mask = digitized == i
        if not np.any(mask):
            continue
        t_bin.append(np.median(t[mask]))
        y_bin.append(np.median(y[mask]))
    return np.array(t_bin), np.array(y_bin)

t_bin, mag_bin = bin_time_series(t, mag)

# Define period and duration grid
min_period, max_period = 0.1, 50.0
qmin, qmax = 0.002, 0.2
durations = np.linspace(qmin * min_period, qmax * max_period, 40)  # safe duration grid

# Run BLS on binned data
bls = BoxLeastSquares(t_bin, mag_bin)
bls_result = bls.autopower(
    durations,
    minimum_period=min_period,
    maximum_period=max_period,
    oversample=5,  # Keep modest to save RAM
    frequency_factor=2.0,
    objective="snr"
)

# Get best result
i = np.argmax(bls_result.power)
best_period = bls_result.period[i]
best_t0 = bls_result.transit_time[i]
best_dur = bls_result.duration[i]
best_depth = bls_result.depth[i]
best_snr = bls_result.power[i]

print(f"Best period: {best_period:.6f} d")
print(f"Epoch T0: {best_t0:.6f}")
print(f"Duration: {best_dur:.4f} d")
print(f"Depth: {best_depth:.4f} mag")
print(f"SNR (BLS power): {best_snr:.2f}")

# Create output folder
outdir = "bls_outputs"
os.makedirs(outdir, exist_ok=True)

# Save periodogram
plt.figure(figsize=(7, 4))
plt.plot(bls_result.period, bls_result.power, color='black')
plt.axvline(best_period, color='red', linestyle='--', label=f"Best P = {best_period:.4f} d")
plt.xlabel("Period [days]")
plt.ylabel("BLS Power (SNR)")
plt.title("Box Least Squares Periodogram")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "bls_periodogram.png"), dpi=150)
plt.close()

# Fold full (unbinned) light curve at best period
phase = ((t - best_t0 + 0.5 * best_period) % best_period) / best_period - 0.5

plt.figure(figsize=(7, 4))
plt.scatter(phase, mag, s=4, alpha=0.6, color='black')
plt.xlabel("Phase")
plt.ylabel("Differential Magnitude")
plt.title(f"Folded Light Curve (P = {best_period:.4f} d)")
plt.gca().invert_yaxis()
plt.xlim(-0.2, 0.2)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "folded_lightcurve.png"), dpi=150)
plt.close()

print(f"Plots saved to folder: {outdir}")
