import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
INPUT_FILE = "star_files_3_csv/Final_1568_Photometry.csv"
OUTPUT_DIR = "bls_phase_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_FILE, delimiter='\t')
df = df.dropna().sort_values('JD')

t = np.asarray(df['JD'], dtype=float)
mag = np.asarray(df['Differential_Magnitude'], dtype=float)
mag_err = np.asarray(df['Differential_Error'], dtype=float)

# Convert magnitudes to normalized flux
flux = 10**(-0.4 * mag)
flux /= np.median(flux)

# Error propagation
flux_err = flux * (0.4 * np.log(10)) * mag_err
flux_err /= np.median(flux)

# ------------------------------------------------------------
# 2. Box Least Squares
# ------------------------------------------------------------
bls = BoxLeastSquares(t, flux, dy=flux_err)

durations = np.linspace(0.1, 0.4, 15)

results = bls.autopower(
    durations,
    minimum_period=0.5,
    maximum_period=10.0,
    objective="snr"
)

# ------------------------------------------------------------
# 3. Best solution
# ------------------------------------------------------------
idx = np.argmax(results.power)

best_period = results.period[idx]
best_duration = results.duration[idx]
t0 = results.transit_time[idx]
depth = results.depth[idx]
snr = results.power[idx]

print("===== BLS RESULTS =====")
print(f"Orbital period        : {best_period:.5f} days")
print(f"Eclipse duration      : {best_duration:.3f} days")
print(f"Reference epoch (T0)  : {t0:.5f} JD")
print(f"Eclipse depth         : {depth:.5f}")
print(f"BLS SNR               : {snr:.2f}")

# ------------------------------------------------------------
# 4. Phase folding (two full cycles)
# ------------------------------------------------------------
phase = ((t - t0) / best_period) % 1.0

# Center primary eclipse at phase = 0.5
phase = (phase + 0.5) % 1.0

# Duplicate ALL points for 0–2 phase
phase2 = np.concatenate([phase, phase + 1.0])
flux2 = np.concatenate([flux, flux])

# ------------------------------------------------------------
# 5. Plot and save
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(phase2, flux2, s=6, alpha=0.6)

plt.xlabel("Phase")
plt.ylabel("Normalized Flux")
plt.xlim(0.0, 2.0)
plt.title(f"BLS phase-folded light curve (P = {best_period:.3f} d)")

plt.tight_layout()

outfile = os.path.join(
    OUTPUT_DIR,
    f"Final_1568_BLS_P{best_period:.4f}d.png"
)

plt.savefig(outfile, dpi=150)
plt.close()

print(f"Saved phase-folded plot to: {outfile}")
