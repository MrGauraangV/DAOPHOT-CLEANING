import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.timeseries import BoxLeastSquares

FILE = "star_files_3_csv/Final_1760_Photometry.csv"


df = pd.read_csv(FILE, delimiter="\t")

time = df["JD"].to_numpy()
mag  = df["Differential_Magnitude"].to_numpy()

mask = np.isfinite(time) & np.isfinite(mag)
time = time[mask]
mag  = mag[mask]

flux = 10 ** (-0.4 * mag)
flux /= np.median(flux)

time = time * u.day

MIN_PERIOD = 2    # days
MAX_PERIOD = 3   # days
N_PERIODS  = 2000

periods = np.linspace(MIN_PERIOD, MAX_PERIOD, N_PERIODS) * u.day

# eclipse duration = fraction of period
durations = np.linspace(0.02, 0.15, 20) * periods.min()

bls = BoxLeastSquares(time, flux)
power = bls.power(periods, durations)

best_idx    = np.argmax(power.power)
best_period = power.period[best_idx]
t0          = power.transit_time[best_idx]
best_dur    = power.duration[best_idx]

print("====================================")
print(f"Best period (BLS) = {best_period.value:.6f} days")
print(f"Transit epoch t0 = {t0.value:.6f}")
print(f"Duration         = {best_dur.value:.6f} days")
print("====================================")

def phase_fold(t, P, t0):
    phase = ((t - t0) / P).to_value(u.dimensionless_unscaled)
    return phase - np.floor(phase + 0.5)

fig, axes = plt.subplots(4, 1, figsize=(7, 10))

axes[0].plot(power.period.to_value(u.day), power.power, "k", lw=0.8)
axes[0].axvline(best_period.value, color="r", ls="--")
axes[0].set_xlabel("Period (days)")
axes[0].set_ylabel("BLS Power")
axes[0].set_title("BLS Periodogram")

phase = phase_fold(time, best_period, t0)
axes[1].scatter(phase, flux, s=6, c="k")
axes[1].set_xlim(-0.5, 0.5)
axes[1].set_ylabel("Flux")
axes[1].set_title(f"Phase Folded at P = {best_period.value:.5f} d")

phase2 = phase_fold(time, 2 * best_period, t0)
axes[2].scatter(phase2, flux, s=6, c="k")
axes[2].set_xlim(-0.5, 0.5)
axes[2].set_ylabel("Flux")
axes[2].set_title(f"Phase Folded at 2P = {2*best_period.value:.5f} d")

phase_half = phase_fold(time, 0.5 * best_period, t0)
axes[3].scatter(phase_half, flux, s=6, c="k")
axes[3].set_xlim(-0.5, 0.5)
axes[3].set_xlabel("Phase")
axes[3].set_ylabel("Flux")
axes[3].set_title(f"Phase Folded at P/2 = {0.5*best_period.value:.5f} d")

plt.tight_layout()
plt.show()
