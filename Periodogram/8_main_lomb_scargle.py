import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip


INPUT_DIR = "star_files_3_csv"
OUTPUT_DIR = "star_plots_2_long"
summary_list = pd.read_csv('Summary of Stars.dat', delimiter=',')
def get_star_data(star_id):
    return summary_list.loc[summary_list["Star"] == star_id]

def extract_star_id(filename):
    """
    Extract numeric star ID from filename.
    Returns int or None if not found.
    """
    m = re.search(r'\d+', filename)
    return int(m.group()) if m else None

MIN_SAMPLES = 20
SIGMA_CLIP = 3.0
MIN_POWER = 0.40
MAX_FAP = 0.01

MIN_PERIOD = 100
MAX_PERIOD = 300

PHASE_BINS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

final_list = []

for filepath in files:

    star = os.path.splitext(os.path.basename(filepath))[0]

    df = pd.read_csv(filepath,delimiter='\t')
    if not {"JD", "Differential_Magnitude", "Differential_Error"} <= set(df.columns):
        continue

    JD = df["JD"].values.astype(float)
    mag = df["Differential_Magnitude"].values.astype(float)
    err = df["Differential_Error"].values.astype(float)

    finite = np.isfinite(JD) & np.isfinite(mag) & np.isfinite(err)
    JD, mag, err = JD[finite], mag[finite], err[finite]

    if len(mag) < MIN_SAMPLES:
        continue

    clipped = sigma_clip(mag, sigma=SIGMA_CLIP, maxiters=5)
    mask = ~clipped.mask
    JD, mag, err = JD[mask], clipped.data[mask], err[mask]

    if len(mag) < MIN_SAMPLES:
        continue

    ls = LombScargle(JD, mag, err)

    frequency, power = ls.autopower(
        minimum_frequency=1.0 / MAX_PERIOD,
        maximum_frequency=1.0 / MIN_PERIOD,
        samples_per_peak=20
    )

    if not np.any(np.isfinite(power)):
        continue

    i_best = np.nanargmax(power)
    best_power = power[i_best]
    if best_power < MIN_POWER:
        continue

    fap = ls.false_alarm_probability(best_power)
    if not np.isfinite(fap) or fap > MAX_FAP:
        continue

    period = 1.0 / frequency[i_best]

    phase = (JD % period) / period

    bins = np.linspace(0, 1, PHASE_BINS + 1)
    which = np.digitize(phase, bins)

    phase_bin, mag_bin = [], []

    for i in range(1, PHASE_BINS + 1):
        m = which == i
        if np.sum(m) >= 3:
            phase_bin.append(np.mean(phase[m]))
            mag_bin.append(np.median(mag[m]))

    if len(phase_bin) < PHASE_BINS // 3:
        continue

    phase_bin = np.array(phase_bin)
    mag_bin = np.array(mag_bin)

    phase2 = np.concatenate([phase, phase + 1])
    mag2 = np.concatenate([mag, mag])
    phaseb2 = np.concatenate([phase_bin, phase_bin + 1])
    magb2 = np.concatenate([mag_bin, mag_bin])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [1, 1.5]}
    )

    ax1.plot(frequency, power, "k", lw=1)
    ax1.axvline(1 / period, color="r", ls="--")
    ax1.set_ylabel("Power")
    ax1.set_xlabel("Frequency (1/day)")
    ax1.set_title(f"{star} | P={period:.3f} d | Power={best_power:.2f}")

    ax2.scatter(
        phase2, mag2,
        s=6, color="black", alpha=0.15, rasterized=True
    )
    #ax2.plot(
     #   phaseb2, magb2,
      #  color="red", lw=2.5
    #)

    ax2.set_xlabel("Phase (0–2)")
    ax2.set_ylabel("Differential Magnitude")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{star}.png"), dpi=150)
    plt.close(fig)

    std_mag = np.std(mag, ddof=1)

    amp = 0.5 * (
        np.percentile(mag, 95) - np.percentile(mag, 5)
    )

    print(f"Saved {star}: P={period:.3f} d")

    star_id = extract_star_id(star)

    x_mean = np.nan
    y_mean = np.nan

    if star_id is not None:
        row = summary_list.loc[summary_list["Star"] == star_id]
        if not row.empty:
            x_mean = row["X_mean"].iloc[0]
            y_mean = row["Y_mean"].iloc[0]

    final_list.append({
        "STAR": star_id,
        "star_name": star,        # keeps filename identity
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "best_period": float(period),
        "amplitude": float(amp),
        "std_mag": float(std_mag)
    })


df_final = pd.DataFrame(final_list)
df_final = df_final.sort_values("STAR")
df_final.round(4).to_csv("Final Data.csv", index=False)

print("Finished.")
