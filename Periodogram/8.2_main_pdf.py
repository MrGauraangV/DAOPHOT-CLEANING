import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from matplotlib.backends.backend_pdf import PdfPages

INPUT_DIR = "star_files_3_csv"
SUMMARY_FILE = "Summary of Stars.dat"
OUTPUT_PDF = "Sinusoidal_Variables.pdf"
OUTPUT_CSV = "Sinusoidal_Variables.csv"
catalog_rows = []

MIN_SAMPLES = 20
SIGMA_CLIP = 3.0
MIN_POWER = 0.40
MAX_FAP = 0.01

MIN_PERIOD = 0.05
MAX_PERIOD = 700.0

PHASE_BINS = 30

summary_list = pd.read_csv(SUMMARY_FILE, delimiter=',')

def extract_star_id(filename):
    m = re.search(r'\d+', filename)
    return int(m.group()) if m else None

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

final_list = []

with PdfPages(OUTPUT_PDF) as pdf:

    for filepath in files:

        star = os.path.splitext(os.path.basename(filepath))[0]

        df = pd.read_csv(filepath, delimiter='\t')
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
            samples_per_peak=10
        )

        i_best = np.argmax(power)
        best_power = power[i_best]
        if best_power < MIN_POWER:
            continue

        fap = ls.false_alarm_probability(best_power)
        if fap > MAX_FAP:
            continue

        period = 1.0 / frequency[i_best]

        # --- harmonic rejection ---
        second_harm_freq = 2 * frequency[i_best]
        if second_harm_freq < frequency.max():
            i2 = np.argmin(np.abs(frequency - second_harm_freq))
            if power[i2] / best_power > 0.5:
                continue

        phase = (JD % period) / period

        bins = np.linspace(0, 1, PHASE_BINS + 1)
        which = np.digitize(phase, bins)

        phase_bin, mag_bin = [], []
        for i in range(1, PHASE_BINS + 1):
            m = which == i
            if np.sum(m) >= 3:
                phase_bin.append(np.mean(phase[m]))
                mag_bin.append(np.median(mag[m]))

        if len(phase_bin) < PHASE_BINS // 2:
            continue

        phase_bin = np.array(phase_bin)
        mag_bin = np.array(mag_bin)

        # --- smoothness test ---
        amp = 0.5 * (np.percentile(mag, 95) - np.percentile(mag, 5))
        curvature = np.std(np.diff(mag_bin)) / amp
        if curvature > 0.8:
            continue

        # --- plotting ---
        phase2 = np.concatenate([phase, phase + 1])
        mag2 = np.concatenate([mag, mag])
        phaseb2 = np.concatenate([phase_bin, phase_bin + 1])
        magb2 = np.concatenate([mag_bin, mag_bin])

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 9),
            gridspec_kw={"height_ratios": [1, 1.5]}
        )

        ax1.plot(frequency, power, "k", lw=1)
        ax1.axvline(1 / period, color="r", ls="--")
        ax1.set_ylabel("Power")
        ax1.set_xlabel("Frequency (1/day)")
        ax1.set_title(f"{star} | P={period:.3f} d")

        ax2.scatter(phase2, mag2, s=6, alpha=0.15)
        #ax2.plot(phaseb2, magb2, color="red", lw=2.5)
        ax2.set_xlabel("Phase (0–2)")
        ax2.set_ylabel("Differential Magnitude")
        ax2.invert_yaxis()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        print(f"Accepted sinusoidal: {star}  P={period:.3f} d")
                # --- catalog bookkeeping ---
        star_id = extract_star_id(star)
        if star_id is not None:
            row = summary_list.loc[summary_list["Star"] == star_id]
            if not row.empty:
                x_mean = row["X_mean"].iloc[0]
                y_mean = row["Y_mean"].iloc[0]

                catalog_rows.append({
                    "star_id": int(star_id),
                    "x_mean": float(x_mean),
                    "y_mean": float(y_mean),
                    "best_period": float(period),
                    "amplitude": float(amp)
                })

if catalog_rows:
    df_out = pd.DataFrame(catalog_rows)
    df_out = df_out.sort_values("star_id")
    df_out.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

print(f"Saved catalog: {OUTPUT_CSV}")

print(f"Saved multi-page PDF: {OUTPUT_PDF}")
