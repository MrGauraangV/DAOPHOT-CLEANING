import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
import warnings

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.visualization import ZScaleInterval

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "star_files_3_dat"
FILE_PATTERN = "Final_*_Photometry.dat"
OUTPUT_DIR = "star_plots_1"

MIN_PERIOD = 0.05    # days
MAX_PERIOD = 50.0    # days

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)))
print(f"Found {len(files)} files")

# ============================================================
# LOAD CATALOG
# ============================================================
catalog = pd.read_csv(
    "All_Data_RADEC_sexagesimal.dat",
    delimiter="\t"
)
catalog["star_id"] = catalog["star_id"].astype(str)

# ============================================================
# LOAD FITS IMAGE
# ============================================================
with fits.open("RA Dec Reference Image.fits") as hdul:
    fits_data = hdul[0].data

# ============================================================
# MAIN LOOP (ONE STAR = ONE TRY)
# ============================================================
for file in files:
    try:
        star_id = os.path.basename(file).split("_")[1]
        print(f"\nProcessing: {os.path.basename(file)}")

        # ------------------------------------------------------------
        # Match catalog
        # ------------------------------------------------------------
        row = catalog[catalog["star_id"] == star_id]
        if row.empty:
            raise ValueError("Star not found in catalog")

        x_star = row.iloc[0]["x_mean"]
        y_star = row.iloc[0]["y_mean"]
        ra_star = row.iloc[0]["ra"]
        dec_star = row.iloc[0]["dec"]

        # ------------------------------------------------------------
        # FITS cutout
        # ------------------------------------------------------------
        cutout_size = 40

        cutout = Cutout2D(
            data=fits_data,
            position=(x_star, y_star),
            size=(cutout_size, cutout_size),
            mode="partial"
        )

        vmin, vmax = ZScaleInterval().get_limits(cutout.data)

        # ------------------------------------------------------------
        # Load photometry
        # ------------------------------------------------------------
        df = pd.read_csv(file, delimiter="\t")

        time = df["JD"].to_numpy()
        mag = df["Differential_Magnitude"].to_numpy()
        mag_err = df["Differential_Error"].to_numpy()

        mask = np.isfinite(time) & np.isfinite(mag)
        time = time[mask]
        mag = mag[mask]
        mag_err = mag_err[mask]

        if len(time) < 2:
            raise ValueError("Too few raw points")

        flux = 10 ** (-0.4 * mag)
        flux_err = flux * (0.4 * np.log(10)) * mag_err
        flux /= np.median(flux)

        lc = lk.LightCurve(
            time=time * u.day,
            flux=flux,
            flux_err=flux_err
        ).remove_nans()

        if len(lc.time) < 2:
            raise ValueError("Too few valid points after cleaning")

        baseline = lc.time.value.max() - lc.time.value.min()
        if baseline <= 0:
            raise ValueError("Zero time baseline")

        # ------------------------------------------------------------
        # Lomb–Scargle
        # ------------------------------------------------------------
        pg = lc.to_periodogram(
            method="lombscargle",
            minimum_period=MIN_PERIOD * u.day,
            maximum_period=MAX_PERIOD * u.day,
            oversample_factor=10
        )

        best_period = pg.period_at_max_power
        print(f"  Best period: {best_period:.5f}")

        # ============================================================
        # PLOTTING
        # ============================================================
        fig, axes = plt.subplots(
            3, 2,
            figsize=(11, 10),
            gridspec_kw={"width_ratios": [3.5, 1.8]},
            sharex=False
        )

        # ---------------- Panel 1: LS ----------------
        axes[0, 0].plot(pg.period.value, pg.power, "k", lw=0.8)
        axes[0, 0].axvline(best_period.value, color="#4682B4", lw=2)
        axes[0, 0].set_xlabel("Period (days)")
        axes[0, 0].set_ylabel("LS Power")
        axes[0, 0].set_title(os.path.basename(file))

        # ---------------- Panel 2: Raw phase ----------------
        lc_fold = lc.fold(period=best_period, normalize_phase=True)
        axes[1, 0].scatter(
            lc_fold.phase.value,
            lc_fold.flux.value,
            s=8,
            color="#4682B4",
            alpha=0.4
        )
        axes[1, 0].set_xlabel("Phase")
        axes[1, 0].set_ylabel("Normalized flux")

        # ---------------- Panel 3: Main phased ----------------
        phase = (lc_fold.phase.value + 0.5) % 1.0 - 0.5
        bin_width = 1.0 / np.sqrt(len(phase))

        lc_phase = lk.LightCurve(time=phase, flux=lc_fold.flux.value)
        lc_phase_binned = lc_phase.bin(bin_width)

        axes[2, 0].plot(
            phase, lc_fold.flux.value, ".", color="#4682B4", alpha=0.25, ms=3
        )
        axes[2, 0].plot(
            lc_phase_binned.time.value,
            lc_phase_binned.flux.value,
            "o", color="black", ms=3
        )
        axes[2, 0].set_xlim(-0.5, 0.5)
        axes[2, 0].set_ylim(0.8, 1.3)
        axes[2, 0].set_xlabel("Phase")
        axes[2, 0].set_ylabel("Normalized flux")
        axes[2, 0].set_title(f"P = {best_period.value:.5f} d")

        # ---------------- Right panels ----------------
        for ax, per, title in [
            (axes[1, 1], best_period / 2, "P / 2"),
            (axes[2, 1], best_period * 2, "2 × P"),
        ]:
            lc_tmp = lc.fold(period=per, normalize_phase=True)
            ph = (lc_tmp.phase.value + 0.5) % 1.0 - 0.5
            lc_tmp_phase = lk.LightCurve(time=ph, flux=lc_tmp.flux.value)
            lc_tmp_bin = lc_tmp_phase.bin(bin_width)

            ax.plot(ph, lc_tmp.flux.value, ".", color="#4682B4", alpha=0.25, ms=2)
            ax.plot(lc_tmp_bin.time.value, lc_tmp_bin.flux.value, "o", color="black", ms=2)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0.8, 1.3)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        # ---------------- FITS image ----------------
        ax_img = axes[0, 1]
        ax_img.imshow(cutout.data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.scatter(
            cutout.data.shape[1] / 2,
            cutout.data.shape[0] / 2,
            s=40,
            facecolors="none",
            edgecolors="red"
        )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title("Reference image")

        ax_img.text(
            0.02, 0.98,
            f"ID: {star_id}\nRA: {ra_star}\nDec: {dec_star}\nx,y=({x_star:.1f},{y_star:.1f})",
            transform=ax_img.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
        )

        plt.tight_layout()
        outname = os.path.basename(file).replace(".dat", "_LS.png")
        plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=200)
        plt.close()

    except Exception as e:
        print(f"  SKIPPED {os.path.basename(file)} → {e}")
        plt.close("all")
        continue

print("\nAll files processed.")
