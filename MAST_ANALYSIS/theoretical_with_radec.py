import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------
INPUT_FILE = "radec_with_tic_ids.csv"
OUTPUT_DIR = "star_plots_2"
SUMMARY_CSV = "ls_summary_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []

# --------------------------------------------------
# LOAD STAR LIST (RA/DEC in sexagesimal)
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE, delimiter=',').loc[874:876, :]

star_ids = df["star_id"].values
ra_list  = df["ra"].astype(str).values
dec_list = df["dec"].astype(str).values

# --------------------------------------------------
# LOOP OVER STARS
# --------------------------------------------------
for star_id, ra, dec in zip(star_ids, ra_list, dec_list):

    print(f"\nProcessing {star_id} @ {ra} {dec}")

    coord = SkyCoord(
        ra=ra,
        dec=dec,
        unit=(u.hourangle, u.deg),
        frame="icrs"
    )

    search = lk.search_lightcurve(
        coord,
        radius=0.75 * u.arcsec,
        mission=("Kepler", "K2", "TESS")
    )

    if len(search) == 0:
        print("  No data found")
        continue

    # --------------------------------------------------
    # LOOP OVER EACH PRODUCT (NO GROUPING)
    # --------------------------------------------------
    for i, row in enumerate(search.table):

        author  = row["author"]
        mission = row["mission"]
        target_name = row["target_name"]
        tic_id = target_name if "TIC" in str(target_name) else "Unknown TIC"

        print(f"  Downloading: {mission} | {author}")

        try:
            lc_raw = search[i].download()
            if lc_raw is None:
                continue

            # --------------------------------------------------
            # FORCE NUMERIC ARRAYS (CDIPS-SAFE)
            # --------------------------------------------------
            time = np.asarray(lc_raw.time.value, dtype=float)
            flux = np.asarray(lc_raw.flux.value, dtype=float)

            mask = np.isfinite(time) & np.isfinite(flux)
            if mask.sum() < 20:
                raise ValueError("Too few valid points after masking")

            lc = lk.LightCurve(
                time=time[mask] * u.day,
                flux=flux[mask]
            ).normalize()

            lc.flux_err = np.full(len(lc.flux), np.nan)

            # --------------------------------------------------
            # LOMB–SCARGLE
            # --------------------------------------------------
            pg = lc.to_periodogram(method="lombscargle")
            best_period = pg.period_at_max_power

            # --------------------------------------------------
            # TIME METADATA
            # --------------------------------------------------
            time_days = lc.time.value
            cadence = np.nanmedian(np.diff(time_days)) * 24 * 60
            time_span = time_days.max() - time_days.min()
            median_year = int(np.nanmedian(lc.time.decimalyear))

            # --------------------------------------------------
            # PLOTTING
            # --------------------------------------------------
            fig, axes = plt.subplots(
                3, 2,
                figsize=(11, 10),
                gridspec_kw={"width_ratios": [3.5, 1.8]}
            )

            # --- Raw Light Curve ---
            lc.plot(
                ax=axes[0, 0],
                marker='o',
                lw=0,
                ms=1,
                alpha=0.3
            )
            axes[0, 0].set_title(
                f"{star_id} | {tic_id}\n{mission} | {author} | Year ≈ {median_year}"
            )

            # --- Periodogram ---
            pg.plot(ax=axes[1, 0])
            axes[1, 0].axvline(best_period.value, color="red", lw=2)

            # --- Phase (P) ---
            lc_phased = lc.fold(best_period, normalize_phase=True)

            # background (unbinned)
            axes[2, 0].scatter(
                lc_phased.phase.value,
                lc_phased.flux.value,
                s=6,
                color="gray",
                alpha=0.25,
                rasterized=True
            )

            # binned
            phase = (lc_phased.phase.value + 0.5) % 1.0 - 0.5
            bin_width = 1.0 / np.sqrt(len(phase))
            lc_bin = lc_phased.bin(bin_width)

            axes[2, 0].scatter(
                lc_bin.phase.value,
                lc_bin.flux.value,
                s=28,
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.4,
                zorder=3
            )

            axes[2, 0].set_xlim(-0.5, 0.5)
            axes[2, 0].set_title(f"P = {best_period.value:.5f} d")

            y = lc_bin.flux.value
            pad = 0.2 * (y.max() - y.min())
            axes[2, 0].set_ylim(y.min() - pad, y.max() + pad)

            # --- Phase (2P) ---
            lc.fold(2 * best_period, normalize_phase=True).plot(
                ax=axes[1, 1],
                marker='.',
                lw=0,
                ms=2,
                alpha=0.3
            )
            axes[1, 1].set_xlim(-0.5, 0.5)
            axes[1, 1].set_title("2 × P")

            # --- Phase (P/2) ---
            lc.fold(0.5 * best_period, normalize_phase=True).plot(
                ax=axes[2, 1],
                marker='.',
                lw=0,
                ms=2,
                alpha=0.3
            )
            axes[2, 1].set_xlim(-0.5, 0.5)
            axes[2, 1].set_title("P / 2")

            plt.tight_layout()

            outname = f"{star_id}_{tic_id}_{mission}_{author}_Y{median_year}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=200)
            plt.close()

            # --------------------------------------------------
            # SAVE METADATA
            # --------------------------------------------------
            summary_rows.append({
                "star_id": star_id,
                "ra": ra,
                "dec": dec,
                "tic_id": tic_id,
                "mission": mission,
                "author": author,
                "year": median_year,
                "n_points": len(lc),
                "time_span_days": time_span,
                "best_period_days": best_period.value,
                "max_power": pg.max_power.value,
                "cadence_minutes": cadence,
                "plot_filename": outname,
                "status": "SUCCESS",
                "error_message": ""
            })

            print("    Success")

        except Exception as e:
            print(f"    Failed - {e}")
            plt.close("all")

            summary_rows.append({
                "star_id": star_id,
                "ra": ra,
                "dec": dec,
                "tic_id": tic_id,
                "mission": mission,
                "author": author,
                "year": np.nan,
                "n_points": np.nan,
                "time_span_days": np.nan,
                "best_period_days": np.nan,
                "max_power": np.nan,
                "cadence_minutes": np.nan,
                "plot_filename": "",
                "status": "FAILED",
                "error_message": str(e)
            })

# --------------------------------------------------
# SAVE SUMMARY CSV
# --------------------------------------------------
pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

print("\nAll targets processed. Summary saved.")
