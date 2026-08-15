import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
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
# ALLOWED AUTHORS (STRICT)
# --------------------------------------------------
ALLOWED_AUTHORS = {
    "TGLC",
    "GSFC-ELEANOR-LITE"
}

# --------------------------------------------------
# LOAD TIC LIST
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

star_ids = df["star_id"].values
tic_ids  = df["TIC_ID"].values

# --------------------------------------------------
# LOOP OVER STARS
# --------------------------------------------------
for star_id, tic in zip(star_ids, tic_ids):

    if pd.isna(tic):
        print(f"SKIPPED {star_id}: no TIC ID")
        continue

    TIC = f"TIC {int(tic)}"
    print(f"\nProcessing {star_id} → {TIC}")

    search = lk.search_lightcurve(TIC)
    if len(search) == 0:
        print("  No MAST data found")
        continue

    # --------------------------------------------------
    # LOOP OVER EACH PRODUCT (NO STITCHING)
    # --------------------------------------------------
    for i, row in enumerate(search.table):

        author  = row["author"]
        mission = row["mission"]

        # ---- STRICT AUTHOR FILTER ----
        if author not in ALLOWED_AUTHORS:
            print(f"  Skipped author: {author}")
            continue

        print(f"  Using author: {author}")

        try:
            lc = search[i].download()
            if lc is None:
                continue

            lc = lc.remove_nans().normalize()
            lc.flux_err = np.full(len(lc.flux), np.nan)

            if len(lc) < 30:
                raise ValueError("Too few data points")

            # --------------------------------------------------
            # TIME METADATA
            # --------------------------------------------------
            time_days = lc.time.value
            cadence = np.nanmedian(np.diff(time_days)) * 24 * 60
            time_span = time_days.max() - time_days.min()
            year = int(np.nanmedian(lc.time.decimalyear))

            # --------------------------------------------------
            # LOMB–SCARGLE
            # --------------------------------------------------
            pg = lc.to_periodogram(method="lombscargle")
            best_period = pg.period_at_max_power

            # --------------------------------------------------
            # PLOTTING
            # --------------------------------------------------
            fig, axes = plt.subplots(
                3, 2,
                figsize=(11, 10),
                gridspec_kw={"width_ratios": [3.5, 1.8]}
            )

            # --- Raw LC
            lc.plot(
                ax=axes[0, 0],
                marker="o",
                lw=0,
                ms=1,
                alpha=0.25
            )
            axes[0, 0].set_title(
                f"{star_id} | {TIC}\n{author} | Year ≈ {year}"
            )

            # --- Periodogram
            pg.plot(ax=axes[1, 0])
            axes[1, 0].axvline(best_period.value, color="red", lw=2)

            # --- Phase (P)
            lc_phased = lc.fold(best_period, normalize_phase=True)

            axes[2, 0].scatter(
                lc_phased.phase.value,
                lc_phased.flux.value,
                s=6,
                alpha=0.25,
                color="gray"
            )

            bin_width = 1.0 / np.sqrt(len(lc_phased))
            lc_bin = lc_phased.bin(bin_width)

            axes[2, 0].scatter(
                lc_bin.phase.value,
                lc_bin.flux.value,
                s=25,
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.4
            )

            axes[2, 0].set_xlim(-0.5, 0.5)
            axes[2, 0].set_title(f"P = {best_period.value:.5f} d")

            y = lc_bin.flux.value
            y = y[np.isfinite(y)]
            if len(y) > 1:
                pad = 0.3 * (y.max() - y.min())
                axes[2, 0].set_ylim(y.min() - pad, y.max() + pad)

            # --- Phase (2P)
            lc.fold(2 * best_period, normalize_phase=True).plot(
                ax=axes[1, 1],
                marker=".",
                lw=0,
                ms=2,
                alpha=0.3
            )
            axes[1, 1].set_xlim(-0.5, 0.5)
            axes[1, 1].set_title("2 × P")

            # --- Phase (P/2)
            lc.fold(0.5 * best_period, normalize_phase=True).plot(
                ax=axes[2, 1],
                marker=".",
                lw=0,
                ms=2,
                alpha=0.3
            )
            axes[2, 1].set_xlim(-0.5, 0.5)
            axes[2, 1].set_title("P / 2")

            plt.tight_layout()

            outname = f"{star_id}_TIC{int(tic)}_{author}_Y{year}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=200)
            plt.close()

            # --------------------------------------------------
            # SAVE METADATA
            # --------------------------------------------------
            summary_rows.append({
                "star_id": star_id,
                "TIC_ID": int(tic),
                "author": author,
                "year": year,
                "mission": mission,
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
            print(f"    Failed – {e}")
            plt.close("all")

            summary_rows.append({
                "star_id": star_id,
                "TIC_ID": int(tic),
                "author": author,
                "year": np.nan,
                "mission": mission,
                "n_points": np.nan,
                "time_span_days": np.nan,
                "best_period_days": np.nan,
                "max_power": np.nan,
                "cadence_minutes": np.nan,
                "plot_filename": "",
                "status": "FAILED",
                "error_message": str(e)
            })


pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
print("\nAll TIC targets processed. Summary saved.")
