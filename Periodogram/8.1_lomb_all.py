import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip

# ----------------------------
# Configuration (tweak as needed)
# ----------------------------
INPUT_DIR = "star_files_3_csv"
OUTPUT_DIR = "star_plots_1_long"
SUMMARY_FILE = "Summary of Stars.dat"

MIN_SAMPLES = 20
SIGMA_CLIP = 3.0

MIN_PERIOD = 0.05       # days (minimum period to search)
MAX_PERIOD = 700.0      # days (maximum period to attempt but will be reduced to data baseline)
PHASE_BINS = 30

# Candidate selection thresholds
FAP_THRESHOLD = 0.01    # accepted false alarm probability (lower = more significant)
R2_THRESHOLD = 0.30     # weighted R^2 for sinusoidal fit (0..1)
AMP_SNR = 3.0           # amplitude must be >= AMP_SNR * median_error

# Detrend options
DETREND = False         # set True if your long-period targets are dominated by linear trends
DETREND_WINDOW_DAYS = None  # not used in this minimal version; currently linear detrend only

# Output
ALL_TABLE = "All_Final_Data.csv"
CANDIDATES_TABLE = "Candidates.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def extract_star_id(filename):
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else None

def weighted_mean(y, w):
    return np.sum(w * y) / np.sum(w)

def weighted_r2(y, model, w):
    # Weighted R^2: 1 - SS_res / SS_tot
    mu = weighted_mean(y, w)
    ss_res = np.sum(w * (y - model) ** 2)
    ss_tot = np.sum(w * (y - mu) ** 2)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def fit_sinusoid(t, y, dy, period):
    """
    Fit y(t) = A*sin(2pi t/P) + B*cos(2pi t/P) + C using weighted least squares.
    Returns amplitude = sqrt(A^2+B^2), offset C, weighted R^2, fitted model
    """
    omega = 2.0 * np.pi / period
    s = np.sin(omega * t)
    c = np.cos(omega * t)
    X = np.vstack([s, c, np.ones_like(t)]).T
    if np.any(dy <= 0) or np.isnan(dy).any():
        w = np.ones_like(dy)
    else:
        w = 1.0 / (dy ** 2)
    # sqrt weights for stable lstsq
    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    yw = y * Wsqrt
    try:
        coefs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except Exception:
        return np.nan, np.nan, 0.0, np.full_like(y, np.nan)
    A, B, C = coefs
    model = (A * s + B * c + C)
    amp = np.hypot(A, B)
    r2 = weighted_r2(y, model, w)
    return amp, C, r2, model

# ----------------------------
# File list and summary lookup
# ----------------------------
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
try:
    summary_list = pd.read_csv(SUMMARY_FILE, delimiter=",")
except Exception:
    summary_list = pd.DataFrame(columns=["Star", "X_mean", "Y_mean"])

final_list = []
candidates = []

# ----------------------------
# Main loop
# ----------------------------
for filepath in files:
    star = os.path.splitext(os.path.basename(filepath))[0]

    # load with tab delimiter fallback to comma
    try:
        df = pd.read_csv(filepath, delimiter="\t")
    except Exception:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            print(f"Skipped (cannot read): {filepath}")
            continue

    # required columns
    if not {"JD", "Differential_Magnitude", "Differential_Error"} <= set(df.columns):
        print(f"Skipped (missing columns): {star}")
        continue

    JD = df["JD"].astype(float).values
    mag = df["Differential_Magnitude"].astype(float).values
    err = df["Differential_Error"].astype(float).values

    finite = np.isfinite(JD) & np.isfinite(mag) & np.isfinite(err)
    JD, mag, err = JD[finite], mag[finite], err[finite]

    if len(mag) < MIN_SAMPLES:
        print(f"Skipped (too few points): {star}")
        continue

    # sigma-clip outliers in magnitude (use masked result)
    clipped = sigma_clip(mag, sigma=SIGMA_CLIP, maxiters=5)
    mask = ~clipped.mask
    JD, mag, err = JD[mask], clipped.data[mask], err[mask]

    if len(mag) < MIN_SAMPLES:
        print(f"Skipped (too few after clip): {star}")
        continue

    # center time
    t = JD.astype(float)
    t0 = t.min()
    t = t - t0
    t_span = t.max() - t.min()
    if t_span <= 0:
        print(f"Skipped (zero timespan): {star}")
        continue

    # optionally detrend linear trend (useful if there is linear instrumental drift)
    if DETREND:
        # simple linear fit and subtract
        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, mag, rcond=None)
        trend = A.dot(coef)
        mag = mag - trend

    # sampling estimate for Nyquist-like upper frequency bound
    dt = np.diff(np.sort(t))
    if len(dt) == 0:
        median_dt = t_span
    else:
        median_dt = np.median(dt[dt > 0]) if np.any(dt > 0) else t_span

    # compute sensible period search bounds based on the data
    max_period_data = min(MAX_PERIOD, t_span * 0.95)  # don't allow searching periods longer than data span
    if max_period_data <= MIN_PERIOD:
        print(f"Skipped (data span too short for MIN_PERIOD): {star}")
        continue

    # frequency boundaries
    min_frequency = 1.0 / max_period_data
    # set a safe nyquist-like upper bound. avoid aliasing if sampling too coarse.
    nyquist_like = 0.5 / max(median_dt, 1e-8)
    max_frequency = min(1.0 / MIN_PERIOD, nyquist_like)

    if min_frequency >= max_frequency:
        # fall back to small frequency range around 1 / t_span
        min_frequency = 1.0 / t_span
        max_frequency = min(1.0 / MIN_PERIOD, min_frequency * 10.0)

    # adaptive samples_per_peak: longer baselines -> higher resolution at low freq
    # samples_per_peak controls frequency spacing: more for long baseline
    samples_per_peak = int(np.clip(5 + (t_span / 50.0), 5, 50))

    # run Lomb-Scargle
    try:
        ls = LombScargle(t, mag, err, center_data=True)
        frequency, power = ls.autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            samples_per_peak=samples_per_peak,
            method="fast"  # 'fast' if available
        )
    except Exception as e:
        print(f"LS failed for {star}: {e}")
        continue

    if not np.any(np.isfinite(power)):
        print(f"No power for {star}")
        continue

    # best peak in power
    i_best = np.nanargmax(power)
    best_freq = frequency[i_best]
    best_power = power[i_best]
    best_period = 1.0 / best_freq

    # false alarm probability for the peak
    try:
        fap = ls.false_alarm_probability(best_power, method="baluev")
    except Exception:
        # fallback to default method
        try:
            fap = ls.false_alarm_probability(best_power)
        except Exception:
            fap = np.nan

    # derived statistics
    std_mag = np.std(mag, ddof=1)
    amp_est = 0.5 * (np.percentile(mag, 95) - np.percentile(mag, 5))
    median_err = np.median(err)

    # Fit sinusoid at the candidate period and compute weighted R^2 and amplitude
    amp, offset, r2, model = fit_sinusoid(t, mag, err, best_period)

    # For plotting: construct phase (0..1) and duplicate to show two cycles
    phase = (t % best_period) / best_period
    # sort for nicer plotting
    order = np.argsort(phase)
    phase_sorted = phase[order]
    mag_sorted = mag[order]

    phase2 = np.concatenate([phase_sorted, phase_sorted + 1.0])
    mag2 = np.concatenate([mag_sorted, mag_sorted])

    # compute binned median curve for the folded light curve
    bin_edges = np.linspace(0.0, 1.0, PHASE_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    medians = np.full_like(bin_centers, np.nan, dtype=float)
    for i in range(len(bin_centers)):
        sel = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.any(sel):
            medians[i] = np.median(mag[sel])

    # Make plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9),
                                   gridspec_kw={"height_ratios": [1, 1.5]})
    # Power spectrum
    ax1.plot(frequency, power, "k-", lw=1)
    ax1.axvline(best_freq, color="r", ls="--", label=f"P={best_period:.3f} d")
    ax1.set_xlabel("Frequency (1/day)")
    ax1.set_ylabel("Power")
    ax1.set_title(f"{star} | P={best_period:.4f} d | power={best_power:.3f} | fap={fap:.3g}")

    # Phase-folded light curve
    ax2.scatter(phase2, mag2, s=8, color="black", alpha=0.25, rasterized=True)
    # overplot binned medians
    valid = np.isfinite(medians)

if np.any(valid):
    bc = bin_centers[valid]
    md = medians[valid]

    ax2.plot(
        np.concatenate([bc, bc + 1.0]),
        np.concatenate([md, md]),
        marker="o",
        linestyle="-",
        linewidth=1.5,
        color="tab:blue"
    )

    # overplot fitted sinusoid (folded)
    t_folded = np.linspace(0, 2.0 * best_period, 500)
    phi_folded = (t_folded % best_period) / best_period
    # reconstruct model on folded grid: use sinusoid parameters from fit
    if np.isfinite(amp):
        omega = 2.0 * np.pi / best_period
        # retrieve A,B from model fit indirectly: rebuild using fit_sinusoid would be better but we have model only for original t
        # approximate continuous sinusoid using ls.model with best frequency
        try:
            model_cont = ls.model(t_folded, 1.0 / best_period)
            # convert to magnitude-space model_cont is in same units as mag
            ax2.plot(phi_folded, model_cont, '-', lw=1.2)
        except Exception:
            pass

    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Differential Magnitude")
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{star}.png"), dpi=150)
    plt.close(fig)

    print(f"Saved {star} (P={best_period:.4f} d, power={best_power:.3f}, fap={fap:.3g}, amp={amp:.4f}, r2={r2:.3f})")

    # Summary lookup
    star_id = extract_star_id(star)
    x_mean = np.nan
    y_mean = np.nan
    if star_id is not None and not summary_list.empty:
        row = summary_list.loc[summary_list["Star"] == star_id]
        if not row.empty:
            x_mean = row["X_mean"].iloc[0]
            y_mean = row["Y_mean"].iloc[0]

    final_list.append({
        "STAR": star_id,
        "star_name": star,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "best_period": best_period,
        "best_power": best_power,
        "fap": fap,
        "amplitude_est": amp_est,
        "fitted_amplitude": amp,
        "r2": r2,
        "std_mag": std_mag,
        "median_err": median_err,
        "n_points": len(mag),
        "t_span_days": t_span
    })

    # Apply candidate selection: require significant power, sinusoidal fit quality, and amplitude above noise
    amp_ok = (np.isfinite(amp) and amp >= AMP_SNR * max(median_err, 1e-8))
    fap_ok = (np.isfinite(fap) and fap <= FAP_THRESHOLD)
    r2_ok = (np.isfinite(r2) and r2 >= R2_THRESHOLD)
    if amp_ok and fap_ok and r2_ok:
        candidates.append(final_list[-1])

# ----------------------------
# Save results
# ----------------------------
df_final = pd.DataFrame(final_list)
if not df_final.empty:
    df_final = df_final.sort_values("best_period", na_position="last")
    df_final.round(6).to_csv(ALL_TABLE, index=False)
    print(f"Saved all results to {ALL_TABLE}")

df_candidates = pd.DataFrame(candidates)
if not df_candidates.empty:
    df_candidates = df_candidates.sort_values("best_power", ascending=False)
    df_candidates.round(6).to_csv(CANDIDATES_TABLE, index=False)
    print(f"Saved {len(df_candidates)} candidate(s) to {CANDIDATES_TABLE}")
else:
    print("No candidates passed the selection thresholds.")

print("Finished.")
