"""
Cross-correlation acoustic leak localization simulation
CSC 416 Computer Systems Project - Stephen Kiarie Mbugua

Pipe of length L with sensor A at x = 0 and sensor B at x = L.
A leak at distance x from A sends noise to both sensors.
  arrival at A: x / v          arrival at B: (L - x) / v
  tau = t_A - t_B              =>  x = (L + v * tau) / 2

Two time-delay estimators are compared:
  1. standard cross-correlation
  2. GCC-PHAT (cross-spectrum whitened before correlating)

Run:  python leak_simulation.py
Outputs (in ./results): results_raw.csv, results_summary.csv, rmse_vs_snr.png,
                        success_vs_snr.png, example_correlation.png
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to files; works on any machine
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURATION - change these to explore different conditions
# ----------------------------------------------------------------------
SEED = 42                      # fixed seed => reproducible results
PIPE_LENGTH_M = 100.0          # L, distance between sensors (m)
WAVE_SPEED_MS = 1200.0         # v (m/s). Roughly: cast iron ~1200, PVC/plastic ~400-500
FS_HZ = 5000.0                 # sampling rate (Hz)
DURATION_S = 1.0               # length of each recording (s)
LEAK_BAND_HZ = (50.0, 1000.0)  # frequency band of the leak noise
LEAK_POSITIONS_M = [10, 25, 40, 60, 85]   # at least 5 leak positions
SNR_LEVELS_DB = [0, -10, -14, -17, -20]      # at least 5 noise levels
RUNS_PER_CONDITION = 50        # repetitions per (position, SNR)
SUCCESS_FRACTION = 0.05        # success = error within 5% of pipe length
PHAT_BAND_ONLY = True          # restrict PHAT whitening to the leak band
OUT_DIR = "results"


# ----------------------------------------------------------------------
# Module 1: signal generator
# ----------------------------------------------------------------------
def generate_leak_source(n_samples, fs, band, rng):
    """Band-limited white Gaussian noise, normalised to unit power."""
    white = rng.standard_normal(n_samples)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    spec[(freqs < band[0]) | (freqs > band[1])] = 0.0
    s = np.fft.irfft(spec, n=n_samples)
    return s / np.std(s)


def delay_signal(x, delay_s, fs):
    """Delay a signal by delay_s seconds (fractional) using a phase shift."""
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec *= np.exp(-2j * np.pi * freqs * delay_s)
    return np.fft.irfft(spec, n=n)


# ----------------------------------------------------------------------
# Module 2: noise and channel model
# ----------------------------------------------------------------------
def simulate_sensors(leak_pos, length, v, fs, duration, band, snr_db, rng):
    """Return the two sensor signals for a leak at leak_pos metres from A."""
    n = int(duration * fs)
    pad = int(np.ceil((length / v) * fs)) + 200   # room so delays do not wrap
    src = generate_leak_source(n + 2 * pad, fs, band, rng)

    t_a = leak_pos / v
    t_b = (length - leak_pos) / v
    a_full = delay_signal(src, t_a, fs)
    b_full = delay_signal(src, t_b, fs)
    a = a_full[pad:pad + n]
    b = b_full[pad:pad + n]

    # additive white noise at the requested SNR (signal power / noise power)
    sig_power = 0.5 * (np.var(a) + np.var(b))
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    a = a + rng.normal(0.0, np.sqrt(noise_power), n)
    b = b + rng.normal(0.0, np.sqrt(noise_power), n)
    return a, b


# ----------------------------------------------------------------------
# Module 3: time-delay estimators
# ----------------------------------------------------------------------
def _correlation(a, b, fs, method, band):
    """Return (lags_in_seconds, correlation) for 'cc' or 'phat'."""
    n = len(a)
    nfft = 2 * n  # zero pad to avoid circular wrap
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    cross = A * np.conj(B)

    if method == "phat":
        mag = np.abs(cross)
        mag[mag < 1e-12] = 1e-12
        cross = cross / mag
        if PHAT_BAND_ONLY:
            freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
            cross[(freqs < band[0]) | (freqs > band[1])] = 0.0

    r = np.fft.irfft(cross, nfft)
    r = np.fft.fftshift(r)
    lags = (np.arange(nfft) - nfft // 2) / fs
    return lags, r


def estimate_delay(a, b, fs, max_delay_s, method, band):
    """Estimate tau = t_A - t_B as the correlation peak (sub-sample refined)."""
    lags, r = _correlation(a, b, fs, method, band)
    mask = np.abs(lags) <= max_delay_s
    idx = np.where(mask)[0]
    k = idx[np.argmax(r[idx])]

    # parabolic interpolation around the peak for sub-sample precision
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    return lags[k] + shift / fs


# ----------------------------------------------------------------------
# Module 4: position calculator
# ----------------------------------------------------------------------
def position_from_delay(tau, length, v):
    """x = (L + v * tau) / 2, clipped to the physical pipe length."""
    x = (length + v * tau) / 2.0
    return float(np.clip(x, 0.0, length))


# ----------------------------------------------------------------------
# Module 5: evaluation and plotting
# ----------------------------------------------------------------------
def run_experiments():
    rng = np.random.default_rng(SEED)
    max_delay = PIPE_LENGTH_M / WAVE_SPEED_MS
    tolerance = SUCCESS_FRACTION * PIPE_LENGTH_M
    rows = []

    for pos in LEAK_POSITIONS_M:
        for snr in SNR_LEVELS_DB:
            for run in range(RUNS_PER_CONDITION):
                a, b = simulate_sensors(pos, PIPE_LENGTH_M, WAVE_SPEED_MS, FS_HZ,
                                        DURATION_S, LEAK_BAND_HZ, snr, rng)
                for method in ("cc", "phat"):
                    tau = estimate_delay(a, b, FS_HZ, max_delay, method, LEAK_BAND_HZ)
                    est = position_from_delay(tau, PIPE_LENGTH_M, WAVE_SPEED_MS)
                    err = est - pos
                    rows.append({
                        "method": method, "leak_pos_m": pos, "snr_db": snr,
                        "run": run, "estimated_m": round(est, 4),
                        "error_m": round(err, 4),
                        "success": int(abs(err) <= tolerance),
                    })
    return rows


def summarise(rows):
    """MAE, RMSE (m) and % success for each method and SNR (all positions)."""
    out = []
    for method in ("cc", "phat"):
        for snr in SNR_LEVELS_DB:
            errs = np.array([r["error_m"] for r in rows
                             if r["method"] == method and r["snr_db"] == snr])
            succ = np.array([r["success"] for r in rows
                             if r["method"] == method and r["snr_db"] == snr])
            out.append({
                "method": method, "snr_db": snr, "n": len(errs),
                "mae_m": float(np.mean(np.abs(errs))),
                "rmse_m": float(np.sqrt(np.mean(errs ** 2))),
                "success_pct": float(100.0 * np.mean(succ)),
            })
    return out


def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_summary(summary):
    labels = {"cc": "Cross-correlation", "phat": "GCC-PHAT"}
    for key, ylabel, fname in (("rmse_m", "RMSE (m)", "rmse_vs_snr.png"),
                               ("success_pct",
                                f"Within {int(SUCCESS_FRACTION*100)}% of pipe length (%)",
                                "success_vs_snr.png")):
        plt.figure(figsize=(6, 4))
        for method in ("cc", "phat"):
            pts = [s for s in summary if s["method"] == method]
            plt.plot([p["snr_db"] for p in pts], [p[key] for p in pts],
                     marker="o", label=labels[method])
        plt.xlabel("Signal-to-noise ratio (dB)")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close()


def plot_example():
    """One example: correlation curves for both methods at 0 dB."""
    rng = np.random.default_rng(SEED)
    pos, snr = 40, 0
    a, b = simulate_sensors(pos, PIPE_LENGTH_M, WAVE_SPEED_MS, FS_HZ,
                            DURATION_S, LEAK_BAND_HZ, snr, rng)
    true_tau = (2 * pos - PIPE_LENGTH_M) / WAVE_SPEED_MS
    max_delay = PIPE_LENGTH_M / WAVE_SPEED_MS
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax, method, title in zip(axes, ("cc", "phat"),
                                 ("Cross-correlation", "GCC-PHAT")):
        lags, r = _correlation(a, b, FS_HZ, method, LEAK_BAND_HZ)
        m = np.abs(lags) <= max_delay
        ax.plot(lags[m] * 1000, r[m] / np.max(np.abs(r[m])))
        ax.axvline(true_tau * 1000, color="red", linestyle="--", label="true delay")
        ax.set_title(f"{title} (leak at {pos} m, SNR {snr} dB)")
        ax.set_ylabel("Normalised")
        ax.legend()
    axes[-1].set_xlabel("Lag (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "example_correlation.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Pipe {PIPE_LENGTH_M} m, v = {WAVE_SPEED_MS} m/s, fs = {FS_HZ} Hz")
    print(f"{len(LEAK_POSITIONS_M)} positions x {len(SNR_LEVELS_DB)} SNR levels x "
          f"{RUNS_PER_CONDITION} runs x 2 methods\n")

    rows = run_experiments()
    summary = summarise(rows)
    save_csv(os.path.join(OUT_DIR, "results_raw.csv"), rows)
    save_csv(os.path.join(OUT_DIR, "results_summary.csv"), summary)
    plot_summary(summary)
    plot_example()

    names = {"cc": "Cross-corr", "phat": "GCC-PHAT"}
    print(f"{'Method':<11}{'SNR(dB)':>8}{'MAE(m)':>9}{'RMSE(m)':>9}{'Success%':>10}")
    for s in summary:
        print(f"{names[s['method']]:<11}{s['snr_db']:>8}{s['mae_m']:>9.3f}"
              f"{s['rmse_m']:>9.3f}{s['success_pct']:>10.1f}")
    print(f"\nFiles saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
