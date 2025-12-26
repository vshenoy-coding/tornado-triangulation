# =========================================
# CELL 1 — Setup, Imports, Constants
# High‑Resolution Monte‑Carlo (HR‑1)
# =========================================

!pip install -q numpy scipy matplotlib tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

# -----------------------------------------
# Global constants
# -----------------------------------------

AREA_SIZE = 2000.0          # meters (2 km × 2 km)
NUM_SENSORS = 60            # number of devices
BASE_SPEED_OF_SOUND = 343.0 # m/s
SPEED_OF_SOUND_STD = 3.0    # m/s event-to-event variation

# Noise levels
CAL_NOISE_STD = 0.01        # s, calibration noise
TORNADO_NOISE_STD = 0.03    # s, tornado noise

# Outlier behavior
OUTLIER_CAL_BIAS = 0.20     # s
OUTLIER_TORNADO_BIAS = 0.20 # s
OUTLIER_EXTRA_NOISE = 0.05  # s

# Calibration outlier detection
CAL_RMS_MULT = 2.5

# Tornado failure threshold
TORNADO_FAIL_THRESHOLD_M = 200.0

# Sentinel error for failed runs
FAIL_ERR = 9999.0

# -----------------------------------------
# High‑Resolution Monte‑Carlo Grid (HR‑1)
# -----------------------------------------

NUM_CAL_EVENTS_GRID = np.array([2, 4, 6, 8, 10, 12])
CLOCK_OFFSET_STD_GRID = np.array([0.01, 0.03, 0.05, 0.08, 0.12, 0.18])  # seconds
OUTLIER_FRACTION_GRID = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30])

NUM_MC_SAMPLES = 6  # per grid point

print("HR‑1 Monte‑Carlo grid:")
print("  NUM_CAL_EVENTS:", NUM_CAL_EVENTS_GRID)
print("  CLOCK_OFFSET_STD:", CLOCK_OFFSET_STD_GRID)
print("  OUTLIER_FRACTION:", OUTLIER_FRACTION_GRID)
print("  MC samples per point:", NUM_MC_SAMPLES)
print("  Total runs:", len(NUM_CAL_EVENTS_GRID) *
                     len(CLOCK_OFFSET_STD_GRID) *
                     len(OUTLIER_FRACTION_GRID) *
                     NUM_MC_SAMPLES)

# =========================================
# CELL 2 — Core Simulation Functions
# =========================================

# -----------------------------------------
# Sensor placement
# -----------------------------------------

def generate_sensor_positions(num, area_size, rng):
    """
    Realistic-ish sensor distribution:
    - Half in a central cluster
    - Half scattered uniformly
    """
    n_cluster = num // 2
    n_scatter = num - n_cluster

    cluster_center = np.array([0.5 * area_size, 0.5 * area_size])
    cluster_std = 0.15 * area_size

    # Use rng.normal (Generator API)
    cluster = cluster_center + rng.normal(0.0, 1.0, size=(n_cluster, 2)) * cluster_std
    cluster = np.clip(cluster, 0, area_size)

    scatter = rng.uniform(0, area_size, size=(n_scatter, 2))

    sensors = np.vstack([cluster, scatter])
    return sensors[:, 0], sensors[:, 1]


# -----------------------------------------
# Parameter unpacking for calibration LS
# -----------------------------------------

def unpack_params(params, num_sensors, num_events):
    """
    Unpack flat parameter vector into:
    - b_i sensor offsets (sensor 0 fixed = 0)
    - event parameters (x_k, y_k, t0_k)
    """
    num_offsets = num_sensors - 1
    num_event_params = num_events * 3

    b_vec = np.zeros(num_sensors)
    b_vec[1:] = params[:num_offsets]

    event_params = params[num_offsets:num_offsets + num_event_params]

    cal_x = np.zeros(num_events)
    cal_y = np.zeros(num_events)
    cal_t0 = np.zeros(num_events)

    for k in range(num_events):
        base = 3 * k
        cal_x[k] = event_params[base + 0]
        cal_y[k] = event_params[base + 1]
        cal_t0[k] = event_params[base + 2]

    return b_vec, cal_x, cal_y, cal_t0


# -----------------------------------------
# Calibration residuals
# -----------------------------------------

def calibration_residuals(params, sx, sy, cal_obs, c_base):
    """
    Residuals for all calibration events and sensors.
    Uses fixed speed c_base for simplicity.
    """
    num_events, num_sensors = cal_obs.shape
    b_vec, cal_x, cal_y, cal_t0 = unpack_params(params, num_sensors, num_events)

    residuals = []
    for k in range(num_events):
        dx = sx - cal_x[k]
        dy = sy - cal_y[k]
        dist = np.sqrt(dx**2 + dy**2)
        pred = cal_t0[k] + dist / c_base + b_vec
        residuals.append(cal_obs[k] - pred)

    return np.concatenate(residuals)


def calibration_residuals_subset(params, sx, sy, cal_obs, c_base):
    """
    Same as above, but for a subset of sensors.
    """
    num_events, num_sensors = cal_obs.shape
    b_vec, cal_x, cal_y, cal_t0 = unpack_params(params, num_sensors, num_events)

    residuals = []
    for k in range(num_events):
        dx = sx - cal_x[k]
        dy = sy - cal_y[k]
        dist = np.sqrt(dx**2 + dy**2)
        pred = cal_t0[k] + dist / c_base + b_vec
        residuals.append(cal_obs[k] - pred)

    return np.concatenate(residuals)


# -----------------------------------------
# Tornado residuals + solver
# -----------------------------------------

def tornado_toa_residuals(params, sx, sy, obs_times, c):
    x, y, t0 = params
    d = np.sqrt((sx - x)**2 + (sy - y)**2)
    pred = t0 + d / c
    return obs_times - pred


def solve_tornado(obs_times, sx, sy, c, robust_f_scale, mask=None):
    """
    Robust tornado TOA solver.
    Returns (result, sx_used, sy_used) or (None, None, None) on failure.
    """
    if mask is not None:
        idx = np.where(mask)[0]
        sx = sx[idx]
        sy = sy[idx]
        obs_times = obs_times[idx]

    if len(sx) < 4:
        return None, None, None

    # Initial guess: centroid of earliest arrivals
    k = max(4, int(len(sx) * 0.2))
    earliest = np.argsort(obs_times)[:k]
    init_x = np.mean(sx[earliest])
    init_y = np.mean(sy[earliest])
    d_guess = np.sqrt((sx - init_x)**2 + (sy - init_y)**2)
    t0_guess = np.min(obs_times) - np.min(d_guess) / c

    init_params = np.array([init_x, init_y, t0_guess])

    try:
        res = least_squares(
            tornado_toa_residuals,
            init_params,
            args=(sx, sy, obs_times, c),
            loss='huber',
            f_scale=robust_f_scale,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=10000
        )
    except Exception:
        return None, None, None

    return res, sx, sy


# -----------------------------------------
# Full simulation: calibration + tornado
# -----------------------------------------

def run_single_simulation(
    num_cal_events,
    clock_offset_std,
    outlier_fraction,
    rng_seed=None,
    verbose=False
):
    """
    Runs one full simulation:
    - Sensor placement
    - True offsets
    - Calibration events
    - Two-pass calibration
    - Tornado event
    - Tornado localization (before/after)
    Returns:
      (err_before, err_after_all, err_after_inliers)
    Or FAIL_ERR triplet on failure.
    """
    rng = np.random.default_rng(rng_seed)

    # -------------------------
    # Sensor placement
    # -------------------------
    sensor_x, sensor_y = generate_sensor_positions(NUM_SENSORS, AREA_SIZE, rng)

    # -------------------------
    # True clock offsets
    # -------------------------
    true_offsets = rng.normal(0.0, clock_offset_std, NUM_SENSORS)
    true_offsets -= true_offsets[0]  # reference sensor

    # -------------------------
    # Outlier sensors
    # -------------------------
    num_outliers = int(np.round(NUM_SENSORS * outlier_fraction))
    idx = np.arange(NUM_SENSORS)
    rng.shuffle(idx)
    outlier_idx = idx[:num_outliers]
    inlier_idx = idx[num_outliers:]

    # -------------------------
    # Calibration events
    # -------------------------
    cal_true_x = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_y = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_t0 = rng.uniform(0.0, 2.0, num_cal_events)
    cal_speeds = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD, num_cal_events)

    cal_obs = np.zeros((num_cal_events, NUM_SENSORS))

    for k in range(num_cal_events):
        c_k = cal_speeds[k]
        dx = sensor_x - cal_true_x[k]
        dy = sensor_y - cal_true_y[k]
        dist = np.sqrt(dx**2 + dy**2)
        base_arrival = cal_true_t0[k] + dist / c_k

        # Inliers
        cal_obs[k, inlier_idx] = (
            base_arrival[inlier_idx]
            + true_offsets[inlier_idx]
            + rng.normal(0.0, CAL_NOISE_STD, len(inlier_idx))
        )

        # Outliers
        cal_obs[k, outlier_idx] = (
            base_arrival[outlier_idx]
            + true_offsets[outlier_idx]
            + OUTLIER_CAL_BIAS
            + rng.normal(0.0, CAL_NOISE_STD + OUTLIER_EXTRA_NOISE, len(outlier_idx))
        )

    # -------------------------
    # First-pass calibration (all sensors)
    # -------------------------
    num_offsets_full = NUM_SENSORS - 1
    init_offsets = np.zeros(num_offsets_full)
    init_x = np.full(num_cal_events, np.mean(sensor_x))
    init_y = np.full(num_cal_events, np.mean(sensor_y))
    init_t0 = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sensor_x - init_x[k]
        dy = sensor_y - init_y[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        init_t0[k] = np.min(cal_obs[k]) - np.median(dist_guess) / BASE_SPEED_OF_SOUND

    init_params = np.concatenate([
        init_offsets,
        np.column_stack([init_x, init_y, init_t0]).ravel()
    ])

    try:
        res_full = least_squares(
            calibration_residuals,
            init_params,
            args=(sensor_x, sensor_y, cal_obs, BASE_SPEED_OF_SOUND),
            loss='huber',
            f_scale=0.03,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=20000
        )
    except Exception:
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    est_offsets_full, _, _, _ = unpack_params(res_full.x, NUM_SENSORS, num_cal_events)

    # -------------------------
    # Outlier detection via RMS
    # -------------------------
    residuals_flat = calibration_residuals(
        res_full.x, sensor_x, sensor_y, cal_obs, BASE_SPEED_OF_SOUND
    )
    residuals = residuals_flat.reshape(num_cal_events, NUM_SENSORS)
    sensor_rms = np.sqrt(np.mean(residuals**2, axis=0))

    med = np.median(sensor_rms)
    mad = np.median(np.abs(sensor_rms - med)) + 1e-9
    thresh = med + CAL_RMS_MULT * mad

    cal_inliers = np.where(sensor_rms <= thresh)[0]
    cal_outliers = np.where(sensor_rms > thresh)[0]

    if len(cal_inliers) < 4:
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    # -------------------------
    # Second-pass calibration (inliers only)
    # -------------------------
    sx_in = sensor_x[cal_inliers]
    sy_in = sensor_y[cal_inliers]
    obs_in = cal_obs[:, cal_inliers]

    num_in = len(cal_inliers)
    num_offsets_in = num_in - 1

    init_offsets2 = np.zeros(num_offsets_in)
    init_x2 = np.full(num_cal_events, np.mean(sx_in))
    init_y2 = np.full(num_cal_events, np.mean(sy_in))
    init_t02 = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sx_in - init_x2[k]
        dy = sy_in - init_y2[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        init_t02[k] = np.min(obs_in[k]) - np.median(dist_guess) / BASE_SPEED_OF_SOUND

    init_params2 = np.concatenate([
        init_offsets2,
        np.column_stack([init_x2, init_y2, init_t02]).ravel()
    ])

    try:
        res_in = least_squares(
            calibration_residuals_subset,
            init_params2,
            args=(sx_in, sy_in, obs_in, BASE_SPEED_OF_SOUND),
            loss='huber',
            f_scale=0.03,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=20000
        )
    except Exception:
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    est_offsets_sub, _, _, _ = unpack_params(res_in.x, num_in, num_cal_events)

    # Map back to full set
    est_offsets_final = np.zeros(NUM_SENSORS)
    est_offsets_final[:] = np.nan

    ref = cal_inliers[0]
    est_offsets_final[ref] = 0.0

    for local_i, global_i in enumerate(cal_inliers[1:], start=1):
        est_offsets_final[global_i] = est_offsets_sub[local_i]

    # Outliers get zero offset (robust LS will downweight)
    for g in cal_outliers:
        if np.isnan(est_offsets_final[g]):
            est_offsets_final[g] = 0.0

    # -------------------------
    # Tornado event
    # -------------------------
    true_tx = 1500.0
    true_ty = 1200.0
    true_t0 = 10.0

    c_t = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD)

    dx = sensor_x - true_tx
    dy = sensor_y - true_ty
    dist = np.sqrt(dx**2 + dy**2)
    true_toa = true_t0 + dist / c_t

    obs_raw = np.zeros(NUM_SENSORS)

    # Inliers
    obs_raw[inlier_idx] = (
        true_toa[inlier_idx]
        + true_offsets[inlier_idx]
        + rng.normal(0.0, TORNADO_NOISE_STD, len(inlier_idx))
    )

    # Outliers
    obs_raw[outlier_idx] = (
        true_toa[outlier_idx]
        + true_offsets[outlier_idx]
        + OUTLIER_TORNADO_BIAS
        + rng.normal(0.0, TORNADO_NOISE_STD + OUTLIER_EXTRA_NOISE, len(outlier_idx))
    )

    obs_corr = obs_raw - est_offsets_final

    # -------------------------
    # Tornado localization
    # -------------------------

    # Before calibration
    res_b, _, _ = solve_tornado(
        obs_raw, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    if res_b is None:
        err_before = FAIL_ERR
    else:
        xb, yb, _ = res_b.x
        err_before = np.sqrt((xb - true_tx)**2 + (yb - true_ty)**2)

    # After calibration (all sensors)
    res_a_all, _, _ = solve_tornado(
        obs_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    if res_a_all is None:
        err_after_all = FAIL_ERR
    else:
        xa, ya, _ = res_a_all.x
        err_after_all = np.sqrt((xa - true_tx)**2 + (ya - true_ty)**2)

    # After calibration (inlier sensors only)
    mask = np.zeros(NUM_SENSORS, dtype=bool)
    mask[cal_inliers] = True

    res_a_in, _, _ = solve_tornado(
        obs_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2,
        mask=mask
    )
    if res_a_in is None:
        err_after_in = FAIL_ERR
    else:
        xi, yi, _ = res_a_in.x
        err_after_in = np.sqrt((xi - true_tx)**2 + (yi - true_ty)**2)

    return err_before, err_after_all, err_after_in

# =========================================
# CELL 3 — High‑Resolution Monte‑Carlo Sweep (HR‑1)
# =========================================

Ne = len(NUM_CAL_EVENTS_GRID)
Nc = len(CLOCK_OFFSET_STD_GRID)
No = len(OUTLIER_FRACTION_GRID)
Nm = NUM_MC_SAMPLES

# Allocate result tensors
errors_before = np.zeros((Ne, Nc, No, Nm))
errors_after_all = np.zeros((Ne, Nc, No, Nm))
errors_after_in = np.zeros((Ne, Nc, No, Nm))

total_runs = Ne * Nc * No * Nm

print("Starting HR‑1 Monte‑Carlo sweep…")
print(f"Grid size: {Ne} × {Nc} × {No}, MC samples: {Nm}")
print(f"Total simulations: {total_runs}\n")

run_counter = 0

with tqdm(total=total_runs, desc="HR‑1 Monte‑Carlo") as pbar:
    for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
        for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
            for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):

                for im in range(NUM_MC_SAMPLES):
                    seed = (
                        100000
                        + 1000 * ie
                        + 100 * ic
                        + 10 * io
                        + im
                    )

                    err_b, err_a_all, err_a_in = run_single_simulation(
                        num_cal_events=num_cal_events,
                        clock_offset_std=clock_std,
                        outlier_fraction=out_frac,
                        rng_seed=seed,
                        verbose=False
                    )

                    errors_before[ie, ic, io, im] = err_b
                    errors_after_all[ie, ic, io, im] = err_a_all
                    errors_after_in[ie, ic, io, im] = err_a_in

                    run_counter += 1
                    pbar.update(1)

print("\nMonte‑Carlo sweep complete.\n")

# =========================================
# Aggregate statistics
# =========================================

median_before = np.median(errors_before, axis=3)
median_after_all = np.median(errors_after_all, axis=3)
median_after_in = np.median(errors_after_in, axis=3)

# Failure = error > threshold OR FAIL_ERR
fail_before = np.mean(
    (errors_before > TORNADO_FAIL_THRESHOLD_M) |
    (errors_before >= FAIL_ERR),
    axis=3
)
fail_after_all = np.mean(
    (errors_after_all > TORNADO_FAIL_THRESHOLD_M) |
    (errors_after_all >= FAIL_ERR),
    axis=3
)
fail_after_in = np.mean(
    (errors_after_in > TORNADO_FAIL_THRESHOLD_M) |
    (errors_after_in >= FAIL_ERR),
    axis=3
)

print("Median tornado localization error (m) AFTER calibration (inliers only):")
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    print(f"\nNUM_CAL_EVENTS = {num_cal_events}:")
    for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
        vals = median_after_in[ie, ic, :]
        line = f"  clock_std={clock_std*1000:.0f} ms: "
        line += " | ".join(
            f"outlier_frac={OUTLIER_FRACTION_GRID[io]:.2f}: {vals[io]:.1f}"
            for io in range(No)
        )
        print(line)

print("\nFailure rate AFTER calibration (inliers only):")
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    print(f"\nNUM_CAL_EVENTS = {num_cal_events}:")
    for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
        vals = fail_after_in[ie, ic, :]
        line = f"  clock_std={clock_std*1000:.0f} ms: "
        line += " | ".join(
            f"outlier_frac={OUTLIER_FRACTION_GRID[io]:.2f}: {vals[io]*100:.0f}%"
            for io in range(No)
        )
        print(line)

# =========================================
# Heatmap plotting helper
# =========================================

def plot_heatmap_slice(
    data, x_vals, y_vals,
    x_label, y_label, title,
    vmin=None, vmax=None, cmap='viridis'
):
    plt.figure(figsize=(6,5))
    im = plt.imshow(
        data,
        origin='lower',
        aspect='auto',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    plt.colorbar(im, label='Median error (m)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================================
# Heatmaps
# =========================================

# 1. Fix OUTLIER_FRACTION, vary events × clock_std
for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):
    data = median_after_in[:, :, io]
    plot_heatmap_slice(
        data,
        x_vals=CLOCK_OFFSET_STD_GRID * 1000,
        y_vals=NUM_CAL_EVENTS_GRID,
        x_label='Clock offset std (ms)',
        y_label='Num calibration events',
        title=f'Median error after calib (inliers)\noutlier_frac={out_frac:.2f}'
    )

# 2. Fix CLOCK_OFFSET_STD, vary events × outlier_fraction
for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
    data = median_after_in[:, ic, :]
    plot_heatmap_slice(
        data,
        x_vals=OUTLIER_FRACTION_GRID,
        y_vals=NUM_CAL_EVENTS_GRID,
        x_label='Outlier fraction',
        y_label='Num calibration events',
        title=f'Median error after calib (inliers)\nclock_std={clock_std*1000:.0f} ms'
    )

# 3. Fix NUM_CAL_EVENTS, vary clock_std × outlier_fraction
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    data = median_after_in[ie, :, :]
    plot_heatmap_slice(
        data,
        x_vals=OUTLIER_FRACTION_GRID,
        y_vals=CLOCK_OFFSET_STD_GRID * 1000,
        x_label='Outlier fraction',
        y_label='Clock offset std (ms)',
        title=f'Median error after calib (inliers)\nnum_events={num_cal_events}'
    )

# =========================================
# Summary curves
# =========================================

mean_vs_events = np.mean(median_after_in, axis=(1,2))
mean_vs_clock = np.mean(median_after_in, axis=(0,2))
mean_vs_outlier = np.mean(median_after_in, axis=(0,1))

plt.figure(figsize=(5,4))
plt.plot(NUM_CAL_EVENTS_GRID, mean_vs_events, marker='o')
plt.xlabel('Num calibration events')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Num calibration events')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(CLOCK_OFFSET_STD_GRID * 1000, mean_vs_clock, marker='o')
plt.xlabel('Clock offset std (ms)')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Clock offset std')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(OUTLIER_FRACTION_GRID, mean_vs_outlier, marker='o')
plt.xlabel('Outlier fraction')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Outlier fraction')
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

print("\nHR‑1 Monte‑Carlo sweep finished.")

# =========================================
# CELL 4 — Median‑Scenario Visualization
# =========================================

print("Running median‑scenario visualization…")

# Median scenario parameters (realistic)
num_cal_events = 6
clock_offset_std = 0.08
outlier_fraction = 0.15

rng = np.random.default_rng(12345)

# -----------------------------------------
# Run a single simulation but with extra outputs
# -----------------------------------------

def run_single_simulation_with_outputs(
    num_cal_events,
    clock_offset_std,
    outlier_fraction,
    rng_seed=None
):
    """
    Same as run_single_simulation, but returns:
    - sensor positions
    - calibration inliers/outliers
    - true tornado location
    - estimated tornado locations (before, after-all, after-inliers)
    """
    rng = np.random.default_rng(rng_seed)

    # Sensor placement
    sensor_x, sensor_y = generate_sensor_positions(NUM_SENSORS, AREA_SIZE, rng)

    # True offsets
    true_offsets = rng.normal(0.0, clock_offset_std, NUM_SENSORS)
    true_offsets -= true_offsets[0]

    # Outlier sensors
    num_outliers = int(np.round(NUM_SENSORS * outlier_fraction))
    idx = np.arange(NUM_SENSORS)
    rng.shuffle(idx)
    outlier_idx = idx[:num_outliers]
    inlier_idx = idx[num_outliers:]

    # Calibration events
    cal_true_x = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_y = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_t0 = rng.uniform(0.0, 2.0, num_cal_events)
    cal_speeds = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD, num_cal_events)

    cal_obs = np.zeros((num_cal_events, NUM_SENSORS))

    for k in range(num_cal_events):
        c_k = cal_speeds[k]
        dx = sensor_x - cal_true_x[k]
        dy = sensor_y - cal_true_y[k]
        dist = np.sqrt(dx**2 + dy**2)
        base_arrival = cal_true_t0[k] + dist / c_k

        cal_obs[k, inlier_idx] = (
            base_arrival[inlier_idx]
            + true_offsets[inlier_idx]
            + rng.normal(0.0, CAL_NOISE_STD, len(inlier_idx))
        )

        cal_obs[k, outlier_idx] = (
            base_arrival[outlier_idx]
            + true_offsets[outlier_idx]
            + OUTLIER_CAL_BIAS
            + rng.normal(0.0, CAL_NOISE_STD + OUTLIER_EXTRA_NOISE, len(outlier_idx))
        )

    # First-pass calibration
    num_offsets_full = NUM_SENSORS - 1
    init_offsets = np.zeros(num_offsets_full)
    init_x = np.full(num_cal_events, np.mean(sensor_x))
    init_y = np.full(num_cal_events, np.mean(sensor_y))
    init_t0 = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sensor_x - init_x[k]
        dy = sensor_y - init_y[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        init_t0[k] = np.min(cal_obs[k]) - np.median(dist_guess) / BASE_SPEED_OF_SOUND

    init_params = np.concatenate([
        init_offsets,
        np.column_stack([init_x, init_y, init_t0]).ravel()
    ])

    res_full = least_squares(
        calibration_residuals,
        init_params,
        args=(sensor_x, sensor_y, cal_obs, BASE_SPEED_OF_SOUND),
        loss='huber',
        f_scale=0.03,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=20000
    )

    est_offsets_full, _, _, _ = unpack_params(res_full.x, NUM_SENSORS, num_cal_events)

    # Outlier detection
    residuals_flat = calibration_residuals(
        res_full.x, sensor_x, sensor_y, cal_obs, BASE_SPEED_OF_SOUND
    )
    residuals = residuals_flat.reshape(num_cal_events, NUM_SENSORS)
    sensor_rms = np.sqrt(np.mean(residuals**2, axis=0))

    med = np.median(sensor_rms)
    mad = np.median(np.abs(sensor_rms - med)) + 1e-9
    thresh = med + CAL_RMS_MULT * mad

    cal_inliers = np.where(sensor_rms <= thresh)[0]
    cal_outliers = np.where(sensor_rms > thresh)[0]

    # Second-pass calibration
    sx_in = sensor_x[cal_inliers]
    sy_in = sensor_y[cal_inliers]
    obs_in = cal_obs[:, cal_inliers]

    num_in = len(cal_inliers)
    num_offsets_in = num_in - 1

    init_offsets2 = np.zeros(num_offsets_in)
    init_x2 = np.full(num_cal_events, np.mean(sx_in))
    init_y2 = np.full(num_cal_events, np.mean(sy_in))
    init_t02 = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sx_in - init_x2[k]
        dy = sy_in - init_y2[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        init_t02[k] = np.min(obs_in[k]) - np.median(dist_guess) / BASE_SPEED_OF_SOUND

    init_params2 = np.concatenate([
        init_offsets2,
        np.column_stack([init_x2, init_y2, init_t02]).ravel()
    ])

    res_in = least_squares(
        calibration_residuals_subset,
        init_params2,
        args=(sx_in, sy_in, obs_in, BASE_SPEED_OF_SOUND),
        loss='huber',
        f_scale=0.03,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=20000
    )

    est_offsets_sub, _, _, _ = unpack_params(res_in.x, num_in, num_cal_events)

    # Map back to full set
    est_offsets_final = np.zeros(NUM_SENSORS)
    est_offsets_final[:] = np.nan

    ref = cal_inliers[0]
    est_offsets_final[ref] = 0.0

    for local_i, global_i in enumerate(cal_inliers[1:], start=1):
        est_offsets_final[global_i] = est_offsets_sub[local_i]

    for g in cal_outliers:
        if np.isnan(est_offsets_final[g]):
            est_offsets_final[g] = 0.0

    # Tornado event
    true_tx = 1500.0
    true_ty = 1200.0
    true_t0 = 10.0

    c_t = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD)

    dx = sensor_x - true_tx
    dy = sensor_y - true_ty
    dist = np.sqrt(dx**2 + dy**2)
    true_toa = true_t0 + dist / c_t

    obs_raw = np.zeros(NUM_SENSORS)

    obs_raw[inlier_idx] = (
        true_toa[inlier_idx]
        + true_offsets[inlier_idx]
        + rng.normal(0.0, TORNADO_NOISE_STD, len(inlier_idx))
    )

    obs_raw[outlier_idx] = (
        true_toa[outlier_idx]
        + true_offsets[outlier_idx]
        + OUTLIER_TORNADO_BIAS
        + rng.normal(0.0, TORNADO_NOISE_STD + OUTLIER_EXTRA_NOISE, len(outlier_idx))
    )

    obs_corr = obs_raw - est_offsets_final

    # Tornado localization
    res_b, _, _ = solve_tornado(
        obs_raw, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    xb, yb, _ = res_b.x

    res_a_all, _, _ = solve_tornado(
        obs_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    xa, ya, _ = res_a_all.x

    mask = np.zeros(NUM_SENSORS, dtype=bool)
    mask[cal_inliers] = True

    res_a_in, _, _ = solve_tornado(
        obs_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2,
        mask=mask
    )
    xi, yi, _ = res_a_in.x

    return (
        sensor_x, sensor_y,
        cal_inliers, cal_outliers,
        (true_tx, true_ty),
        (xb, yb),
        (xa, ya),
        (xi, yi)
    )


# Run the simulation
(
    sensor_x, sensor_y,
    cal_inliers, cal_outliers,
    (true_tx, true_ty),
    (xb, yb),
    (xa, ya),
    (xi, yi)
) = run_single_simulation_with_outputs(
    num_cal_events,
    clock_offset_std,
    outlier_fraction,
    rng_seed=999
)

# -----------------------------------------
# Plotting
# -----------------------------------------

plt.figure(figsize=(8,8))
plt.scatter(sensor_x[cal_inliers], sensor_y[cal_inliers],
            c='green', label='Calibration inliers', s=60)
plt.scatter(sensor_x[cal_outliers], sensor_y[cal_outliers],
            c='red', label='Calibration outliers', s=60)

plt.scatter(true_tx, true_ty, c='black', marker='*', s=300, label='True tornado')

plt.scatter(xb, yb, c='blue', marker='x', s=200, label='Before calibration')
plt.scatter(xa, ya, c='purple', marker='X', s=200, label='After calibration (all)')
plt.scatter(xi, yi, c='orange', marker='D', s=200, label='After calibration (inliers)')

plt.legend()
plt.title("Median‑Scenario Tornado Localization\nBefore vs After Calibration")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True, linestyle=':')
plt.axis('equal')
plt.show()

print("Visualization complete.")

# =========================================
# CELL 5 — Additional Visualizations
# 3D Surface Plot, Violin Plot, Histograms
# =========================================

from mpl_toolkits.mplot3d import Axes3D

print("Generating additional visualizations…")

# ---------------------------------------------------------
# 1. 3D Surface Plot of Error Landscape
# ---------------------------------------------------------

# Choose a fixed OUTLIER_FRACTION index (middle = realistic)
io_mid = len(OUTLIER_FRACTION_GRID) // 2

X, Y = np.meshgrid(
    CLOCK_OFFSET_STD_GRID * 1000,   # ms
    NUM_CAL_EVENTS_GRID             # events
)

Z = median_after_in[:, :, io_mid]

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title(f"3D Error Landscape (Inliers Only)\nOutlier Fraction = {OUTLIER_FRACTION_GRID[io_mid]:.2f}")
ax.set_xlabel("Clock Offset Std (ms)")
ax.set_ylabel("Num Calibration Events")
ax.set_zlabel("Median Error (m)")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 2. Violin Plot of Error Distributions
# ---------------------------------------------------------

# Flatten all MC samples
flat_before = errors_before.flatten()
flat_after_all = errors_after_all.flatten()
flat_after_in = errors_after_in.flatten()

# Remove FAIL_ERR values
flat_before = flat_before[flat_before < FAIL_ERR]
flat_after_all = flat_after_all[flat_after_all < FAIL_ERR]
flat_after_in = flat_after_in[flat_after_in < FAIL_ERR]

plt.figure(figsize=(10,6))
plt.violinplot(
    [flat_before, flat_after_all, flat_after_in],
    showmeans=True,
    showextrema=True
)
plt.xticks([1,2,3], ["Before", "After (All)", "After (Inliers)"])
plt.ylabel("Localization Error (m)")
plt.title("Error Distribution Comparison (Violin Plot)")
plt.grid(True, linestyle=':')
plt.show()


# ---------------------------------------------------------
# 3. Before vs After Histograms
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
plt.hist(flat_before, bins=40, alpha=0.5, label="Before Calibration")
plt.hist(flat_after_in, bins=40, alpha=0.5, label="After Calibration (Inliers)")
plt.xlabel("Localization Error (m)")
plt.ylabel("Count")
plt.title("Before vs After Calibration — Error Histograms")
plt.legend()
plt.grid(True, linestyle=':')
plt.show()

print("Additional visualizations complete.")

# =========================================
# CELL 6 — 3D Scatter Plot of All MC Results
# =========================================

from mpl_toolkits.mplot3d import Axes3D

print("Generating 3D scatter plot of MC results…")

# Flatten the grid into a long list of points
points_E = []
points_C = []
points_O = []
points_err = []

for ie, num_events in enumerate(NUM_CAL_EVENTS_GRID):
    for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
        for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):

            # Median error at this grid point
            err = median_after_in[ie, ic, io]

            points_E.append(num_events)
            points_C.append(clock_std * 1000)  # convert to ms
            points_O.append(out_frac)
            points_err.append(err)

points_E = np.array(points_E)
points_C = np.array(points_C)
points_O = np.array(points_O)
points_err = np.array(points_err)

# 3D scatter plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(
    points_E,
    points_C,
    points_O,
    c=points_err,
    cmap='viridis',
    s=80,
    alpha=0.9
)

ax.set_xlabel("Num Calibration Events")
ax.set_ylabel("Clock Offset Std (ms)")
ax.set_zlabel("Outlier Fraction")
ax.set_title("3D Scatter of Median Error (After Calibration, Inliers Only)")

cb = fig.colorbar(p, ax=ax, shrink=0.6)
cb.set_label("Median Error (m)")

plt.tight_layout()
plt.show()

print("3D scatter plot complete.")

# =========================================
# CELL 7 — Pareto Frontier (Cost vs Accuracy)
# =========================================

print("Generating Pareto frontier plots…")

def compute_pareto_frontier(costs, errors):
    """
    Given:
      costs  = array of cost values (e.g., num calibration events)
      errors = array of median errors for each cost
    Returns:
      (pareto_costs, pareto_errors)
    The frontier is defined as points where increasing cost
    yields strictly better (lower) error.
    """
    # Sort by cost
    idx = np.argsort(costs)
    costs_sorted = costs[idx]
    errors_sorted = errors[idx]

    pareto_costs = []
    pareto_errors = []

    best_so_far = np.inf
    for c, e in zip(costs_sorted, errors_sorted):
        if e < best_so_far:
            pareto_costs.append(c)
            pareto_errors.append(e)
            best_so_far = e

    return np.array(pareto_costs), np.array(pareto_errors)


# ---------------------------------------------------------
# Compute Pareto frontiers for each (clock_std, outlier_frac)
# ---------------------------------------------------------

plt.figure(figsize=(12,8))

for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
    for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):

        # Extract the error curve for this slice
        errors_slice = median_after_in[:, ic, io]
        costs_slice = NUM_CAL_EVENTS_GRID

        # Compute Pareto frontier
        p_costs, p_errors = compute_pareto_frontier(costs_slice, errors_slice)

        # Plot
        label = f"{clock_std*1000:.0f} ms, out={out_frac:.2f}"
        plt.plot(p_costs, p_errors, marker='o', alpha=0.6, label=label)

plt.xlabel("Calibration Cost (Num Calibration Events)")
plt.ylabel("Median Localization Error (m)")
plt.title("Pareto Frontier — Calibration Cost vs Accuracy\n(After Calibration, Inliers Only)")
plt.grid(True, linestyle=':')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Pareto frontier plots complete.")

# =========================================
# CELL 8 — Sensitivity Analysis (Main Effects)
# =========================================

print("Generating sensitivity analysis…")

# ---------------------------------------------------------
# 1. Compute main effects
# ---------------------------------------------------------

# Mean error vs each parameter (averaged over the other two)
mean_vs_events = np.mean(median_after_in, axis=(1,2))
mean_vs_clock = np.mean(median_after_in, axis=(0,2))
mean_vs_outlier = np.mean(median_after_in, axis=(0,1))

# Normalize sensitivities for comparison
sens_events = (mean_vs_events.max() - mean_vs_events.min())
sens_clock = (mean_vs_clock.max() - mean_vs_clock.min())
sens_outlier = (mean_vs_outlier.max() - mean_vs_outlier.min())

total_sens = sens_events + sens_clock + sens_outlier + 1e-12

norm_events = sens_events / total_sens
norm_clock = sens_clock / total_sens
norm_outlier = sens_outlier / total_sens

# ---------------------------------------------------------
# 2. Plot error vs each parameter
# ---------------------------------------------------------

plt.figure(figsize=(6,4))
plt.plot(NUM_CAL_EVENTS_GRID, mean_vs_events, marker='o')
plt.xlabel("Num Calibration Events")
plt.ylabel("Mean Median Error (m)")
plt.title("Sensitivity: Error vs Calibration Events")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(CLOCK_OFFSET_STD_GRID * 1000, mean_vs_clock, marker='o')
plt.xlabel("Clock Offset Std (ms)")
plt.ylabel("Mean Median Error (m)")
plt.title("Sensitivity: Error vs Clock Drift")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(OUTLIER_FRACTION_GRID, mean_vs_outlier, marker='o')
plt.xlabel("Outlier Fraction")
plt.ylabel("Mean Median Error (m)")
plt.title("Sensitivity: Error vs Outlier Fraction")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Bar chart of normalized sensitivities
# ---------------------------------------------------------

plt.figure(figsize=(6,4))
plt.bar(
    ["Cal Events", "Clock Drift", "Outlier Fraction"],
    [norm_events, norm_clock, norm_outlier],
    color=["#4CAF50", "#2196F3", "#FF5722"]
)
plt.ylabel("Normalized Sensitivity")
plt.title("Relative Influence of Each Parameter")
plt.grid(True, linestyle=':', axis='y')
plt.tight_layout()
plt.show()

print("Sensitivity analysis complete.")
