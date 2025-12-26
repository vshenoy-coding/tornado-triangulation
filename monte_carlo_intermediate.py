# Colab cell: full realistic TOA-based tornado localization + 3D Monte Carlo sweep
# - Multiple calibration events
# - Per-device clock offsets
# - Outlier sensors
# - Two-pass joint calibration (offsets + event locations + onset times)
# - Tornado localization before/after calibration
# - 3D Monte Carlo sweep over:
#     NUM_CAL_EVENTS ∈ [3, 6, 9]
#     CLOCK_OFFSET_STD ∈ [0.03, 0.08, 0.15] s
#     OUTLIER_FRACTION ∈ [0.0, 0.15, 0.3]
# - 8 runs per grid point
# - Robust to SVD failures, with logging and plots

!pip install -q numpy scipy matplotlib tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

# =========================================
# 1. Base configuration (global constants)
# =========================================

AREA_SIZE = 2000.0          # meters, 2 km x 2 km
NUM_SENSORS = 60            # phones / smart devices
BASE_SPEED_OF_SOUND = 343.0 # m/s
SPEED_OF_SOUND_STD = 3.0    # m/s event-to-event variability (temperature, etc.)

# Noise levels
CAL_NOISE_STD = 0.01        # s, measurement noise during calibration
TORNADO_NOISE_STD = 0.03    # s, measurement noise during tornado

# Outlier behavior (bad devices)
OUTLIER_CAL_BIAS = 0.20     # s, systematic bias during calibration
OUTLIER_TORNADO_BIAS = 0.20 # s, systematic bias during tornado
OUTLIER_EXTRA_NOISE = 0.05  # s extra noise

# Calibration outlier detection
CAL_RMS_MULT = 2.5          # threshold multiplier for residual RMS outlier detection

# Tornado fail threshold (for MC failure rate)
TORNADO_FAIL_THRESHOLD_M = 200.0  # meters

# Sentinel error used when a run fails numerically
FAIL_ERR = 9999.0


# =========================================
# 2. Utility functions: geometry + model
# =========================================

def generate_sensor_positions(num, area_size, rng):
    """
    Slightly realistic spatial distribution:
    - half in a central cluster
    - half scattered
    """
    n_cluster = num // 2
    n_scatter = num - n_cluster
    cluster_center = np.array([0.5*area_size, 0.5*area_size])
    cluster_std = 0.15 * area_size

    # Use rng.normal (Generator API) instead of rng.randn
    cluster = cluster_center + rng.normal(0.0, 1.0, size=(n_cluster, 2)) * cluster_std
    cluster = np.clip(cluster, 0, area_size)

    scatter = rng.uniform(0, area_size, size=(n_scatter, 2))
    sensors = np.vstack([cluster, scatter])
    return sensors[:, 0], sensors[:, 1]


def unpack_params(params, num_sensors, num_events):
    """
    Unpack flat parameter vector into:
    - per-sensor offsets b_i (sensor 0 in THIS SET fixed = 0)
    - per-event x_k, y_k, t0_k
    """
    num_offsets = num_sensors - 1
    num_event_params = num_events * 3

    b_vec = np.zeros(num_sensors)
    b_vec[1:] = params[:num_offsets]

    event_params = params[num_offsets:num_offsets + num_event_params]
    cal_x_est = np.zeros(num_events)
    cal_y_est = np.zeros(num_events)
    cal_t0_est = np.zeros(num_events)

    for k in range(num_events):
        base = 3 * k
        cal_x_est[k] = event_params[base + 0]
        cal_y_est[k] = event_params[base + 1]
        cal_t0_est[k] = event_params[base + 2]

    return b_vec, cal_x_est, cal_y_est, cal_t0_est


def calibration_residuals(params, sx, sy, cal_obs, c_base):
    """
    Residuals over all calibration events and sensors.
    Approximate propagation with a fixed speed c_base.
    """
    num_events, num_sensors = cal_obs.shape
    b_vec, cal_x_est, cal_y_est, cal_t0_est = unpack_params(params, num_sensors, num_events)

    residuals = []
    for k in range(num_events):
        dx = sx - cal_x_est[k]
        dy = sy - cal_y_est[k]
        dist = np.sqrt(dx**2 + dy**2)
        pred = cal_t0_est[k] + dist / c_base + b_vec
        res_k = cal_obs[k, :] - pred
        residuals.append(res_k)
    return np.concatenate(residuals)


def calibration_residuals_subset(params, sx, sy, cal_obs, c_base):
    """
    Same as calibration_residuals but for a subset of sensors.
    """
    num_events, num_sensors = cal_obs.shape
    b_vec, cal_x_est, cal_y_est, cal_t0_est = unpack_params(params, num_sensors, num_events)

    residuals = []
    for k in range(num_events):
        dx = sx - cal_x_est[k]
        dy = sy - cal_y_est[k]
        dist = np.sqrt(dx**2 + dy**2)
        pred = cal_t0_est[k] + dist / c_base + b_vec
        res_k = cal_obs[k, :] - pred
        residuals.append(res_k)
    return np.concatenate(residuals)


def tornado_toa_residuals(params, sx, sy, obs_times, c):
    x, y, t0 = params
    d = np.sqrt((sx - x)**2 + (sy - y)**2)
    pred = t0 + d / c
    return obs_times - pred


def solve_tornado(obs_times, sx, sy, c, robust_f_scale, mask=None):
    """
    Robust tornado TOA solver with optional sensor mask.
    Returns LS result and used sx, sy.
    """
    if mask is not None:
        idx = np.where(mask)[0]
        sx = sx[idx]
        sy = sy[idx]
        obs_times = obs_times[idx]

    # If too few sensors, bail out with failure
    if len(sx) < 4:
        return None, None, None

    # Initial guess: centroid of earliest arrivals
    k = max(4, int(len(sx) * 0.2))
    earliest_idx = np.argsort(obs_times)[:k]
    init_x = np.mean(sx[earliest_idx])
    init_y = np.mean(sy[earliest_idx])
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


# =========================================
# 3. Single simulation function
# =========================================

def run_single_simulation(
    num_cal_events,
    clock_offset_std,
    outlier_fraction,
    rng_seed=None,
    verbose=False
):
    """
    Run one full simulation:
    - Generate sensor network
    - Assign true clock offsets
    - Simulate calibration events
    - Two-pass calibration (offsets + event params)
    - Simulate tornado event
    - Tornado localization before/after calibration
    Returns:
      (err_before, err_after_all, err_after_inliers)
    Or (FAIL_ERR, FAIL_ERR, FAIL_ERR) if something breaks numerically.
    """
    rng = np.random.default_rng(rng_seed)

    # --- Sensors ---
    sensor_x, sensor_y = generate_sensor_positions(NUM_SENSORS, AREA_SIZE, rng)

    # --- True offsets ---
    true_offsets = rng.normal(0.0, clock_offset_std, size=NUM_SENSORS)
    true_offsets -= true_offsets[0]  # fix gauge: sensor 0 is reference

    # --- Outliers ---
    num_outliers = int(np.round(NUM_SENSORS * outlier_fraction))
    indices = np.arange(NUM_SENSORS)
    rng.shuffle(indices)
    outlier_indices = indices[:num_outliers]
    inlier_indices = indices[num_outliers:]

    # --- Calibration events ---
    cal_true_x = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_y = rng.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, num_cal_events)
    cal_true_t0 = rng.uniform(0.0, 2.0, num_cal_events)
    cal_speeds = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD, size=num_cal_events)

    cal_observed = np.zeros((num_cal_events, NUM_SENSORS))

    for k in range(num_cal_events):
        c_k = cal_speeds[k]
        dx = sensor_x - cal_true_x[k]
        dy = sensor_y - cal_true_y[k]
        dist = np.sqrt(dx**2 + dy**2)
        travel = dist / c_k
        base_arrival = cal_true_t0[k] + travel

        # inliers
        cal_observed[k, inlier_indices] = (
            base_arrival[inlier_indices]
            + true_offsets[inlier_indices]
            + rng.normal(0.0, CAL_NOISE_STD, size=len(inlier_indices))
        )

        # outliers
        cal_observed[k, outlier_indices] = (
            base_arrival[outlier_indices]
            + true_offsets[outlier_indices]
            + OUTLIER_CAL_BIAS
            + rng.normal(0.0, CAL_NOISE_STD + OUTLIER_EXTRA_NOISE, size=len(outlier_indices))
        )

    # --- Calibration: full set ---
    num_offsets_full = NUM_SENSORS - 1
    init_offsets_full = np.zeros(num_offsets_full)
    init_cal_x_full = np.full(num_cal_events, np.mean(sensor_x))
    init_cal_y_full = np.full(num_cal_events, np.mean(sensor_y))
    init_cal_t0_full = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sensor_x - init_cal_x_full[k]
        dy = sensor_y - init_cal_y_full[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        travel_guess = np.median(dist_guess) / BASE_SPEED_OF_SOUND
        init_cal_t0_full[k] = np.min(cal_observed[k]) - travel_guess

    init_params_full = np.concatenate([
        init_offsets_full,
        np.ravel(np.column_stack([init_cal_x_full, init_cal_y_full, init_cal_t0_full]))
    ])

    try:
        res_cal_all = least_squares(
            calibration_residuals,
            init_params_full,
            args=(sensor_x, sensor_y, cal_observed, BASE_SPEED_OF_SOUND),
            loss='huber',
            f_scale=0.03,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=20000
        )
    except Exception as e:
        if verbose:
            print("Full calibration failed:", e)
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    est_offsets_all, est_cal_x_all, est_cal_y_all, est_cal_t0_all = \
        unpack_params(res_cal_all.x, NUM_SENSORS, num_cal_events)

    # Per-sensor residual RMS
    num_events, num_sensors = cal_observed.shape
    residuals_all_flat = calibration_residuals(
        res_cal_all.x, sensor_x, sensor_y, cal_observed, BASE_SPEED_OF_SOUND
    )
    residuals_all = residuals_all_flat.reshape(num_events, num_sensors)
    sensor_rms = np.sqrt(np.mean(residuals_all**2, axis=0))

    med_rms = np.median(sensor_rms)
    mad_rms = np.median(np.abs(sensor_rms - med_rms)) + 1e-9
    rms_threshold = med_rms + CAL_RMS_MULT * mad_rms

    cal_inlier_sensors = np.where(sensor_rms <= rms_threshold)[0]
    cal_outlier_sensors = np.where(sensor_rms > rms_threshold)[0]

    # If we lost almost all sensors, bail out
    if len(cal_inlier_sensors) < 4:
        if verbose:
            print("Too few calibration inliers, failing run.")
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    # --- Second pass: inlier-only calibration ---
    cal_obs_in = cal_observed[:, cal_inlier_sensors]
    sx_in = sensor_x[cal_inlier_sensors]
    sy_in = sensor_y[cal_inlier_sensors]

    num_inliers = len(cal_inlier_sensors)
    num_offsets_in = num_inliers - 1

    init_offsets_in = np.zeros(num_offsets_in)
    init_cal_x_in = np.full(num_cal_events, np.mean(sx_in))
    init_cal_y_in = np.full(num_cal_events, np.mean(sy_in))
    init_cal_t0_in = np.zeros(num_cal_events)

    for k in range(num_cal_events):
        dx = sx_in - init_cal_x_in[k]
        dy = sy_in - init_cal_y_in[k]
        dist_guess = np.sqrt(dx**2 + dy**2)
        travel_guess = np.median(dist_guess) / BASE_SPEED_OF_SOUND
        init_cal_t0_in[k] = np.min(cal_obs_in[k]) - travel_guess

    init_params_in = np.concatenate([
        init_offsets_in,
        np.ravel(np.column_stack([init_cal_x_in, init_cal_y_in, init_cal_t0_in]))
    ])

    try:
        res_cal_inliers = least_squares(
            calibration_residuals_subset,
            init_params_in,
            args=(sx_in, sy_in, cal_obs_in, BASE_SPEED_OF_SOUND),
            loss='huber',
            f_scale=0.03,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            max_nfev=20000
        )
    except Exception as e:
        if verbose:
            print("Inlier-only calibration failed:", e)
        return FAIL_ERR, FAIL_ERR, FAIL_ERR

    est_offsets_in_sub, est_cal_x_in, est_cal_y_in, est_cal_t0_in = \
        unpack_params(res_cal_inliers.x, num_inliers, num_cal_events)

    # Map inlier-only offsets back to all sensors
    est_offsets_final = np.zeros(NUM_SENSORS)
    est_offsets_final[:] = np.nan

    ref_inlier_idx = cal_inlier_sensors[0]
    est_offsets_final[ref_inlier_idx] = 0.0
    for local_idx, global_idx in enumerate(cal_inlier_sensors[1:], start=1):
        est_offsets_final[global_idx] = est_offsets_in_sub[local_idx]

    # Outliers: no calibration (offset ~ 0, robust LS will downweight)
    for idx in cal_outlier_sensors:
        if np.isnan(est_offsets_final[idx]):
            est_offsets_final[idx] = 0.0

    # Offset error metrics (not strictly needed for MC, but useful)
    offset_error = est_offsets_final - true_offsets
    offset_mae = np.mean(np.abs(offset_error))
    offset_rmse = np.sqrt(np.mean(offset_error**2))

    if verbose:
        print(f"  Calibration: {num_cal_events} events, clock_std={clock_offset_std*1000:.0f} ms, "
              f"outlier_frac={outlier_fraction:.2f}")
        print(f"    Simulated outliers:    {len(outlier_indices)}")
        print(f"    Detected cal outliers: {len(cal_outlier_sensors)}")
        print(f"    Offset MAE:            {offset_mae*1000:.1f} ms")
        print(f"    Offset RMSE:           {offset_rmse*1000:.1f} ms")

    # --- Tornado event ---
    true_tornado_x = 1500.0
    true_tornado_y = 1200.0
    true_tornado_t0 = 10.0

    tornado_speed = rng.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD)
    dx_t = sensor_x - true_tornado_x
    dy_t = sensor_y - true_tornado_y
    dist_t = np.sqrt(dx_t**2 + dy_t**2)
    true_tornado_toa = true_tornado_t0 + dist_t / tornado_speed

    tornado_observed_raw = np.zeros(NUM_SENSORS)
    # inliers
    tornado_observed_raw[inlier_indices] = (
        true_tornado_toa[inlier_indices]
        + true_offsets[inlier_indices]
        + rng.normal(0.0, TORNADO_NOISE_STD, size=len(inlier_indices))
    )
    # outliers
    tornado_observed_raw[outlier_indices] = (
        true_tornado_toa[outlier_indices]
        + true_offsets[outlier_indices]
        + OUTLIER_TORNADO_BIAS
        + rng.normal(0.0, TORNADO_NOISE_STD + OUTLIER_EXTRA_NOISE, size=len(outlier_indices))
    )

    tornado_observed_corr = tornado_observed_raw - est_offsets_final

    # --- Tornado localization: before calibration ---
    res_before, _, _ = solve_tornado(
        tornado_observed_raw, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    if res_before is None:
        err_before = FAIL_ERR
    else:
        est_x_before, est_y_before, est_t_before = res_before.x
        err_before = np.sqrt((est_x_before - true_tornado_x)**2 + (est_y_before - true_tornado_y)**2)

    # --- After calibration: all sensors ---
    res_after_all, _, _ = solve_tornado(
        tornado_observed_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2
    )
    if res_after_all is None:
        err_after_all = FAIL_ERR
    else:
        est_x_after_all, est_y_after_all, est_t_after_all = res_after_all.x
        err_after_all = np.sqrt((est_x_after_all - true_tornado_x)**2 + (est_y_after_all - true_tornado_y)**2)

    # --- After calibration: only calibration inliers ---
    inlier_mask = np.zeros(NUM_SENSORS, dtype=bool)
    inlier_mask[cal_inlier_sensors] = True
    res_after_in, _, _ = solve_tornado(
        tornado_observed_corr, sensor_x, sensor_y,
        BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD * 2,
        mask=inlier_mask
    )
    if res_after_in is None:
        err_after_in = FAIL_ERR
    else:
        est_x_after_in, est_y_after_in, est_t_after_in = res_after_in.x
        err_after_in = np.sqrt((est_x_after_in - true_tornado_x)**2 + (est_y_after_in - true_tornado_y)**2)

    if verbose:
        print(f"    Tornado error before:          {err_before:.1f} m")
        print(f"    Tornado error after (all):     {err_after_all:.1f} m")
        print(f"    Tornado error after (inliers): {err_after_in:.1f} m")
        print()

    return err_before, err_after_all, err_after_in


# =========================================
# 4. Monte Carlo sweep (MC-A-2)
# =========================================

# Parameter grids
NUM_CAL_EVENTS_GRID = np.array([3, 6, 9])
CLOCK_OFFSET_STD_GRID = np.array([0.03, 0.08, 0.15])  # in seconds
OUTLIER_FRACTION_GRID = np.array([0.0, 0.15, 0.3])

NUM_MC_SAMPLES = 8  # runs per grid point

# Allocate result tensors: shape (Ne, Nc, No, Nmc)
Ne = len(NUM_CAL_EVENTS_GRID)
Nc = len(CLOCK_OFFSET_STD_GRID)
No = len(OUTLIER_FRACTION_GRID)
Nm = NUM_MC_SAMPLES

errors_before = np.zeros((Ne, Nc, No, Nm))
errors_after_all = np.zeros((Ne, Nc, No, Nm))
errors_after_in = np.zeros((Ne, Nc, No, Nm))

print("=== Starting Monte Carlo sweep (MC-A-2) ===")
print(f"Grid: {Ne} (events) x {Nc} (clock std) x {No} (outlier frac) x {Nm} (MC samples)")
print(f"Total runs: {Ne * Nc * No * Nm}")
print()

run_idx = 0
total_runs = Ne * Nc * No * Nm

with tqdm(total=total_runs, desc="Monte Carlo runs") as pbar:
    for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
        for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
            for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):
                if num_cal_events < 2:
                    continue
                for im in range(NUM_MC_SAMPLES):
                    seed = 1000 + 100*ie + 10*ic + 1000*io + im
                    err_b, err_a_all, err_a_in = run_single_simulation(
                        num_cal_events=num_cal_events,
                        clock_offset_std=clock_std,
                        outlier_fraction=out_frac,
                        rng_seed=seed,
                        verbose=False  # set True to see per-run details
                    )
                    errors_before[ie, ic, io, im] = err_b
                    errors_after_all[ie, ic, io, im] = err_a_all
                    errors_after_in[ie, ic, io, im] = err_a_in

                    run_idx += 1
                    pbar.update(1)

print("\n=== Monte Carlo sweep complete ===\n")

# =========================================
# 5. Aggregate statistics
# =========================================

# Clip or treat FAIL_ERR as failure; we keep them but failure-rate will capture them.
median_before = np.median(errors_before, axis=3)
median_after_all = np.median(errors_after_all, axis=3)
median_after_in = np.median(errors_after_in, axis=3)

# Failure = error_after_in > threshold OR == FAIL_ERR
fail_before = np.mean((errors_before > TORNADO_FAIL_THRESHOLD_M) | (errors_before >= FAIL_ERR), axis=3)
fail_after_all = np.mean((errors_after_all > TORNADO_FAIL_THRESHOLD_M) | (errors_after_all >= FAIL_ERR), axis=3)
fail_after_in = np.mean((errors_after_in > TORNADO_FAIL_THRESHOLD_M) | (errors_after_in >= FAIL_ERR), axis=3)

print("Median tornado localization error (m) AFTER calibration (inliers only):")
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    print(f"\nNUM_CAL_EVENTS = {num_cal_events}:")
    for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
        line = f"  clock_std={clock_std*1000:.0f} ms: "
        vals = median_after_in[ie, ic, :]
        line += " | ".join(
            f"outlier_frac={OUTLIER_FRACTION_GRID[io]:.2f}: {vals[io]:.1f}"
            for io in range(No)
        )
        print(line)

print("\nFailure rate AFTER calibration (inliers only) [threshold {:.0f} m]:".format(TORNADO_FAIL_THRESHOLD_M))
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    print(f"\nNUM_CAL_EVENTS = {num_cal_events}:")
    for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
        line = f"  clock_std={clock_std*1000:.0f} ms: "
        vals = fail_after_in[ie, ic, :]
        line += " | ".join(
            f"outlier_frac={OUTLIER_FRACTION_GRID[io]:.2f}: {vals[io]*100:.0f}%"
            for io in range(No)
        )
        print(line)

# =========================================
# 6. Visualization: heatmaps for slices
# =========================================

def plot_heatmap_slice(
    data, x_vals, y_vals,
    x_label, y_label, title, vmin=None, vmax=None, cmap='viridis'
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


# 6.1: For each OUTLIER_FRACTION, plot NUM_CAL_EVENTS vs CLOCK_OFFSET_STD
for io, out_frac in enumerate(OUTLIER_FRACTION_GRID):
    data = median_after_in[:, :, io]
    plot_heatmap_slice(
        data,
        x_vals=CLOCK_OFFSET_STD_GRID * 1000.0,
        y_vals=NUM_CAL_EVENTS_GRID,
        x_label='Clock offset std (ms)',
        y_label='Num calibration events',
        title=f'Median error after calib (inliers)\nvs events & clock_std, outlier_frac={out_frac:.2f}'
    )

# 6.2: For each CLOCK_OFFSET_STD, plot NUM_CAL_EVENTS vs OUTLIER_FRACTION
for ic, clock_std in enumerate(CLOCK_OFFSET_STD_GRID):
    data = median_after_in[:, ic, :]
    plot_heatmap_slice(
        data,
        x_vals=OUTLIER_FRACTION_GRID,
        y_vals=NUM_CAL_EVENTS_GRID,
        x_label='Outlier fraction',
        y_label='Num calibration events',
        title=f'Median error after calib (inliers)\nvs events & outliers, clock_std={clock_std*1000:.0f} ms'
    )

# 6.3: For each NUM_CAL_EVENTS, plot CLOCK_OFFSET_STD vs OUTLIER_FRACTION
for ie, num_cal_events in enumerate(NUM_CAL_EVENTS_GRID):
    data = median_after_in[ie, :, :]
    plot_heatmap_slice(
        data,
        x_vals=OUTLIER_FRACTION_GRID,
        y_vals=CLOCK_OFFSET_STD_GRID * 1000.0,
        x_label='Outlier fraction',
        y_label='Clock offset std (ms)',
        title=f'Median error after calib (inliers)\nvs outliers & clock_std, events={num_cal_events}'
    )

# =========================================
# 7. Summary curves (averaged over other dimensions)
# =========================================

# Average over other two dimensions
mean_over_c_o = np.mean(median_after_in, axis=(1,2))  # vs NUM_CAL_EVENTS
mean_over_e_o = np.mean(median_after_in, axis=(0,2))  # vs CLOCK_OFFSET_STD
mean_over_e_c_outlier = np.mean(median_after_in, axis=(0,1))  # vs OUTLIER_FRACTION

plt.figure(figsize=(5,4))
plt.plot(NUM_CAL_EVENTS_GRID, mean_over_c_o, marker='o')
plt.xlabel('Num calibration events')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Num calibration events\n(averaged over clock std & outliers)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(CLOCK_OFFSET_STD_GRID * 1000.0, mean_over_e_o, marker='o')
plt.xlabel('Clock offset std (ms)')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Clock offset std\n(averaged over events & outliers)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.plot(OUTLIER_FRACTION_GRID, mean_over_e_c_outlier, marker='o')
plt.xlabel('Outlier fraction')
plt.ylabel('Mean median error (m)')
plt.title('Error vs Outlier fraction\n(averaged over events & clock std)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nDone. You now have 3D error and failure-rate landscapes over:")
print("  - NUM_CAL_EVENTS ∈", NUM_CAL_EVENTS_GRID)
print("  - CLOCK_OFFSET_STD ∈", CLOCK_OFFSET_STD_GRID, "seconds")
print("  - OUTLIER_FRACTION ∈", OUTLIER_FRACTION_GRID)
print("Where FAIL_ERR =", FAIL_ERR, "is used for numerically failed runs.")
