# Colab cell: realistic TOA-based tornado localization with
# - multiple calibration events
# - per-device clock offsets
# - outlier sensors
# - robust joint calibration (offsets + event locations + event onset times)
# - two-pass calibration (full set, then inliers only)
# - tornado localization before/after calibration + outlier rejection

!pip install -q numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =========================================
# 1. Global parameters (TUNE THESE)
# =========================================
np.random.seed(42)

AREA_SIZE = 2000.0          # meters, 2 km x 2 km urban area
NUM_SENSORS = 60            # phones / smart devices
NUM_CAL_EVENTS = 6          # number of calibration events (increase this!)
BASE_SPEED_OF_SOUND = 343.0 # m/s
SPEED_OF_SOUND_STD = 3.0    # m/s event-to-event variability (temperature, etc.)

# Clock behavior
CLOCK_OFFSET_STD = 0.08     # s, per-device true clock offset std (~80 ms); vary this
CAL_NOISE_STD = 0.01        # s, measurement noise during calibration (10 ms)
TORNADO_NOISE_STD = 0.03    # s, measurement noise during tornado (30 ms)

# Outlier sensors (broken devices / bad mics / weird clocks)
OUTLIER_FRACTION = 0.15     # fraction of sensors that are "bad"
OUTLIER_CAL_BIAS = 0.20     # s, systematic bias during calibration
OUTLIER_TORNADO_BIAS = 0.20 # s, systematic bias during tornado
OUTLIER_EXTRA_NOISE = 0.05  # s additional noise

# Robustness / thresholds
CAL_RMS_MULT = 2.5          # threshold multiplier for residual RMS outlier detection


# =========================================
# 2. Sensor network and true offsets
# =========================================

def generate_sensor_positions(num, area_size):
    """
    Slightly realistic spatial distribution:
    - half in a central cluster
    - half scattered
    """
    n_cluster = num // 2
    n_scatter = num - n_cluster
    cluster_center = np.array([0.5*area_size, 0.5*area_size])
    cluster_std = 0.15 * area_size

    cluster = cluster_center + np.random.randn(n_cluster, 2) * cluster_std
    cluster = np.clip(cluster, 0, area_size)

    scatter = np.random.uniform(0, area_size, size=(n_scatter, 2))
    sensors = np.vstack([cluster, scatter])
    return sensors[:, 0], sensors[:, 1]

sensor_x, sensor_y = generate_sensor_positions(NUM_SENSORS, AREA_SIZE)

# True per-device clock offsets
true_offsets = np.random.normal(0.0, CLOCK_OFFSET_STD, size=NUM_SENSORS)
true_offsets -= true_offsets[0]  # fix gauge: sensor 0 offset = 0

# Mark outlier sensors
num_outliers = int(np.round(NUM_SENSORS * OUTLIER_FRACTION))
all_indices = np.arange(NUM_SENSORS)
np.random.shuffle(all_indices)
outlier_indices = all_indices[:num_outliers]
inlier_indices = all_indices[num_outliers:]


# =========================================
# 3. Simulate calibration events
# =========================================

cal_true_x = np.random.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, NUM_CAL_EVENTS)
cal_true_y = np.random.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, NUM_CAL_EVENTS)
cal_true_t0 = np.random.uniform(0.0, 2.0, NUM_CAL_EVENTS)

# event-dependent speed of sound (weather variation)
cal_speeds = np.random.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD, size=NUM_CAL_EVENTS)

# observed timestamps: cal_observed[k, i] = t_{i,k}
cal_observed = np.zeros((NUM_CAL_EVENTS, NUM_SENSORS))

for k in range(NUM_CAL_EVENTS):
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
        + np.random.normal(0.0, CAL_NOISE_STD, size=len(inlier_indices))
    )

    # outliers (broken sensors)
    cal_observed[k, outlier_indices] = (
        base_arrival[outlier_indices]
        + true_offsets[outlier_indices]
        + OUTLIER_CAL_BIAS
        + np.random.normal(0.0, CAL_NOISE_STD + OUTLIER_EXTRA_NOISE, size=len(outlier_indices))
    )


# ===================================================
# 4. Joint calibration model (offsets + event params)
# ===================================================

def unpack_params(params, num_sensors, num_events):
    """
    Unpack flat parameter vector into:
    - per-sensor offsets b_i (sensor 0 in this set fixed = 0)
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
    Residuals over all calibration events and sensors (full set).
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


# Initial guess for full calibration
num_offsets_params_full = NUM_SENSORS - 1
init_offsets_full = np.zeros(num_offsets_params_full)
init_cal_x_full = np.full(NUM_CAL_EVENTS, np.mean(sensor_x))
init_cal_y_full = np.full(NUM_CAL_EVENTS, np.mean(sensor_y))
init_cal_t0_full = np.zeros(NUM_CAL_EVENTS)

for k in range(NUM_CAL_EVENTS):
    dx = sensor_x - init_cal_x_full[k]
    dy = sensor_y - init_cal_y_full[k]
    dist_guess = np.sqrt(dx**2 + dy**2)
    travel_guess = np.median(dist_guess) / BASE_SPEED_OF_SOUND
    init_cal_t0_full[k] = np.min(cal_observed[k]) - travel_guess

init_params_full = np.concatenate([
    init_offsets_full,
    np.ravel(np.column_stack([init_cal_x_full, init_cal_y_full, init_cal_t0_full]))
])

# First robust calibration fit (all sensors)
res_cal_all = least_squares(
    calibration_residuals,
    init_params_full,
    args=(sensor_x, sensor_y, cal_observed, BASE_SPEED_OF_SOUND),
    loss='huber',
    f_scale=0.03,
    xtol=1e-10, ftol=1e-10, gtol=1e-10,
    max_nfev=20000
)

est_offsets_all, est_cal_x_all, est_cal_y_all, est_cal_t0_all = \
    unpack_params(res_cal_all.x, NUM_SENSORS, NUM_CAL_EVENTS)

# Per-sensor residual RMS for outlier detection
num_events, num_sensors = cal_observed.shape
residuals_all_flat = calibration_residuals(res_cal_all.x, sensor_x, sensor_y, cal_observed, BASE_SPEED_OF_SOUND)
residuals_all = residuals_all_flat.reshape(NUM_CAL_EVENTS, NUM_SENSORS)
sensor_rms = np.sqrt(np.mean(residuals_all**2, axis=0))

med_rms = np.median(sensor_rms)
mad_rms = np.median(np.abs(sensor_rms - med_rms)) + 1e-9
rms_threshold = med_rms + CAL_RMS_MULT * mad_rms

cal_inlier_sensors = np.where(sensor_rms <= rms_threshold)[0]
cal_outlier_sensors = np.where(sensor_rms > rms_threshold)[0]

# Second pass: re-calibrate using only inlier sensors (re-estimate event params too, A1)
cal_obs_in = cal_observed[:, cal_inlier_sensors]
sx_in = sensor_x[cal_inlier_sensors]
sy_in = sensor_y[cal_inlier_sensors]

num_inliers = len(cal_inlier_sensors)
num_offsets_params_in = num_inliers - 1

# Initial guess for inlier-only calibration
init_offsets_in = np.zeros(num_offsets_params_in)
init_cal_x_in = np.full(NUM_CAL_EVENTS, np.mean(sx_in))
init_cal_y_in = np.full(NUM_CAL_EVENTS, np.mean(sy_in))
init_cal_t0_in = np.zeros(NUM_CAL_EVENTS)
for k in range(NUM_CAL_EVENTS):
    dx = sx_in - init_cal_x_in[k]
    dy = sy_in - init_cal_y_in[k]
    dist_guess = np.sqrt(dx**2 + dy**2)
    travel_guess = np.median(dist_guess) / BASE_SPEED_OF_SOUND
    init_cal_t0_in[k] = np.min(cal_obs_in[k]) - travel_guess

init_params_in = np.concatenate([
    init_offsets_in,
    np.ravel(np.column_stack([init_cal_x_in, init_cal_y_in, init_cal_t0_in]))
])

res_cal_inliers = least_squares(
    calibration_residuals_subset,
    init_params_in,
    args=(sx_in, sy_in, cal_obs_in, BASE_SPEED_OF_SOUND),
    loss='huber',
    f_scale=0.03,
    xtol=1e-10, ftol=1e-10, gtol=1e-10,
    max_nfev=20000
)

est_offsets_in_sub, est_cal_x_in, est_cal_y_in, est_cal_t0_in = \
    unpack_params(res_cal_inliers.x, num_inliers, NUM_CAL_EVENTS)

# Map inlier-only offset estimates back to full sensor set
est_offsets_final = np.zeros(NUM_SENSORS)
est_offsets_final[:] = np.nan

# sensor 0 in the inlier set is defined as reference offset = 0
ref_inlier_idx = cal_inlier_sensors[0]
est_offsets_final[ref_inlier_idx] = 0.0
# others from est_offsets_in_sub (index 1..)
for local_idx, global_idx in enumerate(cal_inlier_sensors[1:], start=1):
    est_offsets_final[global_idx] = est_offsets_in_sub[local_idx]

# Outlier sensors get no calibration; we set their estimated offset to 0
for idx in cal_outlier_sensors:
    if np.isnan(est_offsets_final[idx]):
        est_offsets_final[idx] = 0.0

# Compare true vs estimated offsets (for all sensors)
offset_error = est_offsets_final - true_offsets
offset_mae = np.mean(np.abs(offset_error))
offset_rmse = np.sqrt(np.mean(offset_error**2))

# Output results
print("=== Calibration summary ===")
print(f"True offsets (first 5):      {true_offsets[:5]}")
print(f"Estimated offsets (first 5): {est_offsets_final[:5]}")
print(f"Number of sensors:           {NUM_SENSORS}")
print(f"Simulated outliers:          {len(outlier_indices)}")
print(f"Detected cal outliers:       {len(cal_outlier_sensors)}")
print(f"Offset MAE:                  {offset_mae*1000:.1f} ms")
print(f"Offset RMSE:                 {offset_rmse*1000:.1f} ms")


# =========================================
# 5. Tornado event using same offsets
# =========================================

true_tornado_x = 1500.0
true_tornado_y = 1200.0
true_tornado_t0 = 10.0

tornado_speed = np.random.normal(BASE_SPEED_OF_SOUND, SPEED_OF_SOUND_STD)
dx_t = sensor_x - true_tornado_x
dy_t = sensor_y - true_tornado_y
dist_t = np.sqrt(dx_t**2 + dy_t**2)
true_tornado_toa = true_tornado_t0 + dist_t / tornado_speed

tornado_observed_raw = np.zeros(NUM_SENSORS)
# inliers
tornado_observed_raw[inlier_indices] = (
    true_tornado_toa[inlier_indices]
    + true_offsets[inlier_indices]
    + np.random.normal(0.0, TORNADO_NOISE_STD, size=len(inlier_indices))
)
# outliers
tornado_observed_raw[outlier_indices] = (
    true_tornado_toa[outlier_indices]
    + true_offsets[outlier_indices]
    + OUTLIER_TORNADO_BIAS
    + np.random.normal(0.0, TORNADO_NOISE_STD + OUTLIER_EXTRA_NOISE, size=len(outlier_indices))
)

tornado_observed_corr = tornado_observed_raw - est_offsets_final


# ===================================================
# 6. Tornado localization (before/after calibration)
# ===================================================

def tornado_toa_residuals(params, sx, sy, obs_times, c):
    x, y, t0 = params
    d = np.sqrt((sx - x)**2 + (sy - y)**2)
    pred = t0 + d / c
    return obs_times - pred

def solve_tornado(obs_times, sx, sy, c, robust_f_scale, mask=None):
    if mask is not None:
        idx = np.where(mask)[0]
        sx = sx[idx]
        sy = sy[idx]
        obs_times = obs_times[idx]

    k = max(4, int(len(sx) * 0.2))
    earliest_idx = np.argsort(obs_times)[:k]
    init_x = np.mean(sx[earliest_idx])
    init_y = np.mean(sy[earliest_idx])
    d_guess = np.sqrt((sx - init_x)**2 + (sy - init_y)**2)
    t0_guess = np.min(obs_times) - np.min(d_guess) / c
    init_params = np.array([init_x, init_y, t0_guess])

    res = least_squares(
        tornado_toa_residuals,
        init_params,
        args=(sx, sy, obs_times, c),
        loss='huber',
        f_scale=robust_f_scale,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=10000
    )
    return res

# naive (no calibration)
res_before = solve_tornado(
    tornado_observed_raw, sensor_x, sensor_y,
    BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD*2
)
est_x_before, est_y_before, est_t_before = res_before.x
err_before = np.sqrt((est_x_before - true_tornado_x)**2 + (est_y_before - true_tornado_y)**2)

# after calibration, using all sensors
res_after_all = solve_tornado(
    tornado_observed_corr, sensor_x, sensor_y,
    BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD*2
)
est_x_after_all, est_y_after_all, est_t_after_all = res_after_all.x
err_after_all = np.sqrt((est_x_after_all - true_tornado_x)**2 + (est_y_after_all - true_tornado_y)**2)

# after calibration, using only calibration inliers
inlier_mask = np.zeros(NUM_SENSORS, dtype=bool)
inlier_mask[cal_inlier_sensors] = True
res_after_in = solve_tornado(
    tornado_observed_corr, sensor_x, sensor_y,
    BASE_SPEED_OF_SOUND, TORNADO_NOISE_STD*2,
    mask=inlier_mask
)
est_x_after_in, est_y_after_in, est_t_after_in = res_after_in.x
err_after_in = np.sqrt((est_x_after_in - true_tornado_x)**2 + (est_y_after_in - true_tornado_y)**2)

# Output results
print("\n=== Tornado localization summary ===")
print(f"True tornado location:            ({true_tornado_x:.2f}, {true_tornado_y:.2f}), t0={true_tornado_t0:.3f}")
print(f"Before calibration (naive):       ({est_x_before:.2f}, {est_y_before:.2f}),  t0={est_t_before:.3f},  error={err_before:.2f} m")
print(f"After calibration (all sensors):  ({est_x_after_all:.2f}, {est_y_after_all:.2f}), t0={est_t_after_all:.3f}, error={err_after_all:.2f} m")
print(f"After calibration (cal inliers):  ({est_x_after_in:.2f}, {est_y_after_in:.2f}),  t0={est_t_after_in:.3f},  error={err_after_in:.2f} m")


# =========================================
# 7. Visualization
# =========================================

plt.figure(figsize=(10,8))
plt.scatter(sensor_x, sensor_y, c='lightgray', label='All sensors')
plt.scatter(sensor_x[cal_inlier_sensors], sensor_y[cal_inlier_sensors],
            c='blue', label='Calibration inliers')
plt.scatter(sensor_x[cal_outlier_sensors], sensor_y[cal_outlier_sensors],
            c='red', marker='x', label='Calibration outliers')

plt.scatter(true_tornado_x, true_tornado_y, c='black', marker='*', s=200, label='True Tornado')

plt.scatter(est_x_before, est_y_before, c='orange', marker='X', s=120,
            label=f'Before calib\nerr={err_before:.1f} m')

plt.scatter(est_x_after_all, est_y_after_all, c='green', marker='D', s=120,
            label=f'After calib (all)\nerr={err_after_all:.1f} m')

plt.scatter(est_x_after_in, est_y_after_in, c='purple', marker='^', s=120,
            label=f'After calib (inliers)\nerr={err_after_in:.1f} m')

plt.title("Realistic tornado localization\nwith clock offsets, outliers, and two-pass calibration")
plt.xlabel("Distance East (m)")
plt.ylabel("Distance North (m)")
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Calibration residual RMS per sensor
plt.figure(figsize=(8,4))
plt.stem(np.arange(NUM_SENSORS), sensor_rms) 
plt.setp(plt.gca().lines[1], visible=False) # hide baseline 
plt.axhline(rms_threshold, color='red', linestyle='--', label='RMS threshold')
plt.title("Calibration residual RMS per sensor")
plt.xlabel("Sensor index")
plt.ylabel("RMS residual (s)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
