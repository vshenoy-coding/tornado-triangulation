# Colab cell: joint calibration of per-device clock offsets + calibration event locations,
# then tornado localization before/after offset correction.

!pip install -q numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# -----------------------------
# 1. Global parameters
# -----------------------------
np.random.seed(42)

SPEED_OF_SOUND = 343.0          # m/s
AREA_SIZE = 2000.0              # meters (2 km x 2 km)
NUM_SENSORS = 40
NUM_CAL_EVENTS = 4              # number of calibration events
CAL_NOISE_STD = 0.01            # s, measurement noise during calibration (10 ms)
TORNADO_NOISE_STD = 0.03        # s, measurement noise during tornado (30 ms)
CLOCK_OFFSET_STD = 0.08         # s, per-device true clock offset std (~80 ms)

# ---------------------------------
# 2. Sensor network + true offsets
# ---------------------------------
sensor_x = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)
sensor_y = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)

# True per-device clock offsets (unknown to solver)
true_offsets = np.random.normal(0.0, CLOCK_OFFSET_STD, size=NUM_SENSORS)

# Fix gauge: choose sensor 0 as offset reference = 0
true_offsets -= true_offsets[0]      # now true_offsets[0] == 0

# -------------------------------
# 3. Simulate calibration events
# -------------------------------
# For each calibration event k:
#   true location (x_k, y_k)
#   true onset time t0_k
cal_true_x = np.random.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, NUM_CAL_EVENTS)
cal_true_y = np.random.uniform(0.2*AREA_SIZE, 0.8*AREA_SIZE, NUM_CAL_EVENTS)
cal_true_t0 = np.random.uniform(0.0, 1.0, NUM_CAL_EVENTS)  # arbitrary relative times

# Observed timestamps t_{i,k}
# shape: (NUM_CAL_EVENTS, NUM_SENSORS)
cal_observed = np.zeros((NUM_CAL_EVENTS, NUM_SENSORS))

for k in range(NUM_CAL_EVENTS):
    dx = sensor_x - cal_true_x[k]
    dy = sensor_y - cal_true_y[k]
    dist = np.sqrt(dx**2 + dy**2)
    travel = dist / SPEED_OF_SOUND
    # true arrival + device clock offset + measurement noise
    cal_observed[k, :] = (
        cal_true_t0[k]
        + travel
        + true_offsets
        + np.random.normal(0.0, CAL_NOISE_STD, size=NUM_SENSORS)
    )

# -----------------------------------------
# 4. Build joint calibration least-squares
# -----------------------------------------
# Parameters:
#   offsets b_1..b_{N-1} (sensor 0 fixed to 0)
#   for each cal event k: x_k, y_k, t0_k
#
# Param vector layout:
#   [b_1, ..., b_{N-1}, x_0, y_0, t0_0, x_1, y_1, t0_1, ..., x_{K-1}, y_{K-1}, t0_{K-1}]

num_offsets_params = NUM_SENSORS - 1
num_event_params = NUM_CAL_EVENTS * 3
total_params = num_offsets_params + num_event_params

def unpack_params(params):
    """Unpack flat parameter vector into offsets and per-event (x,y,t0)."""
    b_vec = np.zeros(NUM_SENSORS)
    # sensor 0 offset fixed to 0; others from params
    b_vec[1:] = params[:num_offsets_params]

    event_params = params[num_offsets_params:]
    cal_x_est = np.zeros(NUM_CAL_EVENTS)
    cal_y_est = np.zeros(NUM_CAL_EVENTS)
    cal_t0_est = np.zeros(NUM_CAL_EVENTS)

    for k in range(NUM_CAL_EVENTS):
        base = 3 * k
        cal_x_est[k] = event_params[base + 0]
        cal_y_est[k] = event_params[base + 1]
        cal_t0_est[k] = event_params[base + 2]

    return b_vec, cal_x_est, cal_y_est, cal_t0_est

def calibration_residuals(params, sx, sy, cal_obs, c):
    """
    Residuals over all (event k, sensor i):
      r_{k,i} = observed_{k,i} - (b_i + t0_k + dist_{k,i}/c)
    """
    b_vec, cal_x_est, cal_y_est, cal_t0_est = unpack_params(params)
    num_events, num_sensors = cal_obs.shape
    residuals = []

    for k in range(num_events):
        dx = sx - cal_x_est[k]
        dy = sy - cal_y_est[k]
        dist = np.sqrt(dx**2 + dy**2)
        pred = cal_t0_est[k] + dist / c + b_vec
        res_k = cal_obs[k, :] - pred
        residuals.append(res_k)

    residuals = np.concatenate(residuals)
    return residuals

# Initial guess for parameters
# - Offsets: zero for all (we know they are roughly small)
# - Cal event positions: centroid of sensors
# - Cal t0: min observed per event minus mean travel time
init_offsets = np.zeros(num_offsets_params)

init_cal_x = np.full(NUM_CAL_EVENTS, np.mean(sensor_x))
init_cal_y = np.full(NUM_CAL_EVENTS, np.mean(sensor_y))

init_cal_t0 = np.zeros(NUM_CAL_EVENTS)
for k in range(NUM_CAL_EVENTS):
    # crude guess: subtract median distance / c from min observed
    dx = sensor_x - init_cal_x[k]
    dy = sensor_y - init_cal_y[k]
    dist_guess = np.sqrt(dx**2 + dy**2)
    travel_guess = np.median(dist_guess) / SPEED_OF_SOUND
    init_cal_t0[k] = np.min(cal_observed[k]) - travel_guess

init_params = np.concatenate([
    init_offsets,
    np.ravel(np.column_stack([init_cal_x, init_cal_y, init_cal_t0]))
])

# Run joint calibration LS
res_cal = least_squares(
    calibration_residuals,
    init_params,
    args=(sensor_x, sensor_y, cal_observed, SPEED_OF_SOUND),
    loss='huber',          # robust to a few bad measurements
    f_scale=0.03,
    xtol=1e-10, ftol=1e-10, gtol=1e-10,
    max_nfev=10000
)

est_offsets, est_cal_x, est_cal_y, est_cal_t0 = unpack_params(res_cal.x)

# --------------------------------
# 5. Evaluate calibration quality
# --------------------------------
offset_error = est_offsets - true_offsets
offset_rmse = np.sqrt(np.mean(offset_error**2))
offset_mae = np.mean(np.abs(offset_error))

print("=== Calibration results ===")
print(f"True offsets (first 5):     {true_offsets[:5]}")
print(f"Estimated offsets (first 5):{est_offsets[:5]}")
print(f"Offset MAE:   {offset_mae*1000:.1f} ms")
print(f"Offset RMSE:  {offset_rmse*1000:.1f} ms")

# -----------------------------------
# 6. Tornado event with same offsets
# -----------------------------------
true_tornado_x = 1500.0
true_tornado_y = 1200.0
true_tornado_t0 = 5.0   # arbitrary

dx_t = sensor_x - true_tornado_x
dy_t = sensor_y - true_tornado_y
dist_t = np.sqrt(dx_t**2 + dy_t**2)
true_tornado_toa = true_tornado_t0 + dist_t / SPEED_OF_SOUND

# Phones report raw timestamps including their clock offset + measurement noise
tornado_observed_raw = (
    true_tornado_toa
    + true_offsets
    + np.random.normal(0.0, TORNADO_NOISE_STD, size=NUM_SENSORS)
)

# Corrected timestamps using estimated offsets
tornado_observed_corr = tornado_observed_raw - est_offsets

# ---------------------------------------------------------------------
# 7. Tornado localization solver (TOA LS), before and after correction
# ---------------------------------------------------------------------
def tornado_toa_residuals(params, sx, sy, obs_times, c):
    """
    Residuals: obs_times - (t0 + dist/c)
    """
    x, y, t0 = params
    d = np.sqrt((sx - x)**2 + (sy - y)**2)
    pred = t0 + d / c
    return obs_times - pred

def solve_tornado_location(obs_times, sx, sy, c):
    # initial guess: centroid of earliest arrivals
    k = max(4, int(NUM_SENSORS * 0.2))
    earliest_idx = np.argsort(obs_times)[:k]
    init_x = np.mean(sx[earliest_idx])
    init_y = np.mean(sy[earliest_idx])
    # crude t0 guess
    d_guess = np.sqrt((sx - init_x)**2 + (sy - init_y)**2)
    t0_guess = np.min(obs_times) - np.min(d_guess) / c
    init_params = np.array([init_x, init_y, t0_guess])

    res = least_squares(
        tornado_toa_residuals,
        init_params,
        args=(sx, sy, obs_times, c),
        loss='huber',
        f_scale=TORNADO_NOISE_STD,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=5000
    )
    return res

# Solve with raw (uncorrected) timestamps
res_before = solve_tornado_location(tornado_observed_raw, sensor_x, sensor_y, SPEED_OF_SOUND)
est_x_before, est_y_before, est_t0_before = res_before.x

# Solve with corrected timestamps
res_after = solve_tornado_location(tornado_observed_corr, sensor_x, sensor_y, SPEED_OF_SOUND)
est_x_after, est_y_after, est_t0_after = res_after.x

# Errors
err_before = np.sqrt((est_x_before - true_tornado_x)**2 + (est_y_before - true_tornado_y)**2)
err_after  = np.sqrt((est_x_after  - true_tornado_x)**2 + (est_y_after  - true_tornado_y)**2)

# Output results
print("\n=== Tornado localization ===")
print(f"True tornado location:           ({true_tornado_x:.2f}, {true_tornado_y:.2f}), t0={true_tornado_t0:.3f}")
print(f"Before correction (naive sync):  ({est_x_before:.2f}, {est_y_before:.2f}), t0={est_t0_before:.3f}, error={err_before:.2f} m")
print(f"After correction (calibrated):   ({est_x_after:.2f},  {est_y_after:.2f}),  t0={est_t0_after:.3f},  error={err_after:.2f} m")

# -----------------------------
# 8. Visualization
# -----------------------------
plt.figure(figsize=(10, 8))
plt.scatter(sensor_x, sensor_y, c='blue', alpha=0.6, label='Sensors')

plt.scatter(true_tornado_x, true_tornado_y, c='red', marker='*', s=200, label='True Tornado')

plt.scatter(est_x_before, est_y_before, c='orange', marker='X', s=120,
            label=f'Before calib\nError={err_before:.1f} m')

plt.scatter(est_x_after, est_y_after, c='green', marker='D', s=120,
            label=f'After calib\nError={err_after:.1f} m')

plt.title("Tornado localization before/after clock-offset calibration")
plt.xlabel("Distance East (m)")
plt.ylabel("Distance North (m)")
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Before correction (naively assumes synchronized clocks)

# After correction (using the estimated offsets)
