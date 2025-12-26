# Colab cell: original TOA simulation + original Nelder-Mead solver vs improved TOA least-squares solver.

!pip install -q numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares

# 1. Simulation Setup
np.random.seed(42)  # For reproducibility

# Constants
SPEED_OF_SOUND = 343.0  # m/s
NUM_SENSORS = 50
AREA_SIZE = 2000  # 2km x 2km neighborhood
NOISE_STD_DEV = 0.05  # 50ms timing jitter

# True Tornado Source (unknown to the solver)
true_source_x = 1500.0
true_source_y = 1200.0
true_onset_time = 0.0  # Relative start time

# 2. Generate Sensor Network
sensor_x = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)
sensor_y = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)

# 3. Simulate Measurements (The "Observation")
distances = np.sqrt((sensor_x - true_source_x)**2 + (sensor_y - true_source_y)**2)
true_arrival_times = true_onset_time + distances / SPEED_OF_SOUND
observed_arrival_times = true_arrival_times + np.random.normal(0, NOISE_STD_DEV, NUM_SENSORS)

# -----------------------------------------
# A. Original time delay of arrival solver 
# -----------------------------------------
def tdoa_loss_original(params, sensor_x, sensor_y, observed_times, c):
    x_est, y_est, t_est = params
    pred_distances = np.sqrt((sensor_x - x_est)**2 + (sensor_y - y_est)**2)
    pred_times = t_est + pred_distances / c
    error = np.sum((observed_times - pred_times)**2)
    return error

initial_guess = [AREA_SIZE/2, AREA_SIZE/2, 0.0]
res_nm = minimize(
    tdoa_loss_original,
    initial_guess,
    args=(sensor_x, sensor_y, observed_arrival_times, SPEED_OF_SOUND),
    method='Nelder-Mead',
    options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8}
)

est_x_nm, est_y_nm, est_t_nm = res_nm.x
error_nm = np.sqrt((est_x_nm - true_source_x)**2 + (est_y_nm - true_source_y)**2)

# ------------------------------------------------------------------------------------------
# B. Improved time of arrival (TOA) least-squares solver
#    (still uses TOAs, no Generalized Cross-Correlation GCC, no Time Delay of Arrival TDOA)
# ------------------------------------------------------------------------------------------
def toa_residuals(params, sx, sy, obs_times, c):
    x, y, t0 = params
    d = np.sqrt((sx - x)**2 + (sy - y)**2)
    pred = t0 + d / c
    return obs_times - pred

# Better initial guess: centroid of earliest arrivals
k = max(4, int(NUM_SENSORS * 0.2))
earliest_idx = np.argsort(observed_arrival_times)[:k]
init_x_ls = np.mean(sensor_x[earliest_idx])
init_y_ls = np.mean(sensor_y[earliest_idx])
init_t_ls = np.min(observed_arrival_times) - np.min(distances[earliest_idx]) / SPEED_OF_SOUND
initial_params_ls = np.array([init_x_ls, init_y_ls, init_t_ls])

res_ls = least_squares(
    toa_residuals,
    initial_params_ls,
    args=(sensor_x, sensor_y, observed_arrival_times, SPEED_OF_SOUND),
    loss='huber',   # robust to a few bad sensors
    f_scale=0.05,   # roughly the noise std in seconds
    xtol=1e-10, ftol=1e-10, gtol=1e-10,
    max_nfev=5000
)

est_x_ls, est_y_ls, est_t_ls = res_ls.x
error_ls = np.sqrt((est_x_ls - true_source_x)**2 + (est_y_ls - true_source_y)**2)

# Covariance estimate for LS solution
J = res_ls.jac
resid = res_ls.fun
resid_var = np.sum(resid**2) / max(1, (len(resid) - len(res_ls.x)))
try:
    cov = np.linalg.inv(J.T @ J) * resid_var
except np.linalg.LinAlgError:
    cov = np.full((3,3), np.nan)

# 6. Visualization
plt.figure(figsize=(10, 8))

plt.scatter(sensor_x, sensor_y, c='blue', alpha=0.6, label='Phone Sensors')

plt.scatter(true_source_x, true_source_y, c='red', s=200, marker='*', label='True Tornado')

plt.scatter(est_x_nm, est_y_nm, c='orange', s=120, marker='X',
            label=f'Original (Nelder-Mead)\nError: {error_nm:.2f} m')

plt.scatter(est_x_ls, est_y_ls, c='green', s=120, marker='D',
            label=f'Improved LS (Huber)\nError: {error_ls:.2f} m')

plt.title(f"Tornado Localization: Original vs Improved\n"
          f"{NUM_SENSORS} Nodes | Noise: {NOISE_STD_DEV*1000:.0f} ms")
plt.xlabel("Distance East (m)")
plt.ylabel("Distance North (m)")
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)

plt.tight_layout()
plt.show()

# Output true location, original triangulated estimate, and improved least squares estimate
print(f"True Location:        ({true_source_x:.2f}, {true_source_y:.2f})")
print(f"Original estimate:    ({est_x_nm:.2f}, {est_y_nm:.2f}),  t={est_t_nm:.4f},  error={error_nm:.2f} m")
print(f"Improved LS estimate: ({est_x_ls:.2f}, {est_y_ls:.2f}),  t={est_t_ls:.4f},  error={error_ls:.2f} m")
print("\nApprox covariance of LS [x,y,t]:\n", cov)

#  When you have relatively clean TOAs with simple Gaussian jitter, turning those into synthetic waveforms, then running GCC‑PHAT, then reconstructing TOAs/TDOAs throws away information 
# and injecting new error sources: finite sampling at 8 kHz, limited waveform duration, noise, imperfect source pulse model.

# If we assume devices can send reasonably precise timestamps (e.g., via Network Time Protocal NTP / Global Positioning System GPS + some calibration), 
# the original code is closer to reality than the synthetic waveform pipleline.

# The right incremental upgrades are: better optimizer (least_squares with robust loss), outlier handling, uncertainty quantification, Monte Carlo evaluation.
