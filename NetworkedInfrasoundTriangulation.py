import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Simulation Setup
np.random.seed(42)  # For reproducibility

# Constants
SPEED_OF_SOUND = 343.0  # m/s
NUM_SENSORS = 50
AREA_SIZE = 2000  # 2km x 2km neighborhood
NOISE_STD_DEV = 0.05  # 50ms timing jitter (realistic for non-synchronized smart devices)

# True Tornado Source (unknown to the solver)
true_source_x = 1500.0
true_source_y = 1200.0
true_onset_time = 0.0  # Relative start time

# 2. Generate Sensor Network
# Randomly place phones in the neighborhood (0 to 2000m)
sensor_x = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)
sensor_y = np.random.uniform(0, AREA_SIZE, NUM_SENSORS)

# 3. Simulate Measurements (The "Observation")
# Calculate distance from source to each sensor
distances = np.sqrt((sensor_x - true_source_x)**2 + (sensor_y - true_source_y)**2)

# Calculate arrival times (Time of Flight)
true_arrival_times = true_onset_time + distances / SPEED_OF_SOUND

# Add noise (measurement error)
observed_arrival_times = true_arrival_times + np.random.normal(0, NOISE_STD_DEV, NUM_SENSORS)

# 4. TDOA Solver (The "Algorithm")
# We want to find (x, y, t) that minimizes the error between observed and predicted arrival times.

def tdoa_loss(params, sensor_x, sensor_y, observed_times, c):
    x_est, y_est, t_est = params
    
    # Predict arrival times for this guess
    pred_distances = np.sqrt((sensor_x - x_est)**2 + (sensor_y - y_est)**2)
    pred_times = t_est + pred_distances / c
    
    # Calculate Sum of Squared Errors
    error = np.sum((observed_times - pred_times)**2)
    return error

# Initial Guess (start at the center of the map)
initial_guess = [AREA_SIZE/2, AREA_SIZE/2, 0]

# Optimize
result = minimize(tdoa_loss, initial_guess, args=(sensor_x, sensor_y, observed_arrival_times, SPEED_OF_SOUND), method='Nelder-Mead')

estimated_x, estimated_y, estimated_t = result.x

# 5. Calculate Accuracy
location_error = np.sqrt((estimated_x - true_source_x)**2 + (estimated_y - true_source_y)**2)

# 6. Visualization
plt.figure(figsize=(10, 8))

# Plot Sensors
plt.scatter(sensor_x, sensor_y, c='blue', alpha=0.6, label='Phone Sensors (Home Arrays)')

# Plot True Source
plt.scatter(true_source_x, true_source_y, c='red', s=200, marker='*', label='True Tornado Location')

# Plot Estimated Source
plt.scatter(estimated_x, estimated_y, c='green', s=150, marker='X', label=f'Triangulated Estimate\n(Error: {location_error:.2f}m)')

# Draw lines from a few sensors to the estimate to visualize triangulation
for i in range(5): # Show just a few lines
    plt.plot([sensor_x[i], estimated_x], [sensor_y[i], estimated_y], 'k--', alpha=0.2)

plt.title(f"Networked Infrasound Triangulation\n{NUM_SENSORS} Nodes | Noise: {NOISE_STD_DEV*1000:.0f}ms | Localization Error: {location_error:.2f}m")
plt.xlabel("Distance East (m)")
plt.ylabel("Distance North (m)")
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)

# Save the plot
plt.savefig('tornado_triangulation.png')

# Output true location of tornado, estimated triangulated location, and error between the true and estimated location
print(f"True Location: ({true_source_x}, {true_source_y})")
print(f"Estimated Location: ({estimated_x:.2f}, {estimated_y:.2f})")
print(f"Localization Error: {location_error:.2f} meters")
