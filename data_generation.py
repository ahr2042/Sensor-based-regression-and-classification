import numpy as np
import pandas as pd

np.random.seed(42)

NUM_SAMPLES = 10000

# Base signals
ambient_temp = np.random.normal(25, 2, NUM_SAMPLES)
voltage = np.random.normal(12.0, 0.3, NUM_SAMPLES)
current = np.random.normal(1.5, 0.2, NUM_SAMPLES)

# Power calculation with noise
power = (voltage * current) + np.random.normal(0, 0.2, NUM_SAMPLES)

# Temperature model
temperature = (ambient_temp + 0.5 * power) + np.random.normal(0, 0.5, NUM_SAMPLES)

# Inject faults
fault = np.zeros(NUM_SAMPLES)

# Fault conditions
overcurrent = current > 2.0
overtemp = temperature > 60
undervoltage = voltage < 10.5

fault[overcurrent | overtemp | undervoltage] = 1

# Create DataFrame
df = pd.DataFrame({
    "ambient_temp": ambient_temp,
    "voltage": voltage,
    "current": current,
    "temperature": temperature,
    "power": power,
    "fault": fault.astype(int)
})

# Save dataset
df.to_csv("sensor_data.csv", index=False)

print(df.head())
print(df["fault"].value_counts())
