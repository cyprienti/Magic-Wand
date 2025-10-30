import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSVs
papers_df = pd.read_csv("./recording_samples/paper.csv", delimiter=";")
scissors_df = pd.read_csv("./recording_samples/scissors.csv", delimiter=";")
rock_df = pd.read_csv("./recording_samples/rock.csv", delimiter=";")

# Ensure output directory exists
output_dir = "./graphs/"
os.makedirs(output_dir, exist_ok=True)

# Plotting function that saves plots
def plot(df, name):
    # Accelerometer plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["accX"], label="accX")
    plt.plot(df["time"], df["accY"], label="accY")
    plt.plot(df["time"], df["accZ"], label="accZ")
    plt.title(f"{name} - Accelerometer X, Y, Z vs Time")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_accelerometer.png")
    plt.close()

    # Gyroscope plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["gyroX"], label="gyroX")
    plt.plot(df["time"], df["gyroY"], label="gyroY")
    plt.plot(df["time"], df["gyroZ"], label="gyroZ")
    plt.title(f"{name} - Gyroscope X, Y, Z vs Time")
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_gyroscope.png")
    plt.close()

# Generate and save plots
plot(papers_df, "papers")
plot(scissors_df, "scissors")
plot(rock_df, "rock")