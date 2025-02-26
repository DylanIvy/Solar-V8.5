# train.py 2

from csv_reader import load_and_combine_csvs
from data_extraction import extract_frames
from Custom_dataset import SolarRadiationImageDataset
from model import SolarRadiationPredictor
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timedelta
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import num_videos, video_files
import matplotlib.pyplot as plt
import os

# ---------- 1. Data Loading and Preparation ----------

# Define file paths (adjust these as necessary)
csv_directory = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\solar_radiation data"
test_directory=r"C:\Users\jldag\OneDrive\Desktop\test_directory"
video_path =r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\output.mp4"
test_path=r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-1-25.mp4"

# Load combined CSV data (ensure it contains "Timestamp" and "Solar Radiation Sensor")
combined_df = load_and_combine_csvs(csv_directory)
test_df=load_and_combine_csvs(test_directory)
print("Combined DataFrame loaded with shape:", combined_df.shape)

# Extract video frames from the timelapse video
video_frames = extract_frames(video_path)
test_frames=extract_frames(test_path)
print(f"Extracted {len(video_frames)} video frames.")
print(f"Extracted {len(test_frames)} test video frames.")
# Get the video frame rate using OpenCV
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print("Detected frame rate:", frame_rate)

# Set the video start time.
# (For a timelapse video, this can be an arbitrary reference; here we use midnight.)
video_start_time = datetime(2025, 1, 1, 0, 0, 0)

# Set the forecast offset.
# If your CSV records data about every minute and you want to predict 5 minutes into the future:
forecast_offset = 5

# Create the full dataset instance with the scaling alignment.
full_dataset = SolarRadiationImageDataset(
    dataframe=combined_df,
    video_frames=video_frames,
    video_start_time=video_start_time,
    frame_rate=frame_rate,
    forecast_offset=forecast_offset,
    image_transform=None  # Replace with your torchvision transforms if needed.
)
test_dataset=SolarRadiationImageDataset(
    dataframe=test_df,
    video_frames=test_frames,
    video_start_time=video_start_time,
    frame_rate=frame_rate,
    forecast_offset=forecast_offset,
    image_transform=None

)

# ---------- 2. Verify Data Alignment Across the Dataset ----------

def verify_data_alignment_scaled(dataset, num_samples=10):
    """
    Prints alignment details for a set of samples evenly distributed across the dataset.
    Handles multiple videos with skipped nighttime periods dynamically.
    """
    df = dataset.dataframe
    day_start = dataset.day_start

    # Define the duration of the active video period in seconds (12 hours = 43200 seconds)
    active_period_seconds = 43200
    skipped_period_seconds = 43200  # Nighttime skipped

    # Total actual seconds, considering the skipped nighttime periods
    total_actual_seconds = num_videos * active_period_seconds

    print("Day Start:", day_start)
    print("Day End  :", dataset.day_end)
    print("Total actual seconds:", total_actual_seconds)
    print("Total video frames:", dataset.total_video_frames)
    print("=" * 50)

    max_index = len(df) - dataset.forecast_offset
    indices = np.linspace(0, max_index - 1, num_samples, dtype=int)

    for idx in indices:
        row = df.iloc[idx]
        csv_time = row["Timestamp"]

        # Calculate elapsed seconds, accounting for skipped nighttime
        elapsed_seconds = (csv_time - day_start).total_seconds()
        video_index = int(elapsed_seconds // (active_period_seconds + skipped_period_seconds))  # Determine which video
        within_video_seconds = elapsed_seconds % (active_period_seconds + skipped_period_seconds)

        # If within nighttime period, adjust to the next active period
        if within_video_seconds >= active_period_seconds:
            video_index += 1
            within_video_seconds -= active_period_seconds + skipped_period_seconds

        # Compute the adjusted elapsed time within the active periods
        adjusted_elapsed_seconds = video_index * active_period_seconds + max(0, within_video_seconds)

        # Compute frame index
        computed_frame_index = round((adjusted_elapsed_seconds / total_actual_seconds) * dataset.total_video_frames)
        computed_frame_index = max(0, min(computed_frame_index, dataset.total_video_frames - 1))
        video_frame_time = dataset.video_start_time + timedelta(seconds=(computed_frame_index / dataset.frame_rate))

        # Sensor values
        current_sensor = row["Solar Radiation Sensor"]
        future_sensor = df.iloc[idx + dataset.forecast_offset]["Solar Radiation Sensor"]

        print(f"Sample index: {idx}")
        print(f"  CSV Timestamp         : {csv_time}")
        print(f"  Elapsed seconds       : {adjusted_elapsed_seconds:.2f} sec")
        print(f"  Computed frame index  : {computed_frame_index}")
        print(f"  Computed video time   : {video_frame_time}")
        print(f"  Current sensor value  : {current_sensor}")
        print(f"  Future sensor value   : {future_sensor}")
        print("-" * 50)


# Verify alignment across 10 samples spaced throughout the CSV data.
verify_data_alignment_scaled(full_dataset, num_samples=10)


sample_idx =1695
sample = full_dataset[sample_idx]
image_tensor = sample["image"]  # Should have shape [C, H, W]
# Convert to numpy array with shape [H, W, C] and values in [0, 1]
image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(image_np)
plt.title(f"Sample index {sample_idx}\nCSV Timestamp: {sample['timestamp']}\nComputed Video Time: {sample['computed_video_time']} sec")
plt.axis("off")
plt.show()

# ---------- 4. Split Dataset into Training and Validation Sets ----------



train_dataset=full_dataset
val_dataset=test_dataset
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ---------- 5. Model, Loss, Optimizer ----------

model = SolarRadiationPredictor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

model_save_dir = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\models"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "trained_model.pth")
checkpoint_save_path=os.path.join(model_save_dir,"model_checkpoints.pth")

# ---------- 6. Training Loop with Validation ----------

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for batch in train_dataloader:
        images = batch["image"].to(device)  # (batch_size, 3, H, W)
        sensor_data = batch["current_sensor"].float().to(device)  # shape: (batch_size,)
        targets = batch["target"].float().unsqueeze(1).to(device)  # shape: (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images, sensor_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
    train_loss = running_train_loss / len(train_dataset)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to(device)
            sensor_data = batch["current_sensor"].float().to(device)
            targets = batch["target"].float().unsqueeze(1).to(device)

            outputs = model(images, sensor_data)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * images.size(0)
    val_loss = running_val_loss / len(val_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }
        torch.save(checkpoint, checkpoint_save_path)
        print(f"Checkpoint saved to {checkpoint_save_path}")

        # Additionally, save the model's state_dict
        torch.save(model.state_dict(), model_save_path)
        print(f"Model state_dict saved to {model_save_path}")

print("Training completed.")