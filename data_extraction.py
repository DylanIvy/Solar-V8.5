
import cv2
import numpy as np
import os
import torch



def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads images as BGR; convert to HSV:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frames.append(frame)
    cap.release()
    return np.array(frames)  # Shape: (num_frames, H, W, 3)


# List of video files
video_files = [r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-1-24.mp4",
               r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-2-24.mp4",
               r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-3-24.mp4",
               r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\1-4-24.mp4"
               #r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\daylight5.mp4",
               #r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\daylight6.mp4",
               #r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\daylight7.mp4",

               ]
num_videos = len(video_files)
print(f"Number of videos to merge: {num_videos}")
output_folder = r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images"
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Full path for the output file
output_file = os.path.join(output_folder, "output.mp4")
# Create VideoWriter object
output_file = "output.mp4"
fps = 30  # Frames per second

# Get properties of the first video
cap = cv2.VideoCapture(video_files[0])
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for video in video_files:
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()

out.release()
print("Videos merged successfully!")
video_path =r"C:\Users\jldag\OneDrive\Desktop\Senior-Design\sky_images\output.mp4"# Replace with your video file path
video_frames = extract_frames(video_path)
print(f"Extracted {len(video_frames)} frames from the video.")