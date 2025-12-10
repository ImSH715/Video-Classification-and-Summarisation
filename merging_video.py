import os
import cv2
import random

# Base directory containing action folders
base_path = 'data/UCF101'

# Output video settings
output_video_path = 'data/merged_n.avi'
output_fps = 30 

#Manual
"""selected_actions = [
    'SoccerPenalty', 
    'SoccerPenalty',
    'Biking',
    'Diving',
    'Archery'
]
"""
# Validate the classes exist
available_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]


#Random selection
if len(available_folders) < 5:
    raise ValueError("Not enough classes available to select 5.")
selected_actions = random.sample(available_folders, 5)

if not selected_actions:
    raise ValueError("None of the specified classes exist in the dataset.")

print("Manually selected action folders:", selected_actions)


selected_actions = [action for action in selected_actions if action in available_folders]
video_clips = []

# For each selected action folder, pick one random video
for action in selected_actions:
    action_path = os.path.join(base_path, action)
    video_files = [f for f in os.listdir(action_path) if f.endswith(('.avi', '.mp4', '.mov'))]
    selected_video = random.choice(video_files)
    video_path = os.path.join(action_path, selected_video)
    print(f"Selected video from {action}: {video_path}")
    video_clips.append(video_path)

# Set frame size based on the first video
first_video = cv2.VideoCapture(video_clips[0])
width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
first_video.release()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

# Concatenate each selected video
for clip in video_clips:
    cap = cv2.VideoCapture(clip)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        out.write(resized)
    cap.release()
out.release()
