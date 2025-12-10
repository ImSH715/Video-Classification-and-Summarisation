import os
import glob



DATA_DIR = "data/train"

if not os.path.exists(DATA_DIR):
    print(f"ERROR: Path '{DATA_DIR}' does not exist!")
else:
    print("Found 'data/train' directory.")

class_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

if len(class_folders) == 0:
    print("ERROR: No class folders found inside 'data/train'!")
else:
    print(f"Found class folders: {class_folders}")


for cls in class_folders:
    cls_path = os.path.join(DATA_DIR, cls)
    video_folders = [d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))]
    
    for video in video_folders:
        video_path = os.path.join(cls_path, video)
        frames = glob.glob(os.path.join(video_path, "*.jpg"))
        
        if len(frames) == 0:
            print(f"ERROR: No frames found in '{video}' inside '{cls}'")
        else:
            print(f"Found {len(frames)} frames in '{video}' inside '{cls}'")
            break  # Stop after the first found
