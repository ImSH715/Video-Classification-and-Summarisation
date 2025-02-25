import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            return

        # Faster directory traversal using os.scandir()
        class_folders = [entry.name for entry in os.scandir(root_dir) if entry.is_dir()]
        self.class_to_idx = {cls: i for i, cls in enumerate(class_folders)}

        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(root_dir, class_name)

            # Load and group frames by video prefix
            grouped_frames = {}
            for frame in glob.iglob(os.path.join(class_path, "*.jpg")):
                video_prefix = "_".join(frame.split("_")[:-1])  # Extract video identifier
                grouped_frames.setdefault(video_prefix, []).append(frame)

            # Store only valid sequences (at least 10 frames)
            self.data.extend(
                (sorted(frames)[:10], class_idx) for frames in grouped_frames.values() if len(frames) >= 10
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, label = self.data[idx]
        images = [self.transform(Image.open(f).convert("RGB")) for f in frames]

        return torch.stack(images), torch.tensor(label)
