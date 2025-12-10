import glob
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

"""
To transform the extracted frames, ImageNet is used to state mean and std for normalization
"""
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class ImageSequenceDataset(Dataset):
    """
    Dataset loading sequence of image frames for video classification
    """
    def __init__(self, root_dir, annotation_file, sequence_length, transform=None, train=True):
        """
        Initialise dataset
        Read the label mapping (classInd.txt in the UCFTrainTestList folder)
        Load train and test video frame sequences from framed files
        """
        self.root_dir = Path(root_dir)
        self.annotation_file = Path(annotation_file)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train = train
        self.label_mapping = self._extract_label_mapping(self.annotation_file)
        self.sequences = self._load_sequences()
        self.label_names = sorted(self.label_mapping.keys())
        self.num_classes = len(self.label_names)

    def _extract_label_mapping(self, split_path):
        """
        Read classInd.txt to map action labels
        """
        label_path = split_path / "classInd.txt"
        label_mapping = {}
        with open(label_path) as f:
            for line in f:
                label, action = line.split()
                label_mapping[action] = int(label) - 1
        return label_mapping

    def _load_sequences(self):
        """
        Load all video frame sequences according to the listed split file based on the trainlist and testlist text file
        Include the videos including sufficient amount of frames
        """
        sequences = []
        split_file = "trainlist01.txt" if self.train else "testlist01.txt"
        split_file_path = self.annotation_file / split_file
        with open(split_file_path) as f:
            for line in f:
                video_name = Path(line.split(" ")[0])
                action_name = video_name.parts[0]
                image_dir = self.root_dir / action_name / video_name.name
                if image_dir.is_dir():
                    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"), key=lambda x: int(Path(x).stem))
                    if len(image_paths) >= self.sequence_length:
                        sequences.append((image_paths, self.label_mapping[action_name]))
        return sequences

    def __len__(self):
        """
        Return number of sequences in the dataset
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a sequence of images and its label for training or testing
        Random contiguous sequence is selected for training set
        Use the first frame for testing set (Padding the dataset if required)
        (Tensor[sequence_length, 3, H, W], target_label) as a return value
        """
        image_paths, target = self.sequences[idx]
        num_frames = len(image_paths)
        if self.train:
            start_index = random.randint(0, num_frames - self.sequence_length)
            selected_paths = image_paths[start_index : start_index + self.sequence_length]
        else:
            selected_paths = image_paths[:self.sequence_length]
            if len(selected_paths) < self.sequence_length:
                selected_paths += [selected_paths[-1]] * (self.sequence_length - len(selected_paths))

        images = [self.transform(Image.open(p).convert('RGB')) for p in selected_paths]
        return torch.stack(images), target
