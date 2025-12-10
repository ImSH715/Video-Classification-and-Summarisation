import argparse
import io
import logging
import os
from pathlib import Path

import skvideo.io
import tqdm
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from data.extract_frames import extract_frames
from dataset import *
from models import *


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("/r/n/t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def main() -> None:
    # A logger for this file
    logger = logging.getLogger(__name__)

    # Define the specific video path
    video_path = r"C:/Users/naya0/Uni/Dissertation/Video-Classification-and-Summarisation/data/UCF101/Archery/v_Archery_g01_c01.avi"
    save_gif_name = Path(video_path).stem + "_predicted"
    checkpoint_path = r"C:/Users/naya0/Uni/Dissertation/Video-Classification-and-Summarisation/checkpoints/checkpoint_007.pth"

    # Define the root dataset path and frames directory name
    dataset_root = r"C:/Users/naya0/Uni/Dissertation/Video-Classification-and-Summarisation/data"
    frames_dirname = "UCF-101-frames"
    dataset_path = (Path(dataset_root) / frames_dirname).as_posix()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (3, 224, 224)  # Example input shape, adjust if needed

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Infer labels from the directory structure
    labels = sorted(list(set(os.listdir(dataset_path))))
    num_classes = len(labels)
    latent_dim = 512  # Example latent dimension, adjust if needed
    lstm_layers = 2  # Example LSTM layers, adjust if needed
    hidden_dim = 512  # Example hidden dimension, adjust if needed
    bidirectional = True  # Example bidirectional, adjust if needed
    attention = True  # Example attention, adjust if needed

    # Define model and load model checkpoint
    model = CNNLSTM(
        num_classes=num_classes,
        latent_dim=latent_dim,
        lstm_layers=lstm_layers,
        hidden_dim=hidden_dim,
        bidirectional=bidirectional,
        attention=attention,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Extract predictions
    output_frames = []
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for frame in tqdm.tqdm(
        extract_frames(video_path), file=tqdm_out, desc="Processing frames"
    ):
        image_tensor = Variable(transform(frame)).to(device)
        image_tensor = image_tensor.view(1, 1, *image_tensor.shape)

        # Get label prediction for frame
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_label = labels[prediction.argmax(1).item()]

        # Draw label on frame
        frame_pil = Image.fromarray(frame)
        d = ImageDraw.Draw(frame_pil)
        d.text(xy=(10, 10), text=predicted_label, fill=(255, 255, 255))
        output_frames += [np.array(frame_pil)]

    # Create video from frames
    writer = skvideo.io.FFmpegWriter(f"{save_gif_name}.gif")
    for frame in tqdm.tqdm(output_frames, file=tqdm_out, desc="Writing to video"):
        writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    main()