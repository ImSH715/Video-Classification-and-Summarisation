import sys
import os
import random
import glob
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from models import CNNLSTM
from dataset import ImageSequenceDataset, mean, std

# dataset directory
dataset_root = "data/UCF-101-frames"
# annotation file directory
annotation_file = "data/UCFTrainTestList"
sequence_length = 20
input_size = (128, 128)
# number of classes for the optimised dataset
num_classes = 57

def get_transform():
    """
    Transform the input for resizing, tensor conversion, and normalisation by using torchvision
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

class VideoClassificationApp(QWidget):
    """
    Video classification Application using a pre-trained CNN-LSTM model
    However, there will be upload button for the users to select a model resulted high accuracy
    """
    def __init__(self):
        """
        App window, and GUI elements initialisation
        """
        super().__init__()
        self.setWindowTitle("Video Classification App")
        self.setGeometry(100, 100, 800, 600)
        self.model = None
        self.class_names = []
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_classes = num_classes
        self.checkpoint_path = ""
        self.init_ui()
        self.load_class_names()

    def init_ui(self):
        """
        UI setup to display classification application for the user
        """
        self.image_label = QLabel("Video Frame:")
        self.prediction_label = QLabel("Prediction:")
        self.model_select_button = QPushButton("Select Model")
        self.model_select_button.clicked.connect(self.select_model_checkpoint)
        self.upload_button = QPushButton("Classify Random Video")
        self.upload_button.clicked.connect(self.load_random_video)
        self.video_upload_button = QPushButton("Upload Video")
        self.video_upload_button.clicked.connect(self.upload_video)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.model_select_button)
        layout.addWidget(self.video_upload_button)
        layout.addWidget(self.upload_button)
        self.setLayout(layout)

    def select_model_checkpoint(self):
        """
        Wait for the user to upload trained model (checkpoint) file
        Load and update the model 
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", ".", "PyTorch Checkpoint Files (*.pth *.pt)")
        if file_path:
            self.checkpoint_path = file_path
            self.load_model()
            QMessageBox.information(self, "Model Selected", f"Model checkpoint selected: {self.checkpoint_path}")

    def load_model(self):
        """
        Load selected trained model file (.pth)
        Loads model weights from selected checkpoint path
        """
        if not self.checkpoint_path:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
        try:
            self.model = CNNLSTM(num_classes=self.num_classes)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            sys.exit()

    def load_class_names(self):
        """ Loads the list of class names from the dataset utility. """
        temp_dataset = ImageSequenceDataset(
            dataset_root,
            annotation_file,
            self.sequence_length,
            transform=get_transform(),
            train=True
        )
        self.class_names = temp_dataset.label_names
        self.num_classes = len(self.class_names)

    def load_random_video(self):
        """
        Allow random video from the dataset
        Predict the class and display the result in the screen
        """
        all_actions = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        if not all_actions:
            QMessageBox.warning(self, "Warning", "UCF-101 frame directory not found")
            return
        random_action = random.choice(all_actions)
        action_path = os.path.join(dataset_root, random_action)
        all_videos = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        if not all_videos:
            QMessageBox.warning(self, "Warning", f"No videos found in {random_action} action")
            return
        random_video = random.choice(all_videos)
        video_path = os.path.join(action_path, random_video)
        image_paths = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=lambda x: int(Path(x).stem))
        if not image_paths or len(image_paths) < self.sequence_length:
            QMessageBox.warning(self, "Warning", "Selected video does not have enough frames")
            return
        selected_paths = image_paths[:self.sequence_length]
        images = [get_transform()(Image.open(p).convert('RGB')) for p in selected_paths]
        input_tensor = torch.stack(images).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            predicted_class_name = self.class_names[predicted_class_index]
            accuracy = probabilities[0, predicted_class_index].item() * 100
        self.prediction_label.setText(f"Prediction: {predicted_class_name} (accuracy: {accuracy:.2f}%)")
        self.display_frame(Image.open(selected_paths[0]).convert('RGB'))

    def upload_video(self):
        """ 
        Display dialogs for the user to upload video
        Accordingly the screen display predicted action class
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", ".", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            frame_paths = self.process_video_to_frames(file_path)
            if not frame_paths or len(frame_paths) < self.sequence_length:
                QMessageBox.warning(self, "Warning", "Selected video does not have enough frames.")
                return
            selected_paths = frame_paths[:self.sequence_length]
            images = [get_transform()(Image.open(p).convert('RGB')) for p in selected_paths]
            input_tensor = torch.stack(images).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_index = torch.argmax(probabilities, dim=1).item()
                predicted_class_name = self.class_names[predicted_class_index]
                accuracy = probabilities[0, predicted_class_index].item() * 100
            self.prediction_label.setText(f"Prediction: {predicted_class_name} (accuracy: {accuracy:.2f}%)")
            self.display_frame(Image.open(selected_paths[0]).convert('RGB'))

    def process_video_to_frames(self, video_path):
        """
        Extract frames from a dataset and return the classified path
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_paths = []
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_dir, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_number += 1
        cap.release()
        return frame_paths

    def display_frame(self, img):
        """ 
        Shows the first fram of the input as a preview 
        """
        width, height = 400, 300
        img = img.resize((width, height))
        q_image = QImage(img.tobytes("raw", "RGB"), width, height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

if __name__ == '__main__':
    """
    Main function to generate user interface for classification application
    """
    app = QApplication(sys.argv)
    window = VideoClassificationApp()
    window.show()
    sys.exit(app.exec())
