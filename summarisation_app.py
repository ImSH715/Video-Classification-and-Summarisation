import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QComboBox, QDialog, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal

from models import CNNLSTM
from dataset import ImageSequenceDataset, mean, std

# dataset directory
dataset_root = "data/UCF-101-frames" 
# train and test set lists
annotation_file = "data/UCFTrainTestList"
sequence_length = 20
input_size = (128, 128)
# number of classes, currently 57 classes from 101
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

class VideosummarisationApp(QWidget):
    """
    Main screen for summarisation application with model loading and GUI controls
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video summarisation App")
        self.setGeometry(100, 100, 800, 600)
        self.model = None
        self.checkpoint_path = ''
        self.video_path = None
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.frame_rate = 30
        self.temp_dir = "temp_frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.detected_actions = []
        self.all_frame_paths = []
        self.frame_predictions = []
        self.init_ui()
        self.class_names = self.load_class_names()
        self.num_classes = len(self.class_names)

    def init_ui(self):
        """
        All GUI setup and outlines
        """
        self.image_label = QLabel("Video Frame:")
        self.prediction_label = QLabel("Detected Actions:")
        self.model_select_button = QPushButton("Select Trained Model")
        self.model_select_button.clicked.connect(self.select_model_checkpoint)
        self.video_upload_button = QPushButton("Upload Video")
        self.video_upload_button.clicked.connect(self.upload_video)
        self.action_selection_combobox = QComboBox()
        self.summarise_button = QPushButton("Summarise Selected Action")
        self.summarise_button.setEnabled(False)
        self.summarise_button.clicked.connect(self.summarise_video)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.model_select_button)
        layout.addWidget(self.video_upload_button)
        layout.addWidget(self.action_selection_combobox)
        layout.addWidget(self.summarise_button)
        self.setLayout(layout)

    def select_model_checkpoint(self):
        """
        Opens dialog for the user to select trained model file(.pth)
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", ".", "PyTorch Checkpoint Files (*.pth *.pt)")
        if file_path:
            self.checkpoint_path = file_path
            self.load_model()

    def load_model(self):
        """
        Loads trained model file(.pth)
        """
        if not self.checkpoint_path:
            return
        try:
            self.model = CNNLSTM(num_classes=self.num_classes)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
            sys.exit()

    def load_class_names(self):
        """
        Load action class names from optimised dataset
        """
        temp_dataset = ImageSequenceDataset(
            dataset_root,
            annotation_file,
            self.sequence_length,
            transform=get_transform(),
            train=True
        )
        return temp_dataset.label_names

    def upload_video(self):
        """
        Opens dialog for the users to select an input to generate frame process
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", ".", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.video_path = file_path
            self.frame_predictions = []
            self.action_selection_combobox.clear()
            self.summarise_button.setEnabled(False)
            self.detected_actions = []
            self.all_frame_paths = []

            self.progress_dialog = QDialog(self)
            self.progress_dialog.setWindowTitle("Processing Video")
            self.progress_dialog.setMinimumWidth(400)
            progress_layout = QVBoxLayout()
            self.progress_label = QLabel("Extracting frames and detecting actions")
            self.progress_bar = QProgressBar()
            progress_layout.addWidget(self.progress_label)
            progress_layout.addWidget(self.progress_bar)
            self.progress_dialog.setLayout(progress_layout)
            self.progress_dialog.show()

            self.process_frames_thread = ProcessFramesThread(
                file_path, self.model, self.class_names, self.input_size, self.sequence_length
            )
            self.process_frames_thread.prediction_signal.connect(self.display_frame_prediction)
            self.process_frames_thread.finished_signal.connect(self.on_finished_processing)
            self.process_frames_thread.progress_signal.connect(self.update_progress)
            self.process_frames_thread.start()

    def update_progress(self, progress):
        """
        Update the progress indicator during frame process
        """
        self.progress_bar.setValue(progress)

    def on_finished_processing(self, detected_actions, all_frame_paths):
        """
        Automatic function processing:
        Close the progress bar,
        Store all frame paths,
        List all detected actions into the dropdown list
        Make the button available
        """
        self.progress_dialog.close()
        self.all_frame_paths = all_frame_paths
        if detected_actions:
            self.detected_actions = detected_actions
            self.action_selection_combobox.addItems(self.detected_actions)
            self.prediction_label.setText(f"Detected Actions: {', '.join(self.detected_actions)}")
            self.summarise_button.setEnabled(True)

    def display_frame_prediction(self, frame_index, image, predicted_class_name, confidence):
        """
        Display preview of predicited class
        """
        width, height = 200, 150
        img = image.resize((width, height))
        q_image = QImage(img.tobytes("raw", "RGB"), width, height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        QApplication.processEvents()
        self.frame_predictions.append(predicted_class_name)

    def summarise_video(self):
        """
        Call all the frames selected by user's decision
        """
        selected_action = self.action_selection_combobox.currentText()
        if not selected_action:
            return
        action_frames = [
            self.all_frame_paths[i]
            for i, prediction in enumerate(self.frame_predictions)
            if prediction == selected_action
        ]
        if not action_frames:
            return
        output_video_path = f"data/test/final/{selected_action}_summary.mp4"
        self.generate_summary_video(action_frames, output_video_path, selected_action)
        self.play_video(output_video_path)

    def generate_summary_video(self, frame_paths, output_path, action_name):
        """
        Collect all the frames based on the user selected action.
        Create and save output video with collected frames.
        """
        if not frame_paths or len(frame_paths) < 2:
            return
        try:
            trimmed_frame_paths = frame_paths[10:-10] if len(frame_paths) > 20 else frame_paths
            if not trimmed_frame_paths:
                return
            first_frame = cv2.imread(trimmed_frame_paths[0])
            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.frame_rate, (width, height))
            font_path = "arial.ttf"
            try:
                font = ImageFont.truetype(font_path, 24)
            except:
                font = ImageFont.load_default()
            for frame_path in trimmed_frame_paths:
                frame = cv2.imread(frame_path)
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((10, 10), action_name, font=font, fill=(255, 255, 255))
                frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_cv2)
            out.release()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Error generating summary video: {e}")

    def play_video(self, video_path):
        """
        Play output video using the default video player
        """
        if sys.platform.startswith('darwin'):
            os.system(f'open "{video_path}"')
        elif os.name == 'nt':
            os.system(f'start "" "{video_path}"')

class ProcessFramesThread(QThread):
    """
    Thread for extracting frames and recognising actions
    """
    prediction_signal = pyqtSignal(int, Image.Image, str, float)
    finished_signal = pyqtSignal(list, list)
    progress_signal = pyqtSignal(int)

    def __init__(self, video_path, model, class_names, input_size, sequence_length):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.class_names = class_names
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.temp_dir = "temp_frames"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.action_counts = {}
        self.all_frame_paths = []

    def run(self):
        """
        Process each frame in the video 
        Generate predictions
        """
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        frame_paths = self.process_video_to_frames(self.video_path)
        self.all_frame_paths = frame_paths
        if not frame_paths:
            self.finished_signal.emit([], [])
            return
        self.progress_signal.emit(0)
        for i, frame_path in enumerate(frame_paths):
            img = Image.open(frame_path).convert('RGB')
            transformed_img = transform(img)
            frame_sequence = [
                transform(Image.open(frame_paths[j]).convert('RGB'))
                for j in range(i, min(i + self.sequence_length, len(frame_paths)))
            ]
            padding_needed = self.sequence_length - len(frame_sequence)
            frame_sequence.extend([transformed_img] * padding_needed)
            input_tensor = torch.stack(frame_sequence).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_index = torch.argmax(probabilities, dim=1).item()
                predicted_class_name = self.class_names[predicted_class_index]
                confidence = probabilities[0, predicted_class_index].item() * 100
            self.prediction_signal.emit(i, img, predicted_class_name, confidence)
            self.action_counts[predicted_class_name] = self.action_counts.get(predicted_class_name, 0) + 1
            self.progress_signal.emit(int((i + 1) / len(frame_paths) * 100))
        sorted_actions = sorted(self.action_counts.items(), key=lambda item: item[1], reverse=True)
        detected_actions = [action[0] for action in sorted_actions]
        self.finished_signal.emit(detected_actions, self.all_frame_paths)

    def process_video_to_frames(self, video_path):
        """
        Extract input video into frames
        """
        cap = cv2.VideoCapture(video_path)
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        frame_paths = []
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(self.temp_dir, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_number += 1
        cap.release()
        return frame_paths

if __name__ == '__main__':
    """
    Main function to generate user interface for summarisation application
    """
    app = QApplication(sys.argv)
    window = VideosummarisationApp()
    window.show()
    sys.exit(app.exec())
