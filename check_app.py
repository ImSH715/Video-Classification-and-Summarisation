import sys
import os
import cv2
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QListWidget
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from torchvision import transforms

try:
    from models import CNNLSTM
    from config import Config  # Import Config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure 'models.py' and 'config.py' are in the Python path.")
    sys.exit(1)

device = torch.device("cpu")
print(f"Using device: {device}")

def extract_frames(video_path, max_frames=Config.test_num_frames): # Use num_frames from config
    """Extracts and preprocesses frames from a video file onto CPU."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Config.test_image_height, Config.test_image_width)), # Use test image size from config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Use standard normalization
    ])
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_count == 0:
                print(f"Error: Could not read any frame from {video_path}")
            else:
                print(f"Warning: Reached end of video, extracted {len(frames)} frames.")
            cap.release()
            break
        frame_count += 1
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = transform(frame_rgb)
            frames.append(processed_frame)
            if len(frames) >= max_frames:
                break
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    cap.release()
    if not frames:
        print("Error: No frames were successfully processed.")
        return None
    frames_tensor = torch.stack(frames).unsqueeze(0)
    return frames_tensor

class VideoClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.num_classes = None
        self.class_labels = getattr(Config, 'class_labels', None) # Get class labels from config if available
        self.video_path = None
        self.checkpoint_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Multi-Action Detection App (CPU)")
        self.setGeometry(100, 100, 650, 500)  # Adjusted size

        # Checkpoint Selection
        self.checkpoint_label = QLabel("Model Checkpoint (.pth): Not Selected", self)
        self.checkpoint_button = QPushButton("Select Checkpoint File", self)
        self.checkpoint_button.clicked.connect(self.select_checkpoint_file)

        # Video Selection
        self.video_label = QLabel("Input Video File: Not Selected", self)
        self.video_label.setStyleSheet("font-size: 14px;")
        self.video_label.setFixedHeight(30)

        self.image_label = QLabel("Video Thumbnail", self)
        self.image_label.setFixedSize(320, 240)
        self.image_label.setStyleSheet("border: 1px solid gray; text-align: center;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.upload_video)

        # Prediction
        self.predict_button = QPushButton("Detect All Actions", self) # Changed button text
        self.predict_button.clicked.connect(self.predict_all_actions) # Changed function name
        self.predict_button.setEnabled(False)

        self.result_label = QLabel("Detected Actions:", self)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.actions_list = QListWidget(self)
        self.actions_list.setStyleSheet("font-size: 14px;")

        # Layout
        layout = QVBoxLayout()

        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(self.checkpoint_label)
        checkpoint_layout.addWidget(self.checkpoint_button)
        layout.addLayout(checkpoint_layout)

        layout.addSpacing(15)

        layout.addWidget(self.video_label)
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.upload_button)

        layout.addSpacing(15)

        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.actions_list)

        self.setLayout(layout)

    def select_checkpoint_file(self):
        """Opens a dialog to select the model checkpoint file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "PyTorch Models (*.pth)")
        if file_path:
            if file_path.lower().endswith('.pth'):
                self.checkpoint_path = file_path
                self.checkpoint_label.setText(f"Checkpoint: ...{os.path.basename(file_path)}")
                self.load_model()
                self._update_predict_button_state()
            else:
                QMessageBox.warning(self, "Invalid File", "Please select a valid .pth file.")
                self.checkpoint_path = None
                self.checkpoint_label.setText("Model Checkpoint (.pth): Not Selected")

    def load_model(self):
        """Loads the model architecture and weights onto CPU."""
        if not self.checkpoint_path:
            QMessageBox.warning(self, "Warning", "Select a checkpoint file first.")
            return False

        try:
            print(f"Loading model from: {self.checkpoint_path} onto CPU...")
            # Instantiate the model architecture and get num_classes from config
            self.num_classes = Config.num_classes
            self.model = CNNLSTM(num_classes=self.num_classes,
                                 latent_dim=Config.test_latent_dim,
                                 lstm_layers=Config.test_lstm_layers,
                                 hidden_dim=Config.test_hidden_dim,
                                 bidirectional=Config.test_bidirectional,
                                 attention=Config.test_attention)

            ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
                print("Model state loaded from 'ckpt['model']'.")
            else:
                self.model.load_state_dict(ckpt)
                print("Model state loaded directly from checkpoint file.")

            self.model.eval()
            print(f"Model loaded successfully onto CPU with {self.num_classes} output classes.")
            self.result_label.setText("Detected Actions:")
            self._update_predict_button_state()
            return True
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Checkpoint file not found: {self.checkpoint_path}")
            self.checkpoint_path = None
            self.checkpoint_label.setText("Model Checkpoint (.pth): Not Selected")
            self.model = None
            return False
        except RuntimeError as e:
            QMessageBox.critical(self, "Error Loading Model State",
                                 f"Error loading checkpoint weights: {e}\n"
                                 f"Ensure the checkpoint matches the model architecture.")
            print(f"Detailed error loading state dict: {e}")
            self.model = None
            return False
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"An unexpected error occurred: {e}")
            print(f"Detailed error loading model: {e}")
            self.model = None
            return False

    def upload_video(self):
        """Opens a dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.avi *.mp4 *.mov *.mkv)")
        if file_path:
            self.video_path = file_path
            self.video_label.setText(f"Video: ...{os.path.basename(file_path)}")
            self.display_video_thumbnail()
            self._update_predict_button_state()

    def display_video_thumbnail(self):
        """Displays the first frame of the selected video."""
        if not self.video_path:
            self.image_label.setText("Video Thumbnail")
            self.image_label.setPixmap(QPixmap())
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.image_label.setText("Error: Cannot open video")
            return
        ret, frame = cap.read()
        cap.release()

        if ret:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(),
                                                        Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))
            except Exception as e:
                print(f"Error displaying thumbnail: {e}")
                self.image_label.setText("Error displaying thumbnail")
                self.image_label.setPixmap(QPixmap())
        else:
            self.image_label.setText("Error reading frame")
            self.image_label.setPixmap(QPixmap())

    def _update_predict_button_state(self):
        """Enables the predict button only if checkpoint and video are selected/loaded."""
        ready = bool(self.checkpoint_path and
                     self.video_path and
                     self.model and
                     self.num_classes is not None)
        self.predict_button.setEnabled(ready)

        if ready:
            if not self.result_label.text().startswith("Detected"):
                self.result_label.setText("Detected Actions:")
        else:
            if not self.checkpoint_path:
                self.result_label.setText("⚠️ Select Checkpoint File (.pth).")
            elif not self.model:
                self.result_label.setText("⚠️ Loading model... or Error.")
            elif not self.video_path:
                self.result_label.setText("⚠️ Upload a Video File.")
            elif self.num_classes is None:
                self.result_label.setText("⚠️ Could not determine the number of classes.")

    def predict_all_actions(self):
        """Performs action prediction to detect all possible actions in the video."""
        if not self.video_path or not self.model or self.num_classes is None:
            QMessageBox.warning(self, "Missing Components",
                                 "Please ensure Checkpoint and Video are selected and the model is loaded.")
            return

        self.actions_list.clear()
        self.result_label.setText("⏳ Processing video and detecting all actions (CPU)...")
        QApplication.processEvents()

        frames_tensor = extract_frames(self.video_path)

        if frames_tensor is None:
            self.result_label.setText("⚠️ Error extracting frames or not enough frames.")
            return

        try:
            with torch.no_grad():
                output = self.model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)

                # Get all predicted probabilities and their indices
                all_probs, all_indices = torch.sort(probabilities, dim=1, descending=True)
                all_probs = all_probs.squeeze().tolist()
                all_indices = all_indices.squeeze().tolist()

                detected_actions = []
                if self.class_labels:
                    for i, index in enumerate(all_indices):
                        if 0 <= index < len(self.class_labels):
                            action_name = self.class_labels[index]
                            confidence = all_probs[i]
                            detected_actions.append(f"{action_name} (Confidence: {confidence:.2%})")
                        else:
                            detected_actions.append(f"Class Index: {index} (Confidence: {all_probs[i]:.2%})")
                    self.result_label.setText("Detected Actions (All):")
                    self.actions_list.addItems(detected_actions)
                    print(f"Action detection successful: All Possible Actions = {detected_actions}")
                else:
                    # If no class labels, show class indices and confidence
                    detected_actions = [f"Class Index: {index} (Confidence: {prob:.2%})"
                                        for prob, index in zip(all_probs, all_indices)]
                    self.result_label.setText("Detected Actions (Class Indices):")
                    self.actions_list.addItems(detected_actions)
                    QMessageBox.information(self, "Information",
                                            "Class labels not found in config. Showing predictions based on class indices.")

        except Exception as e:
            self.result_label.setText(f"⚠️ Detection Error: {e}")
            print(f"Detailed error during detection: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoClassifierApp()
    window.show()
    sys.exit(app.exec())