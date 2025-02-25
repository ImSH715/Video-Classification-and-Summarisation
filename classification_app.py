import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
from torchvision import transforms
from trainModel import CNNLSTM
from dataset import UCF101Dataset  # Import dataset to load class names

# Load class names from the dataset
train_dataset = UCF101Dataset("data/train")  # Adjust path if needed
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}  # Reverse mapping

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes=len(idx_to_class)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Video Frame Processing
def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()
    if len(frames) < max_frames:
        return None
    return torch.stack(frames).unsqueeze(0).to(device)

# PyQt GUI
class VideoClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Video Classification App")
        self.setGeometry(100, 100, 500, 400)

        # UI Elements
        self.label = QLabel("Upload a video file", self)
        self.label.setStyleSheet("font-size: 14px;")
        self.label.setFixedHeight(30)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 200)

        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.upload_video)

        self.predict_button = QPushButton("Predict Action", self)
        self.predict_button.clicked.connect(self.predict_action)
        self.predict_button.setEnabled(False)

        self.result_label = QLabel("", self)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.video_path = None

    def upload_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")

        if file_path:
            self.video_path = file_path
            self.label.setText(f"Selected: {file_path.split('/')[-1]}")
            self.display_video_thumbnail()
            self.predict_button.setEnabled(True)

    def display_video_thumbnail(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(300, 200))

    def predict_action(self):
        if not self.video_path:
            return

        frames = extract_frames(self.video_path)
        if frames is None:
            self.result_label.setText("⚠️ Not enough frames extracted!")
            return

        with torch.no_grad():
            output = model(frames)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()

            # Get the action name
            predicted_class = idx_to_class.get(predicted_index, "Unknown")

        self.result_label.setText(f"Predicted Action: {predicted_class} ({predicted_index})")

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoClassifierApp()
    window.show()
    sys.exit(app.exec())
