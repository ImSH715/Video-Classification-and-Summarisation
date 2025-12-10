import sys
import os
import cv2
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from torchvision import transforms

# Assuming these are in the same directory, otherwise, adjust the import paths
try:
    from models import CNNLSTM
    from config import Config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure that 'models.py', and 'config.py' are in the same directory as this script, or adjust the import paths accordingly.")
    sys.exit(1)

class VideoProcessor(QThread):
    """
    A QThread class to handle the video processing (classification and extraction)
    in a separate thread, to avoid freezing the GUI.
    """
    progress_update = pyqtSignal(int, str)  # Signal for progress updates
    finished = pyqtSignal(str)  # Changed signal to emit the output path
    error = pyqtSignal(str)

    def __init__(self, video_path, checkpoint_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.frame_action_map = {}

    def extract_frames(self, video_path, max_frames=Config.test_num_frames):
        """Extracts and preprocesses frames from a video file.

        Args:
            video_path (str): Path to the input video file.
            max_frames (int, optional): Maximum number of frames to extract. Defaults to Config.test_num_frames.

        Returns:
            torch.Tensor: A tensor of preprocessed frames, or None on error.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_message = f"Error: Could not open video file: {video_path}"
            self.error.emit(error_message)
            print(error_message)
            return None

        frames = []
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.test_image_height, Config.test_image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            frame_count += 1
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = transform(frame_rgb)
                frames.append(processed_frame)
                if len(frames) >= max_frames:
                    break  # Stop if max_frames is reached
            except Exception as e:
                error_message = f"Error processing frame {frame_count}: {e}"
                self.error.emit(error_message)
                print(error_message)
                continue  # Process the next frame

        cap.release()
        if not frames:
            error_message = "Error: No frames were successfully processed."
            self.error.emit(error_message)
            print(error_message)
            return None

        frames_tensor = torch.stack(frames).unsqueeze(0)  # Add batch dimension
        return frames_tensor

    def extract_video_clip_with_action(self, video_path, output_path, frame_action_map):
        """Extracts the entire video and adds the action names as a text overlay.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the extracted video clip.
            frame_action_map (dict):  Dictionary mapping frame numbers to action names.

        Returns:
            bool: True on success, False on failure.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_message = f"Error: Could not open video file: {video_path}"
            self.error.emit(error_message)
            print(error_message)
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Using codec: {fourcc}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            error_message = f"Error: Could not open output file for writing: {output_path}"
            self.error.emit(error_message)
            print(error_message)
            cap.release()
            return False

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add action name text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)  # White
            thickness = 2
            text_x = 10
            text_y = 30
            action_name_to_display = frame_action_map.get(current_frame, "")  # Get action name for the current frame

            if action_name_to_display:
                cv2.putText(frame, action_name_to_display, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        return True

    def run(self):
        """
        Runs the video classification and extraction process.  This is called by QThread.start()
        """
        device = torch.device("cpu")  # Force CPU
        print(f"Using device: {device}")

        # 1. Load the model
        try:
            self.progress_update.emit(10, "Loading model...")
            num_classes = Config.num_classes
            model = CNNLSTM(num_classes=num_classes,
                                latent_dim=Config.test_latent_dim,
                                lstm_layers=Config.test_lstm_layers,
                                hidden_dim=Config.test_hidden_dim,
                                bidirectional=Config.test_bidirectional,
                                attention=Config.test_attention)

            checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))  # Load on CPU
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                print("Model state loaded from 'ckpt['model']'.")
            else:
                model.load_state_dict(checkpoint)
                print("Model state loaded directly from checkpoint file.")
            model.eval()  # Set to evaluation mode
            print(f"Model loaded successfully onto CPU with {num_classes} output classes.")
        except Exception as e:
            error_message = f"Error loading model: {e}"
            self.error.emit(error_message)
            print(error_message)
            return

        # 2. Extract frames
        self.progress_update.emit(30, "Extracting frames...")
        frames_tensor = self.extract_frames(self.video_path)
        if frames_tensor is None:
            return  # Error occurred in extract_frames

        # 3. Classify the video
        self.progress_update.emit(50, "Classifying video...")
        try:
            with torch.no_grad():
                output = model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)
                all_probs, all_indices = torch.sort(probabilities, dim=1, descending=True)
                all_probs = all_probs.squeeze().tolist()
                all_indices = all_indices.squeeze().tolist()

                class_labels = getattr(Config, 'class_labels', None)
                if class_labels:
                    # Iterate through the entire video, detecting actions for each frame
                    video_duration_in_frames = len(frames_tensor[0])
                    for frame_number in range(video_duration_in_frames):
                        # Get the action for the current frame
                        action_index = all_indices[frame_number]
                        if 0 <= action_index < len(class_labels):
                            action_name = class_labels[action_index]
                            confidence = all_probs[frame_number]
                            print(f"Frame: {frame_number}, Detected: {action_name} (Confidence: {confidence:.2%})")
                            self.frame_action_map[frame_number] = action_name
                        else:
                            action_name = f"Class Index: {action_index}"
                            confidence = all_probs[frame_number]
                            print(f"Frame: {frame_number}, Detected:  Class Index: {action_index} (Confidence: {confidence:.2%})")
                            self.frame_action_map[frame_number] = action_name

                else:
                    self.action_name = "No Action Detected"
                    print("Warning: Class labels not found in config.")

        except Exception as e:
            error_message = f"Error during classification: {e}"
            self.error.emit(error_message)
            print(error_message)
            return

        # 4. Extract and save video clip
        self.progress_update.emit(70, "Extracting video clip...")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir, "extracted_video.mp4")
        success = self.extract_video_clip_with_action(self.video_path, output_file, self.frame_action_map)
        if success:
            print(f"Successfully extracted clip with actions")
            self.finished.emit(output_file)
        else:
            error_message = f"Failed to extract clip"
            self.error.emit(error_message)
            print(error_message)
            return
class VideoClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker_thread = None
        self.video_path = None  # Initialize video_path
        self.checkpoint_path = None  # Initialize checkpoint_path
        self.output_dir = ""  # Initialize output_dir

    def initUI(self):
        self.setWindowTitle("Multi-Action Detection App")
        self.setGeometry(100, 100, 650, 500)

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

        # Output Directory Selection
        self.output_label = QLabel("Output Directory: Not Selected", self)
        self.output_button = QPushButton("Select Output Directory", self)
        self.output_button.clicked.connect(self.select_output_directory)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setVisible(False)  # Initially hide the progress bar

        # Start Button
        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)  # Start button is initially disabled

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

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        layout.addWidget(self.progress_bar)  # Add progress bar to the layout
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def select_checkpoint_file(self):
        """Opens a dialog to select the model checkpoint file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "PyTorch Models (*.pth)")
        if file_path:
            if file_path.lower().endswith('.pth'):
                self.checkpoint_path = file_path
                self.checkpoint_label.setText(f"Checkpoint: ...{os.path.basename(file_path)}")
                self.update_start_button_state()
            else:
                QMessageBox.warning(self, "Invalid File", "Please select a valid .pth file.")
                self.checkpoint_path = None
                self.checkpoint_label.setText("Model Checkpoint (.pth): Not Selected")
                self.update_start_button_state()

    def upload_video(self):
        """Opens a dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.avi *.mp4 *.mov *.mkv)")
        if file_path:
            self.video_path = file_path
            self.video_label.setText(f"Video: ...{os.path.basename(file_path)}")
            self.display_video_thumbnail()
            self.update_start_button_state()

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

    def select_output_directory(self):
        """Opens a dialog to select the output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"Output Directory: {dir_path}")
            self.update_start_button_state()

    def update_start_button_state(self):
        """Enables the start button only if checkpoint, video, and output directory are selected."""
        ready = bool(self.checkpoint_path and self.video_path and self.output_dir)
        self.start_button.setEnabled(ready)

    def start_processing(self):
        """
        Starts the video processing in a separate thread.
        """
        if not self.checkpoint_path or not self.video_path or not self.output_dir:
            QMessageBox.warning(self, "Missing Information",
                                "Please select a checkpoint file, a video file, and an output directory.")
            return

        # Disable the Start button and other UI elements to prevent multiple processing
        self.start_button.setEnabled(False)
        self.checkpoint_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.progress_bar.setValue(0)  # Reset the progress bar
        self.progress_bar.setVisible(True)  # Show the progress bar

        # Create and start the worker thread
        self.worker_thread = VideoProcessor(self.video_path, self.checkpoint_path, self.output_dir)
        self.worker_thread.progress_update.connect(self.update_progress)
        self.worker_thread.finished.connect(self.processing_finished)
        self.worker_thread.error.connect(self.handle_error)  # Connect error signal
        self.worker_thread.start()

    def update_progress(self, progress, message):
        """Updates the progress bar and displays a message.

        Args:
            progress (int): The progress value (0-100).
            message (str): A message to display.
        """
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{progress}% - {message}")  # Update progress text

    def handle_error(self, message):
        """Handles errors that occur during processing in the worker thread."""
        QMessageBox.critical(self, "Error", message)
        self.reset_ui()

    def processing_finished(self, output_path):
        """
        Handles the completion of the video processing.
        """
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100% - Processing Complete!")
        QMessageBox.information(self, "Processing Complete", f"Video processing and extraction complete.  Video saved to {output_path}")
        print(f"Video saved to: {output_path}")
        self.reset_ui()

    def reset_ui(self):
        """
        Resets the UI elements to their initial state.
        """
        self.start_button.setEnabled(True)
        self.checkpoint_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.video_path = None
        self.checkpoint_path = None
        self.output_dir = ""
        self.video_label.setText("Input Video File: Not Selected")
        self.checkpoint_label.setText("Model Checkpoint (.pth): Not Selected")
        self.output_label.setText("Output Directory: Not Selected")
        self.image_label.setText("Video Thumbnail")
        self.image_label.setPixmap(QPixmap())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoClassifierApp()
    window.show()
    sys.exit(app.exec())

