# summarisation_app.py
import sys
import os
import random
import glob
from pathlib import Path
import cv2  # Import OpenCV
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
                             QScrollArea, QGridLayout, QComboBox, QDialog, QLineEdit,
                             QProgressBar)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize

# Model and dataset related imports (ensure train.py, models.py, dataset.py are in the same directory)
from models import CNNLSTM  # Assuming CNNLSTM can be used for action detection
from dataset import ImageSequenceDataset, mean, std  # Import from dataset.py

class VideoSummarizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UCF-101 Video Summarization App")
        self.setGeometry(100, 100, 800, 600)

        self.model = None
        self.class_names = None
        self.sequence_length = 20  # Should match the sequence length used during training
        self.input_size = (128, 128)  # Should be similar to the input size used during training
        self.checkpoint_path = ''  # Path to the model checkpoint, initialized as empty
        self.num_classes = 101  # Define the number of classes.  Crucially, this should match your data.
        self.frame_rate = 30  # default frame rate
        self.detected_actions = []
        self.video_path = None  # add video path
        self.all_frame_paths = [] # Store all frame paths

        self.image_label = QLabel("Video Frame:")
        self.prediction_label = QLabel("Detected Actions:")
        self.model_select_button = QPushButton("Select Model Checkpoint")
        self.model_select_button.clicked.connect(self.select_model_checkpoint)
        self.video_upload_button = QPushButton("Upload Video")
        self.video_upload_button.clicked.connect(self.upload_video)
        self.action_selection_combobox = QComboBox()
        self.summarize_button = QPushButton("Summarize Selected Action")
        self.summarize_button.clicked.connect(self.summarize_video)
        self.summarize_button.setEnabled(False)  # Disable until an action is selected

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.prediction_label)
        main_layout.addWidget(self.model_select_button)
        main_layout.addWidget(self.video_upload_button)
        main_layout.addWidget(self.action_selection_combobox)
        main_layout.addWidget(self.summarize_button)
        self.setLayout(main_layout)

        self.load_class_names()
        self.frame_predictions = []
        self.temp_dir = "temp_frames"  # store frames
        os.makedirs(self.temp_dir, exist_ok=True)

    def select_model_checkpoint(self):
        """
        Opens a file dialog to allow the user to select a PyTorch checkpoint file (.pth).
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Model Checkpoint", ".",
                                                   "PyTorch Checkpoint Files (*.pth *.pt)")
        if file_path:
            self.checkpoint_path = file_path
            self.load_model()
            QMessageBox.information(self, "Model Selected",
                                    f"Model checkpoint selected: {self.checkpoint_path}")

    def load_model(self):
        if not self.checkpoint_path:
            QMessageBox.warning(self, "Warning", "Please select a model checkpoint first.")
            return

        try:
            self.model = CNNLSTM(num_classes=self.num_classes)  # Use the defined num_classes
            checkpoint = torch.load(self.checkpoint_path)
            print(f"Keys in checkpoint: {checkpoint.keys()}")  # Debugging
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
        except FileNotFoundError as e:
            error_message = f"Error: Checkpoint file not found: {self.checkpoint_path}"
            print(error_message)
            QMessageBox.critical(self, "Error", error_message)
            sys.exit()
        except Exception as e:
            error_message = f"Error: Failed to load model: {e}"
            print(error_message)
            QMessageBox.critical(self, "Error", error_message)
            sys.exit()

    def load_class_names(self):
        """Loads class names from dataset.py."""
        try:
            temp_dataset = ImageSequenceDataset(
                root_dir="dummy_path",  # Replace with actual path if needed, but only names are used here
                annotation_file="data/UCFTrainTestList",
                sequence_length=self.sequence_length,
                transform=transforms.Compose([transforms.Resize(self.input_size)]),
                train=True,
                print_classes_once=False,
            )
            self.class_names = temp_dataset.label_names
            self.num_classes = len(self.class_names)
        except Exception as e:
            error_message = f"Error: Failed to load class names: {e}"
            print(error_message)
            QMessageBox.critical(self, "Error", error_message)
            sys.exit()

    def upload_video(self):
        """
        Opens a file dialog to allow the user to select a video file.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", ".",
                                                   "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.frame_predictions = []
            self.action_selection_combobox.clear()
            self.summarize_button.setEnabled(False)  # Disable until processed
            self.detected_actions = []
            self.all_frame_paths = [] # reset
            # Create and show progress dialog.
            self.progress_dialog = QDialog(self)
            self.progress_dialog.setWindowTitle("Processing Video")
            self.progress_dialog.setMinimumWidth(400)
            progress_layout = QVBoxLayout()
            self.progress_label = QLabel("Extracting frames and detecting actions...")
            self.progress_bar = QProgressBar()
            progress_layout.addWidget(self.progress_label)
            progress_layout.addWidget(self.progress_bar)
            self.progress_dialog.setLayout(progress_layout)
            self.progress_dialog.show()

            # Process frames in a separate thread
            self.process_frames_thread = ProcessFramesThread(file_path, self.model, self.class_names,
                                                            self.input_size, self.sequence_length)
            self.process_frames_thread.prediction_signal.connect(self.display_frame_prediction)
            self.process_frames_thread.finished_signal.connect(self.on_finished_processing)
            self.process_frames_thread.progress_signal.connect(self.update_progress)  # connect progress
            self.process_frames_thread.start()

            QMessageBox.information(self, "Video Uploaded", "Video uploaded. Detecting actions...")

    def update_progress(self, progress):
        """Updates the progress bar in the dialog. Called from the thread."""
        self.progress_bar.setValue(progress)

    def on_finished_processing(self, detected_actions, all_frame_paths):  # change the argument
        """
        Called when the frame processing thread finishes.
        Populates the action selection combo box.
        """
        self.progress_dialog.close()  # close dialog
        if detected_actions:
            self.detected_actions = detected_actions
            self.action_selection_combobox.addItems(self.detected_actions)
            self.prediction_label.setText(
                f"Detected Actions: {', '.join(self.detected_actions)}")
            self.summarize_button.setEnabled(True)  # Enable the button
            self.all_frame_paths = all_frame_paths # store frame paths
        else:
            QMessageBox.warning(self, "No Actions Detected",
                                    "No actions detected in the video.")
        QMessageBox.information(self, "Finished", "Finished detecting actions.")

    def display_frame_prediction(self, frame_index, image, predicted_class_name, confidence):
        """
        Displays the prediction for a single frame.  This is called from the thread.
        """
        # Ensure this is done on the main thread
        image_label = QLabel()
        prediction_label = QLabel(
            f"Frame {frame_index + 1}: {predicted_class_name} (Confidence: {confidence:.2f}%)")
        width, height = 200, 150
        img = image.resize((width, height))
        q_image = QImage(img.tobytes("raw", "RGB"), width, height,
                            QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        image_label.setPixmap(pixmap)

        QApplication.processEvents()  # keep the GUI responsive
        self.frame_predictions.append(predicted_class_name)  # store

    def summarize_video(self):
        """
        Generates a summarized video based on the selected action.
        """
        selected_action = self.action_selection_combobox.currentText()
        if not selected_action:
            QMessageBox.warning(self, "Warning", "Please select an action to summarize.")
            return

        # 1. Get frame paths and predictions.
        frame_paths = self.all_frame_paths # use stored frame paths
        if not frame_paths:
            QMessageBox.critical(self, "Error",
                                    "Failed to extract frames from the video.")
            return

        # 2. Find frames that match the selected action.
        action_frames = []
        for i, prediction in enumerate(self.frame_predictions):
            if prediction == selected_action:
                action_frames.append(frame_paths[i])

        if not action_frames:
            QMessageBox.information(self, "No Matching Frames",
                                    f"No frames found for action: {selected_action}")
            return

        # 3. Create the summarized video.
        output_video_path = f"data/test/{selected_action}_summary.mp4"  # change the output path
        self.generate_summary_video(action_frames, output_video_path, selected_action)
        QMessageBox.information(self, "Summary Generated",
                                f"Summary video saved to {output_video_path}")
        self.open_video(output_video_path)

    def generate_summary_video(self, frame_paths, output_path, action_name):
        """
        Generates a video from a list of frame paths, adding the action name as a title.

        Args:
            frame_paths (list): List of paths to the frames.
            output_path (str): Path to save the output video.
            action_name (str): The name of the action to display as a title.
        """
        if not frame_paths:
            return
        try:
            # Get the dimensions of the first frame to use for the video writer.
            first_frame = cv2.imread(frame_paths[0])
            height, width, _ = first_frame.shape

            # Define the video writer.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 format
            out = cv2.VideoWriter(output_path, fourcc, self.frame_rate, (width, height))

            # Use a font that is compatible with OpenCV.  PIL's TrueType fonts can be used.
            font = ImageFont.truetype("arial.ttf", 24)  # You may need to adjust the path and size

            # Create a progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("Generating Summary Video")
            progress_dialog.setMinimumWidth(400)
            progress_layout = QVBoxLayout()
            progress_label = QLabel("Processing frames...")
            progress_bar = QProgressBar()
            progress_bar.setRange(0, len(frame_paths) - 1)  # Set the range
            progress_layout.addWidget(progress_label)
            progress_layout.addWidget(progress_bar)
            progress_dialog.setLayout(progress_layout)
            progress_dialog.show()

            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                # Convert the frame to a PIL Image for drawing the text.
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                # Draw the text onto the PIL image.
                draw.text((10, 10), action_name, font=font, fill=(255, 255, 255))  # White text
                # Convert the PIL image back to a cv2 frame.
                frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_cv2)
                progress_bar.setValue(i)  # Update the progress bar
                QApplication.processEvents()
            out.release()
            progress_dialog.close()
        except Exception as e:
            error_message = f"Error generating summary video: {e}"
            print(error_message)
            QMessageBox.critical(self, "Error", error_message)
            return

    def open_video(self, video_path):
        """
        Opens the video using the default player
        """
        if sys.platform.startswith('darwin'):  # macOS
            os.system(f'open "{video_path}"')
        elif os.name == 'nt':  # Windows
            os.system(f'start "" "{video_path}"')
        elif os.name == 'posix':  # Linux
            os.system(f'xdg-open "{video_path}"')
        else:
            QMessageBox.warning(self, "Cannot open video",
                                    "No default video player found.")

    def process_video_to_frames(self, video_path):
        """
        This is a function to convert a video file into a sequence of frames using OpenCV.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A list of paths to the extracted image frames.
                   Returns empty list on error.
        """
        import cv2
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return []

            # Clear the existing frames.
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

            frame_paths = []
            frame_number = 0
            while True:
                # Read the next frame
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Save the frame as a JPEG image
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_number:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_number += 1
            # Release the video
            cap.release()
            return frame_paths
        except Exception as e:
            print(f"Error processing video to frames: {e}")
            return []

    def display_frame(self, img):
        width, height = 400, 300
        img = img.resize((width, height))
        q_image = QImage(img.tobytes("raw", "RGB"), width, height,
                            QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        frame_number = int(Path(image_path).stem)
        return frame_number

    def clear_layout(self, layout):
        """
        Clears all widgets from a layout.
        """
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class ProcessFramesThread(QThread):
    """
    A thread to process video frames and make predictions.
    """
    prediction_signal = pyqtSignal(int, Image.Image, str,
                                     float)  # frame index, image, predicted class, confidence
    finished_signal = pyqtSignal(list, list)
    progress_signal = pyqtSignal(int)  # Add progress signal.

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
        Processes each frame in self.video_path, makes a prediction,
        and emits the prediction_signal.
        """
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        frame_paths = self.process_video_to_frames(self.video_path)
        self.all_frame_paths = frame_paths # store frame paths

        if not frame_paths:
            self.finished_signal.emit([], [])
            return

        self.progress_signal.emit(0)  # Initialize progress to 0.
        for i, frame_path in enumerate(frame_paths):
            img = Image.open(frame_path).convert('RGB')
            transformed_img = transform(img)
            # Make a sequence of frames.  If you have fewer than sequence_length,
            # repeat the last frame.
            if i + self.sequence_length <= len(frame_paths):
                frame_sequence = [
                    transform(Image.open(frame_paths[j]).convert('RGB'))
                    for j in range(i, i + self.sequence_length)
                ]
            else:
                frame_sequence = [
                    transform(Image.open(frame_paths[j]).convert('RGB'))
                    for j in range(i, len(frame_paths))
                ]
                padding_needed = self.sequence_length - len(frame_sequence)
                for _ in range(padding_needed):
                    frame_sequence.append(
                        transformed_img)  # pad with the last frame

            input_tensor = torch.stack(frame_sequence).unsqueeze(
                0)  # [1, sequence_length, C, H, W]

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class_index = torch.argmax(probabilities, dim=1).item()
                predicted_class_name = self.class_names[predicted_class_index]
                confidence = probabilities[0, predicted_class_index].item() * 100

            self.prediction_signal.emit(i, img, predicted_class_name,
                                         confidence)  # Emit the signal
            if predicted_class_name not in self.action_counts:
                self.action_counts[predicted_class_name] = 0
            self.action_counts[predicted_class_name] += 1
            progress = int((i + 1) / len(frame_paths) * 100)  # Calculate progress
            self.progress_signal.emit(progress)  # Emit progress.

        # Sort the actions by their counts
        sorted_actions = sorted(self.action_counts.items(),
                                key=lambda item: item[1],
                                reverse=True)
        detected_actions = [action[0] for action in sorted_actions]
        self.finished_signal.emit(detected_actions, self.all_frame_paths) # Return frame paths

    def process_video_to_frames(self, video_path):
        """
        This is a function to convert a video file into a sequence of frames using OpenCV.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A list of paths to the extracted image frames.
                   Returns empty list on error.
        """
        import cv2
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                return []

            # Clear the existing frames.
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

            frame_paths = []
            frame_number = 0
            while True:
                # Read the next frame
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Save the frame as a JPEG image
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_number:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_number += 1
            # Release the video
            cap.release()
            return frame_paths
        except Exception as e:
            print(f"Error processing video to frames: {e}")
            return []

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoSummarizationApp()
    window.show()
    sys.exit(app.exec())
