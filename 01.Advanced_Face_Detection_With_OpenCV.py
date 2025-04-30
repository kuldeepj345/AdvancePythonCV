import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QCheckBox, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize
from datetime import datetime
import time

class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_size = (640, 480)  # Define frame size here
        self.initUI()
        self.setupCamera()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize parameters
        self.recording = False
        self.out = None
        self.filter = 'None'

        # Timer for FPS
        self.last_time = time.time()
        self.fps = 0

    def initUI(self):
        self.setWindowTitle('Enhanced Face Detection GUI')
        self.setGeometry(100, 100, 1000, 800)

        # Layouts
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Video display label
        self.video_label = QLabel(self)
        self.setVideoLabelSize()
        main_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignCenter)

        # Control panel
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel)

        # Frame size selection
        self.frame_size_combo = QComboBox(self)
        self.frame_size_combo.addItems(['640x480', '800x600', '1280x720'])
        self.frame_size_combo.setCurrentText('640x480')
        self.frame_size_combo.setToolTip('Select video frame size')
        self.frame_size_combo.currentTextChanged.connect(self.updateFrameSize)
        control_panel.addWidget(self.frame_size_combo)

        # Filter selection
        self.filter_combo = QComboBox(self)
        self.filter_combo.addItems([
            'None',
            'Grayscale',
            'Sepia',
            'Invert',
            'Sketch',
            'Blur',
            'Cartoon',
            'Emboss',
            'Edge Detection',
            'Pencil Sketch'
        ])
        self.filter_combo.setCurrentText('None')
        self.filter_combo.setToolTip('Select video filter')
        self.filter_combo.currentTextChanged.connect(self.updateFilter)
        control_panel.addWidget(self.filter_combo)

        # Filter checkbox
        self.filter_checkbox = QCheckBox('Apply Filter', self)
        self.filter_checkbox.setToolTip('Enable or disable video filter')
        self.filter_checkbox.stateChanged.connect(self.toggleFilter)
        control_panel.addWidget(self.filter_checkbox)

        # Start Recording Button
        self.start_button = QPushButton('Start Recording', self)
        self.start_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.startRecording)
        self.start_button.setToolTip('Start recording the video feed')
        control_panel.addWidget(self.start_button)

        # Stop Recording Button
        self.stop_button = QPushButton('Stop Recording', self)
        self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.stopRecording)
        self.stop_button.setToolTip('Stop recording the video feed')
        control_panel.addWidget(self.stop_button)

        # Save Snapshot Button
        self.snapshot_button = QPushButton('Save Snapshot', self)
        self.snapshot_button.setStyleSheet("background-color: blue; color: white; font-weight: bold;")
        self.snapshot_button.clicked.connect(self.saveSnapshot)
        self.snapshot_button.setToolTip('Save a snapshot of the current frame')
        control_panel.addWidget(self.snapshot_button)

        # Status label
        self.status_label = QLabel('Status: Ready', self)
        main_layout.addWidget(self.status_label, alignment=QtCore.Qt.AlignCenter)

        # Statistics label
        self.stats_label = QLabel('Faces Detected: 0', self)
        main_layout.addWidget(self.stats_label, alignment=QtCore.Qt.AlignCenter)

        # FPS label
        self.fps_label = QLabel('FPS: 0', self)
        main_layout.addWidget(self.fps_label, alignment=QtCore.Qt.AlignCenter)

        # Timer for updating frames
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def setVideoLabelSize(self):
        self.video_label.setFixedSize(QSize(*self.frame_size))

    def setupCamera(self):
        self.cap = cv2.VideoCapture(0)

    def updateFrameSize(self, size):
        if size == '640x480':
            self.frame_size = (640, 480)
        elif size == '800x600':
            self.frame_size = (800, 600)
        elif size == '1280x720':
            self.frame_size = (1280, 720)
        self.setVideoLabelSize()

    def updateFilter(self, filter):
        self.filter = filter

    def toggleFilter(self, state):
        if state == Qt.Checked:
            self.filter_checkbox.setText('Filter Enabled')
        else:
            self.filter_checkbox.setText('Filter Disabled')

    def applyFilter(self, frame):
        if self.filter == 'Grayscale':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.filter == 'Sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            # Apply sepia filter and ensure it has 3 channels
            frame_sepia = cv2.transform(frame, sepia_filter)
            return np.clip(frame_sepia, 0, 255).astype(np.uint8)
        elif self.filter == 'Invert':
            return cv2.bitwise_not(frame)

        elif self.filter == 'Sketch':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        elif self.filter == 'Blur':
            return cv2.GaussianBlur(frame, (15, 15), 0)

        elif self.filter == 'Cartoon':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon

        elif self.filter == 'Emboss':
            kernel = np.array([[ -2, -1, 0],
                            [ -1,  1, 1],
                            [  0,  1, 2]])
            embossed = cv2.filter2D(frame, -1, kernel)
            return cv2.convertScaleAbs(embossed)

        elif self.filter == 'Edge Detection':
            edges = cv2.Canny(frame, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif self.filter == 'Pencil Sketch':
            gray, sketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return sketch  # Or return gray for grayscale pencil sketch
        else:
            return frame

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Apply filter
        filtered_frame = self.applyFilter(frame)

        # Ensure the filtered frame is in color format if it was grayscale
        if len(filtered_frame.shape) == 2:  # Grayscale
            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(filtered_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

        # Update statistics
        self.stats_label.setText(f'Faces Detected: {len(faces)}')

        # Update FPS
        current_time = time.time()
        self.fps = int(1 / (current_time - self.last_time))
        self.last_time = current_time
        self.fps_label.setText(f'FPS: {self.fps}')

        if self.recording and self.out:
            self.out.write(filtered_frame)

    def startRecording(self):
        if not self.recording:
            self.recording = True
            self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, self.frame_size)
            self.status_label.setText('Status: Recording')
            self.status_label.setStyleSheet("color: green;")

    def stopRecording(self):
        if self.recording:
            self.recording = False
            self.out.release()
            self.status_label.setText('Status: Recording Stopped')
            self.status_label.setStyleSheet("color: red;")

    def saveSnapshot(self):
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f'face_snapshot_{timestamp}.png'
            cv2.imwrite(snapshot_filename, frame)
            self.status_label.setText(f'Snapshot saved as {snapshot_filename}')
            self.status_label.setStyleSheet("color: blue;")

    def closeEvent(self, event):
        self.cap.release()
        if self.recording and self.out:
            self.out.release()
        event.accept()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
    sys.exit(app.exec_())
