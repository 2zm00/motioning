import cv2
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from ui.video_thread import VideoThread
from ui.explorer_widget import ExplorerWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("모션 감지 분석 서비스")
        self.resize(1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        h_layout = QHBoxLayout(central_widget)

        v_layout_left = QVBoxLayout()
        self.video_label = QLabel("비디오 화면")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        v_layout_left.addWidget(self.video_label)

        h_controls = QHBoxLayout()
        self.btn_camera = QPushButton("카메라 로드")
        self.btn_camera.clicked.connect(self.load_camera)
        
        self.btn_record = QPushButton("녹화 시작")
        self.btn_record.clicked.connect(self.toggle_record)
        
        h_controls.addWidget(self.btn_camera)
        h_controls.addWidget(self.btn_record)
        v_layout_left.addLayout(h_controls)

        h_layout.addLayout(v_layout_left, stretch=3)

        v_layout_right = QVBoxLayout()
        self.explorer = ExplorerWidget()
        self.explorer.video_selected_signal.connect(self.play_video_file)
        v_layout_right.addWidget(self.explorer, stretch=2)

        self.lbl_angles = QLabel("각도 측정 대기 중...")
        self.lbl_angles.setStyleSheet("background-color: #1e1e24; color: #a3b8cc; padding: 15px; font-size: 15px; border-radius: 8px;")
        self.lbl_angles.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        v_layout_right.addWidget(self.lbl_angles, stretch=1)

        h_layout.addLayout(v_layout_right, stretch=1)

        self.thread = VideoThread(source=0)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.recording_status_signal.connect(self.update_record_status)
        self.thread.angles_update_signal.connect(self.update_angles_panel)
        self.thread.start()

    def load_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
        self.thread = VideoThread(source=0)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.recording_status_signal.connect(self.update_record_status)
        self.thread.angles_update_signal.connect(self.update_angles_panel)
        self.thread.start()

    def play_video_file(self, file_path):
        if self.thread.isRunning():
            self.thread.stop()
        self.thread = VideoThread(source=file_path)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.recording_status_signal.connect(self.update_record_status)
        self.thread.angles_update_signal.connect(self.update_angles_panel)
        self.thread.start()

    def toggle_record(self):
        self.thread.toggle_record()

    def update_record_status(self, is_recording):
        if is_recording:
            self.btn_record.setText("녹화 종료")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
        else:
            self.btn_record.setText("녹화 시작")
            self.btn_record.setStyleSheet("")
            self.explorer.load_videos() 

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convertToQtFormat = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_angles_panel(self, angles):
        def fmt(val):
            return f"{val:.0f}°" if val is not None else "N/A"

        text = (
            "<b>[실시간 각도 계산]</b><br><br>"
            "<table width='100%'>"
            f"<tr><td><b>R. Shoulder:</b> {fmt(angles['rightShoulder'])}</td>"
            f"<td><b>L. Shoulder:</b> {fmt(angles['leftShoulder'])}</td></tr>"
            
            f"<tr><td><b>R. Elbow:</b> {fmt(angles['rightElbow'])}</td>"
            f"<td><b>L. Elbow:</b> {fmt(angles['leftElbow'])}</td></tr>"
            
            f"<tr><td><b>R. Knee:</b> {fmt(angles['rightKnee'])}</td>"
            f"<td><b>L. Knee:</b> {fmt(angles['leftKnee'])}</td></tr>"
            
            f"<tr><td><b>R. Hip:</b> {fmt(angles['rightHip'])}</td>"
            f"<td><b>L. Hip:</b> {fmt(angles['leftHip'])}</td></tr>"
            "</table>"
        )
        self.lbl_angles.setText(text)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
