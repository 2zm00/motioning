import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal

class ExplorerWidget(QWidget):
    video_selected_signal = pyqtSignal(str)

    def __init__(self, directory="recordings"):
        super().__init__()
        self.directory = directory
        
        layout = QVBoxLayout()
        
        self.label = QLabel("녹화된 영상 목록")
        layout.addWidget(self.label)
        
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.load_videos)
        layout.addWidget(self.refresh_btn)
        
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)
        
        self.setLayout(layout)
        self.load_videos()

    def load_videos(self):
        self.list_widget.clear()
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            
        for file in os.listdir(self.directory):
            if file.endswith(('.mp4', '.avi')):
                self.list_widget.addItem(file)
                
    def on_item_double_clicked(self, item):
        file_path = os.path.join(self.directory, item.text())
        self.video_selected_signal.emit(file_path)
