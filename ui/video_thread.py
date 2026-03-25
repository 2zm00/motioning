import cv2
import time
import os
from PyQt6.QtCore import QThread, pyqtSignal
from core.pose_estimator import PoseEstimator

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)
    recording_status_signal = pyqtSignal(bool)
    angles_update_signal = pyqtSignal(dict)

    def __init__(self, source=0):
        super().__init__()
        self._run_flag = True
        self.source = source
        self.pose_estimator = PoseEstimator()
        
        self.is_recording = False
        self.video_writer = None
        self.fps = 30
        self.save_dir = "recordings"

    def set_source(self, source):
        self.source = source

    def toggle_record(self):
        self.is_recording = not self.is_recording
        if not self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording_status_signal.emit(self.is_recording)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps is None:
            self.fps = 30.0

        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            results = self.pose_estimator.process(cv_img)
            self.pose_estimator.draw_landmarks(cv_img, results)
            self.pose_estimator.draw_plumb_line(cv_img, results)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_pose = self.pose_estimator.mp_pose
                
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                angle = self.pose_estimator.calculate_angle(shoulder, elbow, wrist)
                
                h, w, _ = cv_img.shape
                cv2.putText(cv_img, f"Right Arm: {int(angle)} deg", 
                            (int(elbow[0]*w), int(elbow[1]*h) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            
                # --- 스쿼트 감지 및 동작 추론 ---
                motion_type = self.pose_estimator.guess_current_motion(landmarks)
                squat_count, squat_state, squat_angle = self.pose_estimator.analyze_squat(landmarks)
                
                # 화면 좌측 상단에 텍스트 렌더링 (y=30은 Plumb Line이 사용중이므로 그 아래에 표시)
                cv2.putText(cv_img, f"Motion: {motion_type}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(cv_img, f"Squats: {squat_count} ({squat_state})", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(cv_img, f"Knee Angle: {int(squat_angle)}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # --- 전체 8개 관절 각도 계산 및 전송 ---
                angles_dict = self.pose_estimator.calculate_all_angles(landmarks)
                self.angles_update_signal.emit(angles_dict)

            if self.is_recording:
                if self.video_writer is None:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    filename = os.path.join(self.save_dir, f"record_{int(time.time())}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (frame_w, frame_h))
                self.video_writer.write(cv_img)
            else:
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

            self.change_pixmap_signal.emit(cv_img)
            time.sleep(1 / self.fps)

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()

    def stop(self):
        self._run_flag = False
        self.wait()
        if self.video_writer is not None:
            self.video_writer.release()
