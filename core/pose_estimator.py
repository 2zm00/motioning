import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 동작 인식을 위한 상태 변수 (State Machine)
        self.squat_state = 'UP'
        self.squat_count = 0
        self.pushup_state = 'UP'
        self.pushup_count = 0

    def process(self, frame):
        """이미지 프레임에서 Pose를 추정합니다."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        return results

    def calculate_angle(self, a, b, c):
        """세 점 a, b, c (x, y 내포)를 이용해 2D 각도를 계산합니다."""
        a = np.array(a) # 첫 번째 점
        b = np.array(b) # 두 번째 점 (중심점)
        c = np.array(c) # 세 번째 점
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def calculate_all_angles(self, landmarks):
        """좌우 어깨, 팔꿈치, 무릎, 고관절(엉덩이)의 8개 각도를 동시에 계산하여 딕셔너리로 반환합니다."""
        angles = {
            'leftShoulder': None, 'rightShoulder': None,
            'leftElbow': None, 'rightElbow': None,
            'leftKnee': None, 'rightKnee': None,
            'leftHip': None, 'rightHip': None
        }
        if not landmarks:
            return angles

        p = self.mp_pose.PoseLandmark
        try:
            # 관절 좌표 추출
            ls = [landmarks[p.LEFT_SHOULDER.value].x, landmarks[p.LEFT_SHOULDER.value].y]
            rs = [landmarks[p.RIGHT_SHOULDER.value].x, landmarks[p.RIGHT_SHOULDER.value].y]
            le = [landmarks[p.LEFT_ELBOW.value].x, landmarks[p.LEFT_ELBOW.value].y]
            re = [landmarks[p.RIGHT_ELBOW.value].x, landmarks[p.RIGHT_ELBOW.value].y]
            lw = [landmarks[p.LEFT_WRIST.value].x, landmarks[p.LEFT_WRIST.value].y]
            rw = [landmarks[p.RIGHT_WRIST.value].x, landmarks[p.RIGHT_WRIST.value].y]
            lh = [landmarks[p.LEFT_HIP.value].x, landmarks[p.LEFT_HIP.value].y]
            rh = [landmarks[p.RIGHT_HIP.value].x, landmarks[p.RIGHT_HIP.value].y]
            lk = [landmarks[p.LEFT_KNEE.value].x, landmarks[p.LEFT_KNEE.value].y]
            rk = [landmarks[p.RIGHT_KNEE.value].x, landmarks[p.RIGHT_KNEE.value].y]
            la = [landmarks[p.LEFT_ANKLE.value].x, landmarks[p.LEFT_ANKLE.value].y]
            ra = [landmarks[p.RIGHT_ANKLE.value].x, landmarks[p.RIGHT_ANKLE.value].y]

            # 팔꿈치 각도 (어깨-팔꿈치-손목)
            angles['leftElbow'] = self.calculate_angle(ls, le, lw)
            angles['rightElbow'] = self.calculate_angle(rs, re, rw)

            # 어깨 각도 (엉덩이-어깨-팔꿈치)
            angles['leftShoulder'] = self.calculate_angle(lh, ls, le)
            angles['rightShoulder'] = self.calculate_angle(rh, rs, re)

            # 무릎 각도 (엉덩이-무릎-발목)
            angles['leftKnee'] = self.calculate_angle(lh, lk, la)
            angles['rightKnee'] = self.calculate_angle(rh, rk, ra)

            # 엉덩이(고관절) 각도 (어깨-엉덩이-무릎)
            angles['leftHip'] = self.calculate_angle(ls, lh, lk)
            angles['rightHip'] = self.calculate_angle(rs, rh, rk)
        except Exception:
            pass

        return angles


    def analyze_squat(self, landmarks):
        """
        랜드마크를 바탕으로 스쿼트 동작인지 분석하고 카운트합니다.
        (오른쪽 다리 기준 예시)
        """
        if not landmarks:
            return self.squat_count, self.squat_state, 0

        # 오른쪽 엉덩이, 무릎, 발목 좌표 추출
        hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
               landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                 landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # 무릎 각도 계산
        angle = self.calculate_angle(hip, knee, ankle)

        # 상태 전환 로직 (State Machine)
        # 무릎 각도가 160도 이상이면 서있는 상태(UP)
        if angle > 160:
            self.squat_state = 'UP'
        # 무릎 각도가 90도 이하로 내려가고, 이전 상태가 UP이었다면 스쿼트 1회로 인정
        if angle < 90 and self.squat_state == 'UP':
            self.squat_state = 'DOWN'
            self.squat_count += 1

        return self.squat_count, self.squat_state, angle

    def guess_current_motion(self, landmarks):
        """
        현재 랜드마크 구조(서있는지 엎드렸는지 등)를 보고 어떤 동작을 하고 있을 가능성이 높은지 추론합니다.
        """
        if not landmarks:
            return "UNKNOWN"
            
        shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        
        # 어깨와 발목의 Y좌표(세로) 차이가 작다면 누워있거나 엎드려있음을 의미
        # (Y좌표는 화면 최상단이 0, 하단이 1.0)
        y_diff = abs(ankle_y - shoulder_y)
        
        if y_diff < 0.3:
            return "PUSH-UP / PLANK" # 몸이 가로로 누운 경우
        else:
            return "SQUAT / STANDING" # 몸이 세로로 서있는 경우

    def draw_landmarks(self, frame, results):
        """기본 관절 맵을 그립니다."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

    def draw_plumb_line(self, frame, results):
        """측면 몸통 정렬 선(Plumb Line)을 그립니다."""
        if not results.pose_landmarks:
            return

        landmarks = results.pose_landmarks.landmark
        
        # 오른쪽, 왼쪽 중 화면에 더 잘 보이는(visibility 기준) 측면을 선택합니다.
        right_vis = sum([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
        ])
        
        left_vis = sum([
            landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
        ])
        
        h, w, c = frame.shape
        
        if right_vis > left_vis:
            side = 'RIGHT'
            ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        else:
            side = 'LEFT'
            ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # 정렬선 좌표 변환
        pts = [
            (int(ear.x * w), int(ear.y * h)),
            (int(shoulder.x * w), int(shoulder.y * h)),
            (int(hip.x * w), int(hip.y * h)),
            (int(knee.x * w), int(knee.y * h)),
            (int(ankle.x * w), int(ankle.y * h))
        ]

        # 평균 X 위치를 이용한 완전 수직선 그리기 (이상적인 정렬선)
        avg_x = sum([pt[0] for pt in pts]) // len(pts)
        cv2.line(frame, (avg_x, 0), (avg_x, h), (0, 255, 0), 2)  # 녹색 수직선

        # 실제 각 포인트들을 잇는 선 (현재 자세)
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], (0, 0, 255), 3) # 빨간색 자세선
            cv2.circle(frame, pts[i], 5, (255, 0, 0), -1)
        cv2.circle(frame, pts[-1], 5, (255, 0, 0), -1)
        
        # 화면에 텍스트 표시
        cv2.putText(frame, f"{side} Plumb Line", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
