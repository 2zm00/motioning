import cv2
import mediapipe as mp
import numpy as np

class PullupEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=True  # 세그멘테이션 마스크 활성화
        )
        
        # 턱걸이 동작 인식을 위한 상태 변수
        self.pullup_state = 'DOWN'
        self.pullup_count = 0

    def process(self, frame):
        """이미지 프레임에서 Pose를 추정합니다."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        return results

    def calculate_angle(self, a, b, c):
        """세 점 a, b, c (x, y)를 이용해 2D 각도를 계산합니다."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def analyze_pullup(self, landmarks):
        """
        랜드마크를 바탕으로 턱걸이 동작을 분석하고 카운트합니다.
        가시성이 좋은 팔(왼쪽/오른쪽)을 기준으로 계산하거나,
        코(Nose)와 손목(Bar)의 위치를 비교합니다.
        """
        if not landmarks:
            return self.pullup_count, self.pullup_state, 0
        
        p = self.mp_pose.PoseLandmark
        
        # 1. 랜드마크 추출 (왼쪽 기준)
        left_shoulder = [landmarks[p.LEFT_SHOULDER.value].x, landmarks[p.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[p.LEFT_ELBOW.value].x, landmarks[p.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[p.LEFT_WRIST.value].x, landmarks[p.LEFT_WRIST.value].y]
        
        # 턱걸이 정확도를 높이기 위한 얼굴/바(Bar) 위치 비교 포인트 추출
        nose_y = landmarks[p.NOSE.value].y
        
        # 양쪽 손목의 Y 위치 평균을 철봉(Bar)의 위치로 가정
        left_wrist_y = landmarks[p.LEFT_WRIST.value].y
        right_wrist_y = landmarks[p.RIGHT_WRIST.value].y
        bar_y = (left_wrist_y + right_wrist_y) / 2.0

        # 각도 계산 (왼팔 기준) - 주로 디버그 표시용으로 사용
        angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        # 2. 상태 전환 로직 (State Machine) - 엄격한(Strict) 분리 처리
        # 하체가 잘리거나 손목이 화면 밖으로 벗어나 각도가 튀는 현상(=무한 카운트 버그) 방지용
        if self.pullup_state == 'UP':
            # 완전히 매달려서 내려간 상태를 증명하기 전까지는 DOWN으로 리셋하지 않음
            # (코가 손목보다 화면 높이의 15% 이상 아래로 확실하게 내려왔을 때만 리셋)
            if nose_y > bar_y + 0.15:
                self.pullup_state = 'DOWN'
                
        elif self.pullup_state == 'DOWN':
            # 완전히 올라가서 코가 바(손목) 위로 돌파했을 때 1회 인정
            if nose_y < bar_y:
                self.pullup_state = 'UP'
                self.pullup_count += 1

        return self.pullup_count, self.pullup_state, angle

    def draw_landmarks(self, frame, results):
        """비디오 프레임에 관절 위치, 연결선 및 파란색 Segmentation Mask 틴트를 그립니다."""
        
        # 1. 파란색 Segmentation Mask 틴트
        if getattr(results, 'segmentation_mask', None) is not None:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5
            overlay_color = np.zeros(frame.shape, dtype=np.uint8)
            overlay_color[:] = (255, 120, 50)  # BGR 포맷: 파란색 (Soft Blue)
            
            frame = np.where(condition, cv2.addWeighted(frame, 0.6, overlay_color, 0.4, 0), frame)

        # 2. 관절 마커(Pin) 및 뼈대(Line) 커스텀 그리기
        if results.pose_landmarks:
            # 첫 번째 그리기: 두꺼운 하얀 점(테두리 역할)과 얇은 하얀 선
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=6),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            # 두 번째 그리기: 내부를 채우는 작은 검은 점 (선은 그리지 않음)
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                None,  # 연결선 없이 점만 덮어씀
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=-1, circle_radius=3),
                connection_drawing_spec=None
            )
        return frame


if __name__ == "__main__":
    # 간단한 웹캠 테스트용 코드
    cap = cv2.VideoCapture(0)
    estimator = PullupEstimator()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽어올 수 없습니다.")
            break
            
        # 프레임 해상도 고정 (초기 해상도 변동으로 인한 MediaPipe Segmentation 에러 방지)
        frame = cv2.resize(frame, (1280, 720))
            
        # 화면 미러링 효과 (좌우 반전)
        frame = cv2.flip(frame, 1)
        
        # 1. Pose 추정
        results = estimator.process(frame)
        
        count = 0
        state = 'DOWN'
        
        # 2. 턱걸이 분석 실행
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            count, state, angle = estimator.analyze_pullup(landmarks)
            
        # 3. 마스크 및 관절 그리기 (np.where 덮어쓰기 위해 결과를 frame 에 다시 저장)
        frame = estimator.draw_landmarks(frame, results)
            
        # 4. 화면 UI 표시
        cv2.rectangle(frame, (0, 0), (280, 80), (245, 117, 16), -1)
        
        # 상태(State) 텍스트
        cv2.putText(frame, 'STAGE', (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, state, (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # 카운트 텍스트
        cv2.putText(frame, 'PULL-UPS', (140, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(count), (140, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                    
        # 결과 이미지 렌더링
        cv2.imshow('Pull-up Tracker Tracker', frame)
        
        # 'q' 키 입력시 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
