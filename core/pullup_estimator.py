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
        
        # 1. 랜드마크 추출 (신체 Y좌표 위주 평균치 사용으로 안정성 극대화)
        shoulder_y = (landmarks[p.LEFT_SHOULDER.value].y + landmarks[p.RIGHT_SHOULDER.value].y) / 2.0
        elbow_y = (landmarks[p.LEFT_ELBOW.value].y + landmarks[p.RIGHT_ELBOW.value].y) / 2.0
        bar_y = (landmarks[p.LEFT_WRIST.value].y + landmarks[p.RIGHT_WRIST.value].y) / 2.0
        
        # 각도 대신 어깨와 철봉 사이의 거리(Distance)를 계산하여 디버깅용으로 활용
        dist = shoulder_y - bar_y

        # 2. 상태 전환 로직 (State Machine) - 물리 기반 완화 모델
        if self.pullup_state == 'UP':
            # UP -> DOWN 리셋: '팔꿈치가 어깨보다 위로 간다'는 조건 삭제 (인식률 저하 원인)
            # 순수하게 영상 인물의 '어깨가 철봉(손목)에서 다시 멀어짐' 만 체크 (거리를 0.25에서 0.18로 대폭 완화)
            if dist > 0.18:
                self.pullup_state = 'DOWN'
                
        elif self.pullup_state == 'DOWN':
            # DOWN -> UP 1회 인정 로직 (제안해주신 방식 적용)
            # 조건 1. 어깨가 팔꿈치와 손목 사이로 들어감 (당기면서 팔꿈치(Elbow)가 어깨(Shoulder) 아래로 내려가므로 Y값이 커짐) -> shoulder_y < elbow_y
            # 조건 2. 어깨가 손목에 얼추 접근함 -> dist < 0.15
            if shoulder_y < elbow_y and (dist < 0.15):
                self.pullup_state = 'UP'
                self.pullup_count += 1

        return self.pullup_count, self.pullup_state, dist

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


def resize_with_pad(image, target_width=1280, target_height=720):
    """원본 비율을 유지하면서 타겟 크기에 맞추고, 모자란 부분은 검은색 여백(레터박스)으로 채웁니다."""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    delta_w = target_width - new_w
    delta_h = target_height - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


if __name__ == "__main__":
    import sys
    
    # 터미널 실행 시 뒤에 동영상 파일 경로를 주면 영상을 재생, 없으면 웹캠(0) 실행
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    is_webcam = (video_source == 0)
    
    cap = cv2.VideoCapture(video_source)
    estimator = PullupEstimator()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("영상 스트림이 종료되었거나 완전히 읽었습니다.")
            break
            
        # 프레임을 1280x720 박스 안에 비율을 유지한 채 끼워넣고 남는 곳은 검정칠함
        # 이렇게 하면 y좌표 정규화(0~1) 환경에서 상하 비율이 일정해져 턱걸이 로직이 틀어지지 않음
        frame = resize_with_pad(frame, 1280, 720)
            
        # 웹캠인 경우에만 거울처럼 좌우 반전 시킴 (동영상 파일은 뷰어 방향 그대로 유지)
        if is_webcam:
            frame = cv2.flip(frame, 1)
        
        # 1. Pose 추정
        results = estimator.process(frame)
        
        count = 0
        state = 'DOWN'
        dist_val = 0.0
        
        # 2. 턱걸이 분석 실행
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            count, state, dist_val = estimator.analyze_pullup(landmarks)
            
        # 3. 마스크 및 관절 그리기 (np.where 덮어쓰기 위해 결과를 frame 에 다시 저장)
        frame = estimator.draw_landmarks(frame, results)
            
        # 4. 화면 UI 표시
        cv2.rectangle(frame, (0, 0), (280, 110), (245, 117, 16), -1)
        
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
                    
        # 거리를 눈으로 보며 튜닝할 수 있도록 Dist 수치 표시
        cv2.putText(frame, f'Dist(Shoulder-Bar): {dist_val:.2f}', (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
        # 결과 이미지 렌더링
        cv2.imshow('Pull-up Tracker Tracker', frame)
        
        # 'q' 키 입력시 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
