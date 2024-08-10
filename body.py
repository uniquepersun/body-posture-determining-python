import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    radians = math.acos(((a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1]) + (a[2] - b[2]) * (c[2] - b[2])) /
                       ((math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)) *
                        (math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2 + (c[2] - b[2])**2))))
    return math.degrees(radians)

def classify_posture(landmarks):
    shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    ankle_left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ankle_right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    knee_angle_left = calculate_angle(shoulder_left, hip_left, knee_left)
    knee_angle_right = calculate_angle(shoulder_right, hip_right, knee_right)
    hip_angle = calculate_angle(shoulder_left, hip_left, hip_right)
    shoulder_width = abs(shoulder_left[0] - shoulder_right[0])
    hip_width = abs(hip_left[0] - hip_right[0])
    standing_threshold = 170
    sitting_threshold = 120
    crouching_threshold = 90
    if knee_angle_left > standing_threshold and knee_angle_right > standing_threshold and shoulder_width > hip_width:
        posture = "standing"
    elif hip_angle < sitting_threshold and shoulder_width < hip_width:
        posture = "sitting"
    elif knee_angle_left < crouching_threshold and knee_angle_right < crouching_threshold:
        posture = "crouching"
    else:
        posture = "undetermined"

    return posture

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]
        posture = classify_posture(landmarks)
        cv2.putText(image, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('body posture detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
