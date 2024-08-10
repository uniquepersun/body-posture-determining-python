import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        draw_landmarks(image, results)

    cv2.imshow('body posture detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def calculate_angle(a, b, c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)

  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle

def calculate_distance(p1, p2):
  return np.linalg.norm(p1 - p2)

def calculate_ratios(landmarks):
  shoulder_left = 11
  elbow_left = 13
  wrist_left = 15
  hip_left = 23
  knee_left = 25
  ankle_left = 27
  arm_length_ratio_left = calculate_distance(landmarks[elbow_left], landmarks[wrist_left]) / \
                         calculate_distance(landmarks[shoulder_left], landmarks[elbow_left])
  leg_length_ratio_left = calculate_distance(landmarks[knee_left], landmarks[ankle_left]) / \
                         calculate_distance(landmarks[hip_left], landmarks[knee_left])
  torso_to_arm_ratio_left = calculate_distance(landmarks[shoulder_left], landmarks[hip_left]) / \
                           calculate_distance(landmarks[shoulder_left], landmarks[elbow_left])
  ratios = [arm_length_ratio_left, leg_length_ratio_left, torso_to_arm_ratio_left]

  return ratios

def extract_keypoints(results):
 
  if not results.pose_landmarks:
    return np.zeros((33, 3))

  keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
  return keypoints

def extract_features(results):
  landmarks = extract_keypoints(results)

  angles = []
  for i in range(22):
    for j in range(i + 1, 22):
      for k in range(j + 1, 22):
        angle = calculate_angle(landmarks[i * 3:(i + 1) * 3],
                               landmarks[j * 3:(j + 1) * 3],
                               landmarks[k * 3:(k + 1) * 3])
        angles.append(angle)

  distances = []
  for i in range(22):
    for j in range(i + 1, 22):
      distance = calculate_distance(landmarks[i * 3:(i + 1) * 3],
                                   landmarks[j * 3:(j + 1) * 3])
      distances.append(distance)

  ratios = calculate_ratios(landmarks)

  features = np.concatenate([landmarks, angles, distances, ratios])
  return features

cap.release()
cv2.destroyAllWindows()
