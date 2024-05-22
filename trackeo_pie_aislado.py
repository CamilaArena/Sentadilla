import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from utils import *

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Read input video
video_path = '/Users/aitorortunio/Downloads/Fisica/caida_talon.MOV'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
tiempo_por_frame = 1/fps

print(frame_width, frame_height, fps)

# Defino cuales son las articulaciones que me interesa estudiar
articulaciones = [
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_KNEE
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')

columns.append("VelocidadAngular")

# Prepare output video
out = cv2.VideoWriter('/Users/aitorortunio/Downloads/Fisica/tracked_pie.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Prepare CSV file for landmark data
csv_file_path = '/Users/aitorortunio/Downloads/Fisica/landmarks.csv'
df = pd.DataFrame(columns=columns)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect landmarks
    result = pose.process(rgb_frame)
    
    pose_row = {'frame_number': frame_index}
    #print(result.pose_landmarks)
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        #print(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

        # Extract the required landmarks
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
        left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        # Por cada articulacion, guarda en su posicion de X, Y, Z el resultado
        for landmark in articulaciones:
                pose_row[landmark.name + '_X'] = landmarks[landmark].x
                pose_row[landmark.name + '_Y'] = landmarks[landmark].y
    else:
        for landmark in articulaciones:
                pose_row[landmark.name + '_X'] = None
                pose_row[landmark.name + '_Y'] = None

    pose_row_df = pd.DataFrame(pose_row, index=[pose_row['frame_number']])
    df = pd.concat([df, pose_row_df], ignore_index=True)

    # Convert normalized coordinates to pixel values
    h, w, _ = frame.shape
    left_ankle_coords = (int(left_ankle.x * w), int(left_ankle.y * h))
    left_heel_coords = (int(left_heel.x * w), int(left_heel.y * h))
    left_foot_index_coords = (int(left_foot_index.x * w), int(left_foot_index.y * h))
    left_knee_coords = (int(left_knee.x * w), int(left_knee.y * h))
    # Draw the landmarks
    cv2.circle(rgb_frame, left_ankle_coords, 15, (0, 255, 0), -1)
    cv2.circle(rgb_frame, left_heel_coords, 15, (0, 255, 0), -1)
    cv2.circle(rgb_frame, left_foot_index_coords, 15, (0, 255, 0), -1)
    cv2.circle(rgb_frame, left_knee_coords, 15, (0, 255, 0), -1)

    # Draw lines between the landmarks
    cv2.line(rgb_frame, left_ankle_coords, left_heel_coords, (0, 0, 255), 2)
    cv2.line(rgb_frame, left_heel_coords, left_foot_index_coords, (0, 0, 255), 2)
    cv2.line(rgb_frame, left_foot_index_coords, left_ankle_coords, (0, 0, 255), 2)
    cv2.line(rgb_frame, left_ankle_coords, left_knee_coords, (0, 0, 255), 2)

    # Extraer posiciones
    #if (frame_index > 0):
        #pos_prev_left_ankle, pos_prev_left_heel, pos_prev_left_foot_index = extraer_posiciones(df, frame_index-1, 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX')
        # VELOCIDAD ANGULAR
        #angulo_anterior = calculate_angle((pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_heel[0], pos_prev_left_heel[1]), (pos_prev_left_foot_index[0], pos_prev_left_foot_index[1]))
        #angulo_actual = calculate_angle((pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_heel[0], pos_prev_left_heel[1]), (pos_prev_left_foot_index[0], pos_prev_left_foot_index[1]))
        #vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
        #df.loc[df["frame_number"] == frame_index, "VelocidadAngular"] = vel_angular

    # Write the frame to the output video
    out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    
    frame_index += 1

    df.to_csv(csv_file_path, index=False)

# Convert lists to numpy arrays for easier calculations

# Input de datos
M = 1.25  # masa del pie en kg
L = 0.24  # longitud del pie en m
P = 65 * 9.81  # peso de la persona en N
r = L  # brazo de momento

F_m = (1/12 * M * L**2 * angular_acceleration - r * P) / r

time = np.arange(len(F_m)) / cap.get(cv2.CAP_PROP_FPS)  # tiempo en segundos

plt.figure(figsize=(10, 6))
plt.plot(time, F_m)
plt.xlabel('Time (s)')
plt.ylabel('Muscle Force (N)')
plt.title('Muscle Force vs Time')
plt.grid(True)
plt.show()

# Release resources
cap.release()
pose.close()
out.release()

cv2.destroyAllWindows()