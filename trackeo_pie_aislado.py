import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from utils import *
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Read input video
video_path = '/Users/valen/Downloads/Fisica/caida_talon.MOV'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
tiempo_por_frame = 1/fps

print(frame_width, frame_height, fps)

# Define output video resolution (e.g., half of original)
output_width = frame_width // 2
output_height = frame_height // 2

# Defino cuales son las articulaciones que me interesa estudiar
articulaciones = [
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.LEFT_HIP
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')

columns.append("VelocidadAngular")
columns.append("AceleracionAngular")
columns.append("FuerzaGemelo")

# Prepare output video
out = cv2.VideoWriter('/Users/valen/Downloads/Fisica/tracked_pie_sentado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Prepare CSV file for landmark data
csv_file_path = '/Users/valen/Downloads/Fisica/landmarks.csv'
df = pd.DataFrame(columns=columns)

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (output_width, output_height))

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect landmarks
    result = pose.process(rgb_frame)

    pose_row = {'frame_number': frame_index}
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Extract landmark positions
        for landmark in articulaciones:
            pos = landmarks[landmark]
            pose_row[landmark.name + '_X'] = pos.x * (0.44/0.15116006135)
            pose_row[landmark.name + '_Y'] = (1-pos.y) * (0.46/0.26961168646)

#        # Draw landmarks
#        mp_drawing.draw_landmarks(
#            rgb_frame, 
#            result.pose_landmarks, 
#            mp_pose.POSE_CONNECTIONS)

    df = pd.concat([df, pd.DataFrame([pose_row])], ignore_index=True)
    if(frame_index>0):
        pos_prev_left_knee, pos_prev_left_ankle, pos_prev_left_heel = extraer_posiciones(df, frame_index-1, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL')
        pos_actual_left_knee, pos_actual_left_ankle, pos_actual_left_heel = extraer_posiciones(df, frame_index, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL')

        # VELOCIDAD ANGULAR
        angulo_anterior = calculate_angle((pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_heel[0], pos_prev_left_heel[1]))
        angulo_actual = calculate_angle((pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]), (pos_actual_left_heel[0], pos_actual_left_heel[1]))
        vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
        df.loc[df["frame_number"] == frame_index, "VelocidadAngular"] = vel_angular
    # Convert back to BGR for video writing
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(output_frame)
    
    frame_index += 1

# create new dataframe with smoothed data
df_nuevo = pd.DataFrame(columns=columns)
df_nuevo['frame_number'] = df['frame_number']

columnas_a_suavizar = [
    'LEFT_ANKLE_X', 'LEFT_ANKLE_Y',
    'LEFT_HEEL_X', 'LEFT_HEEL_Y',
    'LEFT_FOOT_INDEX_X', 'LEFT_FOOT_INDEX_Y',
    'LEFT_KNEE_X', 'LEFT_KNEE_Y',
    'LEFT_HIP_X', 'LEFT_HIP_Y'
]

# Aplicar el filtro Savitzky-Golay a cada columna
for columna in columnas_a_suavizar:
    df_nuevo[columna] = savgol_filter(df[columna], window_length=3, polyorder=2)

for i in range(0, frame_index-1):
    pos_prev_left_knee, pos_prev_left_ankle, pos_prev_left_foot_index = extraer_posiciones(df_nuevo, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT_INDEX')
    pos_actual_left_knee, pos_actual_left_ankle, pos_actual_left_foot_index = extraer_posiciones(df_nuevo, i+1, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT_INDEX')

    # VELOCIDAD ANGULAR
    angulo_anterior = calculate_angle((pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]), (pos_prev_left_foot_index[0], pos_prev_left_foot_index[1]))
    angulo_actual = calculate_angle((pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]), (pos_actual_left_foot_index[0], pos_actual_left_foot_index[1]))
    vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
    df_nuevo.loc[df_nuevo["frame_number"] == i, "VelocidadAngular"] = vel_angular

df_nuevo['VelocidadAngular'] = savgol_filter(df_nuevo['VelocidadAngular'], window_length=3, polyorder=2)

for i in range(0, frame_index-1):
    vel_prev, vel_actual = extraer_velocidad(df_nuevo, i), extraer_velocidad(df_nuevo, i+1)
    acel_angular = (vel_actual - vel_prev) / tiempo_por_frame
    df_nuevo.loc[df_nuevo["frame_number"] == i, "AceleracionAngular"] = acel_angular

df_nuevo['AceleracionAngular'] = savgol_filter(df_nuevo['AceleracionAngular'], window_length=3, polyorder=2)
df_nuevo.interpolate(method='linear',inplace=True)

for i in range(0, frame_index-1):
    pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index = extraer_posiciones(df_nuevo, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX')
    # Posiciones normalizadas para graficar en el video
    pos_left_knee_normalizada, pos_left_ankle_normalizada, pos_left_heel_normalizada, pos_left_foot_index_normalizada = extraer_posiciones(df, i, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX')
    magnitud_fuerza_gemelo = calcular_fuerza_gemelo(df_nuevo, i, pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index)
    #magnitud_fuerza = (vector_fuerza_gemelo[0]**2 + vector_fuerza_gemelo[1]**2)**0.5
    df_nuevo.loc[df_nuevo["frame_number"] == i, "FuerzaGemelo"] = magnitud_fuerza_gemelo

df_nuevo['FuerzaGemelo'] = savgol_filter(df_nuevo['FuerzaGemelo'], window_length=10, polyorder=2)
df_nuevo.interpolate(method='linear',inplace=True)
df_nuevo.to_csv(csv_file_path, index=False)

cap.release()
out.release()

# Read input video
video_path = '/Users/valen/Downloads/Fisica/caida_talon.MOV'
cap2 = cv2.VideoCapture(video_path)
# Prepare output video
out2 = cv2.VideoWriter('/Users/valen/Downloads/Fisica/tracked_pie_sentado2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

# Recorrer el video para dibujar el vector fuerza gemelo
frame_index = 0
while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (output_width, output_height))
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    #cv2.circle(rgb_frame, (int(40) , int(40)) , 20, (255,0,255), -1,3)
    #cv2.circle(rgb_frame, (int(0.5 * output_width) , int(0.5 * output_height)) , 20, (255,0,255), -1,3)
    pos_left_knee, pos_left_ankle, pos_left_heel = extraer_posiciones(df_nuevo, frame_index, 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_HEEL')
    magnitud_fuerza_gemelo = df_nuevo.loc[df_nuevo["frame_number"] == frame_index, "FuerzaGemelo"].iloc[0]
    graficar_vector_fuerza(rgb_frame,magnitud_fuerza_gemelo,pos_left_ankle,pos_left_knee,pos_left_heel,output_width,output_height)
    
    # Convert back to BGR for video writing
    output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out2.write(output_frame)
    
    frame_index += 1

# Release resources
cap2.release()
pose.close()
out2.release()

cv2.destroyAllWindows()
