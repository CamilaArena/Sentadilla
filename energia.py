import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import *

# Rutas
video_path = '/Users/camia/Desktop/proyecto/lat_tincho.mov'
output_video_path = '/Users/camia/Desktop/proyecto/tracked_video.mp4'
output_csv_path = '/Users/camia/Desktop/proyecto/pose_data.csv'

# Input usuario
peso_persona = 65  # kg
altura_persona = 176  # cm

# Crear columnas del dataframe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Defino cuales son las articulaciones que me interesa estudiar, en este caso solo necesitamos usar la cadera como punto de referencia
articulaciones = [
    mp_pose.PoseLandmark.LEFT_HIP,
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')
    columns.append(landmark.name + '_Z')

columns.append("Tiempo")
columns.append("Velocidad(Cadera)_X")
columns.append("Velocidad(Cadera)_Y")
columns.append("Energia Potencial(Cadera)")
columns.append("Energia Cinetica(Cadera)")

# Código para recorrer frames del video y realizar cálculos
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_fps = cap.get(cv2.CAP_PROP_FPS)
tiempo_por_frame = 1 / video_fps

# Inicializar DataFrame y pose para el video actual
df_completo = pd.DataFrame(columns=columns)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

frame_number = 0  # Reiniciar el contador de fotogramas para cada video

# Procesar cada fotograma del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB (el fotograma)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Procesar la imagen con MediaPipe y guardar los resultados
    results = pose.process(image)
    # Recolectar y guardar los datos de la pose en el dataframe
    pose_row = {'frame_number': frame_number}

    # Extraer posiciones
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Por cada articulacion, guarda en su posicion de X, Y, Z el resultado
        for landmark in articulaciones:
            pose_row[landmark.name + '_X'] = landmarks[landmark].x
            pose_row[landmark.name + '_Y'] = landmarks[landmark].y
            pose_row[landmark.name + '_Z'] = landmarks[landmark].z
    else:
        for landmark in articulaciones:
            pose_row[landmark.name + '_X'] = None
            pose_row[landmark.name + '_Y'] = None
            pose_row[landmark.name + '_Z'] = None

    pose_row_df = pd.DataFrame(pose_row, index=[pose_row['frame_number']])
    df_completo = pd.concat([df_completo, pose_row_df], ignore_index=True)

    # Agregar los landmarks al gráfico
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=5, circle_radius=5),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=5, circle_radius=5))

    if frame_number > 0:
        df_completo.loc[df_completo["frame_number"] == frame_number, "Tiempo"] = tiempo_por_frame * frame_number
        previous_frame = frame_number - 1

        pos_prev_left_hip = (df_completo.loc[previous_frame, 'LEFT_HIP_X'], df_completo.loc[previous_frame, 'LEFT_HIP_Y'])
        pos_actual_left_hip = (df_completo.loc[frame_number, 'LEFT_HIP_X'], df_completo.loc[frame_number, 'LEFT_HIP_Y'])

        # VELOCIDAD
        velocidad_cadera_x, velocidad_cadera_y = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"] = velocidad_cadera_x
        df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"] = velocidad_cadera_y

        # Calcula la altura maxima de la cadera en el video (altura max del salto)
        altura_cadera = calcular_altura(df_completo)

        # ENERGIA POTENCIAL
        energia_potencial_cadera = calcular_energia_potencial(peso_persona, altura_cadera)
        df_completo.loc[df_completo["frame_number"] == frame_number, "Energia Potencial(Cadera)"] = energia_potencial_cadera

        # ENERGIA CINETICA
        velocidad_total_cadera = np.sqrt(velocidad_cadera_x**2 + velocidad_cadera_y**2)
        energia_cinetica_cadera = calcular_energia_cinetica(peso_persona, velocidad_total_cadera)
        df_completo.loc[df_completo["frame_number"] == frame_number, "Energia Cinetica(Cadera)"] = energia_cinetica_cadera

    # Escribir el frame procesado en el video de salida
    video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    frame_number += 1

# Liberar recursos y guardar resultados después de procesar cada video
pose.close()
video_writer.release()
cap.release()

df_completo.to_csv(output_csv_path, index=False)

print("Proceso completado. Video trackeado guardado en:", output_video_path)
print("Datos de la pose guardados en:", output_csv_path)

#-----------------GRAFICOS-------------------

# Código para graficar energías potencial y cinética
df_completo = pd.read_csv(output_csv_path)

# Suavizar las energías potencial y cinética
window_size = 50
energia_potencial_smoothed = df_completo['Energia Potencial(Cadera)'].rolling(window=window_size).mean()
energia_cinetica_smoothed = df_completo['Energia Cinetica(Cadera)'].rolling(window=window_size).mean()

# Crear trazas para las energías
trace_energia_potencial = go.Scatter(x=df_completo['Tiempo'], y=energia_potencial_smoothed, mode='lines', name='Energía Potencial de la Cadera', line=dict(color='blue'))
trace_energia_cinetica = go.Scatter(x=df_completo['Tiempo'], y=energia_cinetica_smoothed, mode='lines', name='Energía Cinética de la Cadera', line=dict(color='red'))

# Crear la figura con subplots
fig_energias = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
fig_energias.add_trace(trace_energia_potencial, row=1, col=1)
fig_energias.add_trace(trace_energia_cinetica, row=1, col=1)

# Actualizar el diseño de la figura
fig_energias.update_layout(
    title='Energía Potencial y Cinética de la Cadera',
    xaxis=dict(title='Tiempo'),
    yaxis=dict(title='Energía (Joules)'),
    legend=dict(x=0.7, y=1.1),
    height=600,
    width=800
)

# Mostrar la figura
fig_energias.show()