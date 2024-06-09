import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import *
from plotly.subplots import make_subplots

# # Rutas
video_paths = ["D:/Fisica/lat_tincho.mov"]
output_video_paths = ["D:/Fisica/tracked_video.mp4"]
output_csv_paths = ["D:/Fisica/pose_data.csv"]

# # Input usuario
peso_persona = 65 #kg
altura_persona = 176 #cm
peso_pesa = 140 #kg

# # Crear columnas del dataframe
# Procesa el video y almacena los datos de las poses. Define las columnas para un DataFrame donde se guardarán las coordenadas de las articulaciones detectadas en cada cuadro del video.

# %%
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Defino cuales son las articulaciones que me interesa estudiar
articulaciones = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

columns = ['frame_number']

for landmark in articulaciones:
    columns.append(landmark.name + '_X')
    columns.append(landmark.name + '_Y')
    columns.append(landmark.name + '_Z')

columns.append("Tiempo")
columns.append("VelocidadAngular")
columns.append("Velocidad(Rodilla)_X")
columns.append("Velocidad(Rodilla)_Y")
columns.append("Velocidad(Cadera)_X")
columns.append("Velocidad(Cadera)_Y")
columns.append("Aceleracion(Rodilla)_X")
columns.append("Aceleracion(Rodilla)_Y")
columns.append("Aceleracion(Cadera)_X")
columns.append("Aceleracion(Cadera)_Y")
columns.append("Torque(Rodilla)")
columns.append("Torque(Cadera)")

# # Centro de Masa
# Es el punto donde se puede considerar que toda la masa del objeto está distribuida uniformemente.
# 
# Centro de masa = (sumatoria de las masas * sumatoria de las posiciones)/masa total
def obtenerCentroDeMasa():
   # suponiendo que la persona pesa 65kg y la barra viendo los discos pesa otros 60kg
    masaTotal = 125 # en kilos
    SumatoriaX = 0
    SumatoriaY = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        for landmark in mp_pose.PoseLandmark:
            SumatoriaX += masaTotal*results.pose_landmarks.landmark[landmark].x
            SumatoriaY += masaTotal*results.pose_landmarks.landmark[landmark].y

    SumatoriaX /= masaTotal
    SumatoriaY /= masaTotal

    # preguntar si lo que vamos a retornar nos va a dar la posicion de un frame o que nos de una coordenada
    # deberia dar coordenada
    return (SumatoriaX , SumatoriaY)

# # Dibujar diagrama
def diagrama_cuerpo(frame_number):
  #centro = obtenerCentroDeMasa()
  left_wrist, right_wrist, left_ankle =  extraer_posiciones(df_completo, frame_number, 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ANKLE')
  print()
  centro = ( (left_wrist[0] + right_wrist[0]) / 2 , (left_wrist[1] + left_ankle[1]) / 2 )
  cv2.circle(image, (int(centro[0] * video_width) , int(centro[1] * video_height)) , 20, (255,0,255), -1,3)
  # Grafico Peso
  cv2.arrowedLine(image, (int(centro[0] * video_width) , int(centro[1] * video_height)) , (int(centro[0]* video_width) , int(centro[1]* video_height + video_height/6) ) , (255,0,0), 4)
  # Grafico normal
  cv2.arrowedLine(image, (int(centro[0] * video_width) , int(centro[1] * video_height)) , (int(centro[0]* video_width) , int(centro[1]* video_height - video_height/6) ) , (255,0,0), 4)
  textoPeso =  str(peso_persona + peso_pesa)+" Peso"
  textoNormal =  str(peso_persona + peso_pesa)+" Normal"

  cv2.putText(image, textoPeso, (int(centro[0]* video_width) , int(centro[1]* video_height + video_height/6) ), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
  cv2.putText(image, textoPeso, (int(centro[0]* video_width) , int(centro[1]* video_height - video_height/6) ), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
  #cv2.line(image, int(centro[0]) * video_width , int(centro[0]-20) * video_width , (255,0,0), 4) #peso



# # Diagrama de cuerpo libre / Peso, Normal y Fuerza
# 
def diagramaDeCuerpo():
    # Valores fijos se encuentran declarados al inicio
    # Valor de la gravedad / aceleracion m/s2
    g = 9.81
    # Peso del cuerpo = 637
    # Peso de la barra / Evaluamos la barra y sus discos como un cuerpo = 1372N
    peso_barra = peso_pesa * g

    # Normal de la barra = 1372N
    normal_barra = peso_barra
    # Normal del cuerpo = 2009
    normal_cuerpo = normal_barra + peso_persona

    # Tomaremos un valor estático para simular la fuerza realizada por la persona para levantar la barra. Se aplica sobre el cuerpo
    # Segunda ley de Newton: sumatoria de fuerzas = masa * aceleracion. La fuerza debe ser mayor que 1372N
    fuerza_empuje = 1400

    if normal_cuerpo > 0:
        normal_direccion_cuerpo = "Upward"
        peso_direccion_cuerpo = "Upward"
    else:
        normal_direccion_cuerpo = "Downward"
        peso_direccion_cuerpo = "Downward"

    if normal_barra > 0:
        normal_direccion_barra = "Upward"
        peso_direccion_barra = "Upward"
    else:
        normal_direccion_barra = "Downward"
        peso_direccion_barra = "Downward"

    if fuerza_empuje > 0:
        fuerza_direccion_cuerpo = "Upward"
    else:
        fuerza_direccion_cuerpo = "Downward"

    return normal_direccion_cuerpo, peso_direccion_cuerpo, normal_direccion_barra, peso_direccion_barra, fuerza_direccion_cuerpo

def generate_free_body_diagrams():
    # Figura y ejes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Diagrama de cuerpo libre de la barra
    axs[0].arrow(0, 0, 0, -1372, head_width=50, head_length=100, fc='red', ec='red')
    axs[0].arrow(0, 0, 0, -637, head_width=50, head_length=100, fc='blue', ec='blue')
    axs[0].set_xlim(-2000, 2000)
    axs[0].set_ylim(-2000, 2000)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('Diagrama de Cuerpo libre - Barra')

    # Diagrama de cuerpo libre para el cuerpo
    axs[1].arrow(0, 0, 0, -2009, head_width=50, head_length=100, fc='red', ec='red')
    axs[1].arrow(0, 0, 0, -1400, head_width=50, head_length=100, fc='green', ec='green')
    axs[1].arrow(0, 0, 0, -637, head_width=50, head_length=100, fc='blue', ec='blue')
    axs[1].set_xlim(-2000, 2000)
    axs[1].set_ylim(-2000, 2000)
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Diagrama de cuerpo libre - Cuerpo')

    # Diagrama de cuerpo libre del cuerpo con la barra encima
    axs[2].arrow(0, 0, 0, -2009, head_width=50, head_length=100, fc='red', ec='red')
    axs[2].arrow(0, 0, 0, -2772, head_width=50, head_length=100, fc='green', ec='green')
    axs[2].arrow(0, 0, 0, -1372, head_width=50, head_length=100, fc='blue', ec='blue')
    axs[2].arrow(0, -1372, 0, -637, head_width=50, head_length=100, fc='blue', ec='blue')
    axs[2].set_xlim(-2000, 2000)
    axs[2].set_ylim(-4000, 0)
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Y')
    axs[2].set_title('Diagrama de cuerpo libre - Cuerpo con la barra')

    plt.tight_layout()
    plt.show()

#Funciones para ejecutar el codigo
diagramaDeCuerpo()
generate_free_body_diagrams()

# # Cálculo de Torque
# 
# Se analiza los torques de cadera y rodilla durante una sentadilla. Primero, identifica las posiciones de las articulaciones de la cadera, rodilla y tobillo.
# 
# Luego, calcula las distancias y ángulos entre las articulaciones, y utiliza estas medidas para estimar el torque generado por cada articulación.
# 
# TORQUE = F x brazo momento
# 
# Brazo momento = distancia entre articulación y carga (es la distancia desde el punto donde se aplica una fuerza hasta el punto de giro, en este caso el punto de giro es la cadera y la rodilla, y el punto de fuerza se hace desde la muñeca)
# F = 2/3 de lo que pesa la persona (cabeza y torso) + peso de carga
def calcular_torques(a, b, c, d):
   # left_wrist, left_hip, left_knee, left_ankle = extraer_posiciones(landmarks,frame_number ,'LEFT_WRIST', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
    e = (a[0], b[1])
    r = (a[0], c[1])
    #distancia_a_b = coordenadas_a_distancia(a, b)
    #distancia_b_c = coordenadas_a_distancia(b, c)
    #distancia_c_d = coordenadas_a_distancia(c, d)
    # Perpendicular a la fuerza
    distancia_b_e = coordenadas_a_distancia(b, e)
    distancia_c_r = coordenadas_a_distancia(c, r)

    #angulo_a_b_c = calculate_angle(a, b, c)
    #angulo_a_b_e = calculate_angle(a, b, e)

    # La biomecanica de la rodilla funciona al revez que en el resto del cuerpo.
    # angulo_b_c_d = 180 - calculate_angle(b, c, d)
    # print("angulo de 90:", calculate_angle(a,e,b))
    # print("distancia_b_e ", distancia_b_e)
    # print("(a[0] - b[0]) ", (a[0] - b[0]) * video_width)
    # print("brazo calculado con angulo", distancia_a_b * math.cos(calculate_angle(a,b,e)))

    start_point_torque_cadera = (int(b[0] * video_width), int(b[1] * video_height))
    end_point_torque_cadera = (int(e[0] * video_width), int(e[1] * video_height))
    cv2.line(image, start_point_torque_cadera, end_point_torque_cadera,(0,255,0), 3)

    start_point_torque_rodilla = (int(c[0] * video_width), int(c[1] * video_height))
    end_point_torque_rodilla = (int(r[0] * video_width), int(r[1] * video_height))
    cv2.line(image, start_point_torque_rodilla, end_point_torque_rodilla,(0,255,255), 3)
    # Torque en N.m
    torque_cadera = (peso_persona * 2/3 + peso_pesa) * distancia_b_e
    torque_rodilla = (peso_persona * 2/3 + peso_pesa) * distancia_c_r
    return [torque_rodilla, torque_cadera]
    # Comparar resultados de los torques y mostrar algo en el video

# # Visualizar trackeo articulaciones relevantes
def dibujar_articulaciones_relevantes(articulacion1, articulacion2, articulacion3, articulacion4):
    art1 = np.array(articulacion1)
    art2 = np.array(articulacion2)
    art3 = np.array(articulacion3)
    art4 = np.array(articulacion4)
    angle = calculate_angle(articulacion1, articulacion2, articulacion3)
    color = (0, 0, 255)
    #Dibujo el primer tramo de la pierna
    i1 = (int(art1[0] * video_width), int(art1[1] * video_height))
    f1 = (int(art2[0] * video_width), int(art2[1] * video_height))
    cv2.line(image, i1, f1, color, 5)

    #Dibujo el segundo tramo de la pierna
    i2 = (int(art2[0] * video_width), int(art2[1] * video_height))
    f2 = (int(art3[0] * video_width), int(art3[1] * video_height))
    cv2.line(image, i2, f2, color, 5)

    i3 = (int(art3[0] * video_width), int(art3[1] * video_height))
    f3 = (int(art4[0] * video_width), int(art4[1] * video_height))
    cv2.line(image, i3, f3, color, 5)

# # Visualizar esfuerzo piernas
def dibujar_piernas(articulacion1, articulacion2, articulacion3):
    art1 = np.array(articulacion1)
    art2 = np.array(articulacion2)
    art3 = np.array(articulacion3)
    #angle = calculate_angle(articulacion1,articulacion2,articulacion3)

    torques = calcular_torques(pos_actual_wrist, pos_actual_left_hip, pos_actual_left_knee, pos_actual_left_ankle)
    df_completo.loc[df_completo["frame_number"] == frame_number, "Torque(Rodilla)"] = torques[0]
    df_completo.loc[df_completo["frame_number"] == frame_number, "Torque(Cadera)"] = torques[1]
    #Dibujo el primer tramo de la pierna
    centro_art2 = (int(art2[0] * video_width), int(art2[1] * video_height))
    centro_art3 = (int(art3[0] * video_width), int(art3[1] * video_height))
    #f1 = (int(art2[0] * video_width), int(art2[1] * video_height))

    if(torques[0] >= torques[1] * 3/4):
      cv2.circle(image, centro_art2,20, (255,0,0), -1,3)
      cv2.circle(image, centro_art3,20, (255,0,0), -1,3)
    else:
      if(torques[1] * 1/4 < torques[0] < torques[1] * 3/4):
        color = (255,255,0)
        cv2.circle(image, centro_art2,20, (255,0,0), -1,3)
        cv2.circle(image, centro_art3,20, (255,255,0), -1,3)
      else:
        cv2.circle(image, centro_art2,20, (255,0,0), -1,3)
        cv2.circle(image, centro_art3,20, (0,255,0), -1,3)

    # Dibujo el segundo tramo de la pierna
    # f2 = (int(art3[0] * video_width), int(art3[1] * video_height))
    # cv2.circle(image, centro_art3,20, color, 3, 20)

# # Código para recorrer frames del video y realizar cálculos
# Este bloque de código recorre cada frame del video, procesa la imagen utilizando MediaPipe para detectar landmarks de la pose, y guarda los datos en un DataFrame. Luego, calcula el ángulo entre las articulaciones de la cadera, la rodilla y el tobillo. Después, dibuja los landmarks detectados en el video y guarda el video procesado en el archivo de salida. Finalmente, libera los recursos utilizados y guarda los datos de la pose en un archivo CSV.
for i, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)

    # Obtener propiedades del video
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    tiempo_por_frame = 1 / video_fps

    # Inicializar DataFrame y pose para el video actual
    df_completo = pd.DataFrame(columns=columns)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_writer = cv2.VideoWriter(output_video_paths[i], cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

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

            pos_prev_left_hip, pos_prev_left_knee, pos_prev_left_ankle = extraer_posiciones(df_completo, previous_frame, 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
            pos_actual_wrist, pos_actual_left_hip, pos_actual_left_knee, pos_actual_left_ankle = extraer_posiciones(df_completo, frame_number, 'LEFT_WRIST', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')

            # VELOCIDAD ANGULAR
            angulo_anterior = calculate_angle((pos_prev_left_hip[0], pos_prev_left_hip[1]), (pos_prev_left_knee[0], pos_prev_left_knee[1]), (pos_prev_left_ankle[0], pos_prev_left_ankle[1]))
            angulo_actual = calculate_angle((pos_actual_left_hip[0], pos_actual_left_hip[1]), (pos_actual_left_knee[0], pos_actual_left_knee[1]), (pos_actual_left_ankle[0], pos_actual_left_ankle[1]))
            vel_angular = velocidad_angular(angulo_anterior, angulo_actual, tiempo_por_frame)
            df_completo.loc[df_completo["frame_number"] == frame_number, "VelocidadAngular"] = vel_angular

            # VELOCIDAD
            velocidad_cadera = velocidad_instantanea(pos_prev_left_hip, pos_actual_left_hip, tiempo_por_frame)
            velocidad_rodilla = velocidad_instantanea(pos_prev_left_knee, pos_actual_left_knee, tiempo_por_frame)
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"] = velocidad_cadera[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"] = velocidad_cadera[1]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"] = velocidad_rodilla[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"] = velocidad_rodilla[1]

            # ACELERACION
            aceleracion_actual_cadera = aceleracion_instantanea(
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Cadera)_Y"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Cadera)_Y"].iloc[0], tiempo_por_frame)
            
            aceleracion_actual_rodilla = aceleracion_instantanea(
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_X"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == frame_number, "Velocidad(Rodilla)_Y"].iloc[0],
                df_completo.loc[df_completo["frame_number"] == previous_frame, "Velocidad(Rodilla)_Y"].iloc[0], tiempo_por_frame)
            
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_X"] = aceleracion_actual_rodilla[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Rodilla)_Y"] = aceleracion_actual_rodilla[1]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_X"] = aceleracion_actual_cadera[0]
            df_completo.loc[df_completo["frame_number"] == frame_number, "Aceleracion(Cadera)_Y"] = aceleracion_actual_cadera[1]

        # Escribir el frame procesado en el video de salida
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame_number += 1

    # Liberar recursos y guardar resultados después de procesar cada video
    pose.close()
    video_writer.release()
    cap.release()

    df_completo.to_csv(output_csv_paths[i], index=False)

    print("Proceso completado. Videos trackeados guardados en:", output_video_paths[i])
    print("Datos de la pose guardados en:", output_csv_paths[i])


# %% [markdown]
# # Gráficos

# %%

# Variables para almacenar trazas de velocidad y aceleración de todos los videos
all_vel_traces = []
all_acc_traces = []
all_pos_traces = []

for i, csv_path in enumerate(output_csv_paths):
    df_completo = pd.read_csv(csv_path)
    
    window_size = 50
    left_hip_y_smoothed = df_completo['LEFT_HIP_Y'].rolling(window=window_size).mean()
    left_knee_y_smoothed = df_completo['LEFT_KNEE_Y'].rolling(window=window_size).mean()

    #------------POSICIONES DE CADERA Y RODILLA----------------------
    trace1 = go.Scatter(x=df_completo.index, y=left_hip_y_smoothed, mode='lines', name=f'Altura de la cadera (Video {i+1})', line=dict(color='blue'))
    trace2 = go.Scatter(x=df_completo.index, y=left_knee_y_smoothed, mode='lines', name=f'Posición de la rodilla (Video {i+1})', line=dict(color='red'))

    fig_posiciones = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig_posiciones.add_trace(trace1, row=1, col=1)
    fig_posiciones.add_trace(trace2, row=1, col=1)
    fig_posiciones.update_xaxes(range=[0, df_completo.index.max()])

    fig_posiciones.update_layout(
        title=f'Evolución de la posición de la cadera y la rodilla con respecto al tiempo (Video {i+1})',
        xaxis=dict(title='Tiempo'),
        yaxis=dict(title='Posición', autorange='reversed'),  # Invertir eje Y
        yaxis2=dict(title='Posición', autorange='reversed'),  # Invertir eje Y
        legend=dict(x=0.7, y=1.1),
        height=600,
        width=800
    )

    fig_posiciones.show()

    #-------------VELOCIDAD, POSICIÓN Y ACELERACION DE CADERA-------------------
    velocidad_smoothed = df_completo['Velocidad(Cadera)_Y'].rolling(window=window_size).mean()
    aceleracion_smoothed = df_completo['Aceleracion(Cadera)_Y'].rolling(window=window_size).mean()
    posicion_cadera_smoothed = df_completo['LEFT_HIP_Y'].rolling(window=window_size).mean()

    # Crear trazas para velocidad, posición y aceleración de cada video
    vel_trace = go.Scatter(x=df_completo['Tiempo'], y=velocidad_smoothed, mode='lines', name=f'Velocidad de la cadera (Video {i+1})')
    acc_trace = go.Scatter(x=df_completo['Tiempo'], y=aceleracion_smoothed, mode='lines', name=f'Aceleración de la cadera (Video {i+1})')
    pos_trace = go.Scatter(x=df_completo['Tiempo'], y=posicion_cadera_smoothed, mode='lines', name=f'Posición de la cadera (Video {i+1})')

    # Agregar las trazas a las listas de todas las trazas
    all_vel_traces.append(vel_trace)
    all_acc_traces.append(acc_trace)
    all_pos_traces.append(pos_trace)

# Crear figura para la posición de la cadera con todas las trazas superpuestas
fig_pos = go.Figure()
for trace in all_pos_traces:
    fig_pos.add_trace(trace)
fig_pos.update_xaxes(title_text='Tiempo')
fig_pos.update_yaxes(title_text='Posición de la cadera')
fig_pos.update_layout(title='Posición de la cadera en todos los videos')
fig_pos.show()

# Crear figura para la velocidad de la cadera con todas las trazas superpuestas
fig_velocidad = go.Figure()
for trace in all_vel_traces:
    fig_velocidad.add_trace(trace)
fig_velocidad.update_xaxes(title_text='Tiempo')
fig_velocidad.update_yaxes(title_text='Velocidad de la cadera')
fig_velocidad.update_layout(title='Velocidad de la cadera en todos los videos')
fig_velocidad.show()

# Crear figura para la aceleración de la cadera con todas las trazas superpuestas
fig_aceleracion = go.Figure()
for trace in all_acc_traces:
    fig_aceleracion.add_trace(trace)
fig_aceleracion.update_xaxes(title_text='Tiempo')
fig_aceleracion.update_yaxes(title_text='Aceleración de la cadera')
fig_aceleracion.update_layout(title='Aceleración de la cadera en todos los videos')
fig_aceleracion.show()
