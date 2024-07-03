import numpy as np
import math
import cv2

longitud_brazo_x = 0.65  # m --> 0.22330 px
longitud_pierna_y = 0.94  # m --> 0.550944 px

def calculate_angle(a, b, c):
  #return np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  #El ángulo calculado se convierte de radianes a grados y se toma el valor absoluto --> Grados = Radianes * 180 / π
  angle = np.abs(radians*180.0/np.pi)

  #Se normaliza el ángulo calculado para asegurarse de que esté en el rango de 0 a 180 grados.
  if angle>180.0:
    angle = 360-angle
  # Lo retorno como radianes
  return angle * np.pi / 180.0
  

# formato output:([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]).
def extraer_posiciones(df, frame_number, *articulaciones):
    data = []
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]

    for articulacion in articulaciones:
        x = row[articulacion+ '_X'].iloc[0]
        y = row[articulacion+ '_Y'].iloc[0]

        data.append([x, y])
    return data

# Dado un dataframe y un numero de frame, retorna la velocidad instantánea correspondiente a la fila con el numero de frame pasado por parámetro.
def extraer_velocidad(df, frame_number):
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]
    return row["VelocidadAngular"].iloc[0]

def coordenadas_a_distancia(a, b):
    # Calcula la diferencia en coordenadas normalizadas
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    # Relación pixel-distancia en el eje x y en el eje y
    relacion_pixel_distancia = (0.44/0.15116006135, 0.46/0.26961168646)
    # 0.15116006135 distancia de rodilla a tobillo en frame 68 en Y (medir) = 44cm
    # 0.26961168646 distancia de cadera a rodilla en frame 68 en X (medir) = 46cm

    # Convierte la diferencia en píxeles a distancia real usando la relación pixel-distancia
    distancia_x = dx * relacion_pixel_distancia[0]
    distancia_y = dy * relacion_pixel_distancia[1]

    # Calcula la distancia euclidiana en el mundo real
    return (distancia_x**2 + distancia_y**2)**0.5

def velocidad_angular(angulo_inicial, angulo_final, delta_tiempo):
    # Calcular el cambio en el ángulo
    delta_theta = angulo_final - angulo_inicial

    # Calcular la velocidad angular
    # Recordar que omega = theta punto = vel angular = Delta theta / Delta t
    # Donde Delta theta es el cambio en rotación angular y Delta t es el cambio en el tiempo
    # ENTONCES: la velocidad angular se calcula dividiendo la diferencia total del ángulo (delta_theta) por el tiempo transcurrido entre las mediciones (1 / frame_rate).
    angular_velocity = delta_theta / delta_tiempo
    
    return angular_velocity

def velocidad_instantanea(pos_anterior, pos_actual, tiempo):
  dx = pos_actual[0] - pos_anterior[0]
  dy = pos_actual[1] - pos_anterior[1]
  return (dx/tiempo, dy/tiempo)

def aceleracion_instantanea(vel_actual_x, vel_anterior_x, vel_actual_y, vel_anterior_y, tiempo):
  dvx = vel_actual_x - vel_anterior_x
  dvy = vel_actual_y - vel_anterior_y
  return (dvx/tiempo, dvy/tiempo)

def calcular_fuerza_gemelo(df, frame_number, pos_left_knee, pos_left_ankle, pos_left_heel, pos_left_foot_index):
  # Masa del pie, esta masa es el 1.5% del peso total de la persona
  masa_pie = 0.015 * (65 / 9.8) 
  longitud_pie = ((pos_left_heel[0]-pos_left_foot_index[0])**2 + (pos_left_heel[1]-pos_left_foot_index[1])**2)**0.5
  # Distancia desde el tobillo a donde se aplica la fuerza, es decir, talon.
  distancia_momento = ((pos_left_heel[0]-pos_left_ankle[0])**2 + (pos_left_heel[1]-pos_left_ankle[1])**2)**0.5
  # Obtengo la aceleracion angular del dataframe
  aceleracionAngular = df.loc[df["frame_number"] == frame_number, "AceleracionAngular"].iloc[0]
  # Obtengo angulo entre la rodilla, tobillo, talon y lo paso a radianes para calcular el sen
  angulo_gemelo_talon = calculate_angle(pos_left_knee, pos_left_ankle, pos_left_heel)
  # Obtengo el angulo que se forma entre la punta del pie, tobillo, y un punto en la direccion del peso
  angulo_peso = calculate_angle(pos_left_foot_index, pos_left_ankle, (pos_left_ankle[0],pos_left_ankle[1]-1))
  # Angulo peso estatico
  angulo_peso_estatico = 0.6024579780018424
  # Angulo gemelo estatico
  angulo_gemelo_estatico = 2.6741865548826484
  # Distancia desde el centro del pie al tobillo para teorema de Steiner
  distancia_al_centro = 0.08  
  # Momento inercial
  momento_inercial = (1/12 * masa_pie * longitud_pie**2) + (masa_pie * distancia_al_centro**2)
  # Momento del peso en movimiento
  momento_peso_movimiento = 65 * distancia_al_centro * math.sin(angulo_peso)
  # Momento del peso estatico
  momento_peso_estatico = 65 * distancia_al_centro * math.sin(0.6024579780018424)
  # Calculo la fuerza que realiza el gemelo
  magnitud_fuerza_gemelo = abs(((momento_inercial * aceleracionAngular) + momento_peso_movimiento - momento_peso_estatico) / (-distancia_momento * math.sin(angulo_gemelo_talon) + distancia_momento * math.sin(2.6741865548826484)))
  # Vector fuerza gemelo es el vector unitario que va desde el tobillo a la rodilla
  vector_fuerza_gemelo_unitario = (pos_left_knee[0] - pos_left_ankle[0], pos_left_knee[1] - pos_left_ankle[1]) / ((pos_left_ankle[0]-pos_left_knee[0])**2 + (pos_left_ankle[1]-pos_left_knee[1])**2)**0.5
  # Al vector fuerza gemelo lo multiplico por la fuerza que realiza este y lo devuelvo
  return magnitud_fuerza_gemelo #(vector_fuerza_gemelo_unitario[0] * magnitud_fuerza_gemelo, vector_fuerza_gemelo_unitario[1] * magnitud_fuerza_gemelo)

def graficar_vector_fuerza(image, magnitud_fuerza_gemelo, pos_left_ankle, pos_left_knee, pos_left_heel, video_width, video_height):
  normalized_pos_left_ankle = (pos_left_ankle[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_ankle[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_knee = (pos_left_knee[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_knee[1] * 0.2489993274 / longitud_pierna_y))
  normalized_pos_left_heel = (pos_left_heel[0] * 0.3214562238 / longitud_brazo_x, 1-(pos_left_heel[1] * 0.2489993274 / longitud_pierna_y))
  vector = (normalized_pos_left_knee[0] - normalized_pos_left_ankle[0], normalized_pos_left_knee[1] - normalized_pos_left_ankle[1])
  
  distancia_vector_pixeles = ((vector[0] * video_width)**2 + (vector[1] * video_height)**2)**0.5
  
  versor = ((vector[0] * video_width) / distancia_vector_pixeles, (vector[1] * video_height) /distancia_vector_pixeles)
  
  #cv2.arrowedLine(image, (int(normalized_pos_left_ankle[0] * video_width) , int(normalized_pos_left_ankle[1] * video_height)) , (int(versor[0] * magnitud_fuerza_gemelo + normalized_pos_left_ankle[0] * video_width) , int(versor[1] * magnitud_fuerza_gemelo + normalized_pos_left_ankle[1] * video_height)) , (255,0,0), 4)
  cv2.arrowedLine(image, (int(normalized_pos_left_heel[0] * video_width) , int(normalized_pos_left_heel[1] * video_height)) , (int(versor[0] * (magnitud_fuerza_gemelo/2) + normalized_pos_left_heel[0] * video_width) , int(versor[1] * (magnitud_fuerza_gemelo/2) + normalized_pos_left_heel[1] * video_height)) , (0,255,0), 2)
  