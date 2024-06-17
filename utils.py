import numpy as np

def calculate_angle(a, b, c):
  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

  #El ángulo calculado se convierte de radianes a grados y se toma el valor absoluto --> Grados = Radianes * 180 / π
  angle = np.abs(radians*180.0/np.pi)

  #Se normaliza el ángulo calculado para asegurarse de que esté en el rango de 0 a 180 grados.
  if angle>180.0:
    angle = 360-angle

  return angle

# formato output:([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]).
def extraer_posiciones(df, frame_number, *articulaciones):
    data = []
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]

    for articulacion in articulaciones:
        x = row[articulacion+ '_X'].iloc[0]
        y = row[articulacion+ '_Y'].iloc[0]
        z = row[articulacion+ '_Z'].iloc[0]

        data.append([x, y, z])
    return data

# Dado un dataframe y un numero de frame, retorna la velocidad instantánea correspondiente a la fila con el numero de frame pasado por parámetro.
def extraer_velocidad(df, frame_number):
    # Buscar la fila correspondiente al número de frame
    row = df[df['frame_number'] == frame_number]
    return row["Velocidad(Cadera)"].iloc[0]

def coordenadas_a_distancia(a, b):
    # Calcula la diferencia en coordenadas normalizadas
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    # Relación pixel-distancia en el eje x y en el eje y
    relacion_pixel_distancia = (0.58/0.292, 0.614)

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


#--------------FUNCIONES PARA ENERGIA------------------
def calcular_energia_potencial(masa, altura, g):
    return masa * g * altura

def calcular_energia_cinetica(masa, velocidad):
    return 0.5 * masa * (velocidad ** 2)

# Trabajo=energiamecanica.diff() --> sumatoria de todos --> trabajo total