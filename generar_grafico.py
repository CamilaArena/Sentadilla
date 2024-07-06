import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import plotly.graph_objects as go
output_csv_paths = [
    '/Users/valen/Downloads/Fisica/pose_data_tincho.csv',
    '/Users/valen/Downloads/Fisica/pose_data_aitor.csv'
    #'/Users/camia/Desktop/proyecto/pose_data3.csv'
]

def generar_grafico_csv():
    df_completo = pd.read_csv('/Users/valen/Downloads/Fisica/pose_data_tincho.csv')

    # Suavizar las energías potencial, cinética y mecánica
    posicion_cadera_y = savgol_filter(df_completo['LEFT_HIP_Y'], window_length=11, polyorder=2)

    # Crear trazas para las energías
    trace_energia_potencial = go.Scatter(x=df_completo['Tiempo'], y=posicion_cadera_y, mode='lines', name='Energía Potencial de la Cadera', line=dict(color='blue'))


