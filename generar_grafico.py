import pandas as pd
import matplotlib.pyplot as plt

def generar_grafico_csv(ruta_csv, columna):
    # Cargar el archivo CSV
    df = pd.read_csv(ruta_csv)
    
    # Verificar si la columna existe en el DataFrame
    if columna not in df.columns:
        print(f"La columna '{columna}' no existe en el archivo CSV.")
        return
    
    # Omitir valores NaN
    df = df.dropna(subset=[columna])
    
    # Crear el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df[columna], label=columna)
    plt.title(f'{columna} - {ruta_csv}')
    plt.xlabel('Índice')
    plt.ylabel(columna)
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso
ruta_csv_nuevo = '/Users/aitorortunio/Downloads/Fisica/landmarks.csv'
columna = 'AceleracionAngular'

generar_grafico_csv(ruta_csv_nuevo, columna)