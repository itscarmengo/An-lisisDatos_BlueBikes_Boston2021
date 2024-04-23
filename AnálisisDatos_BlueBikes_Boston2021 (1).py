#!/usr/bin/env python
# coding: utf-8

# # ADQUISICIÓN DE DATOS

# In[100]:


import pandas as pd
import matplotlib.pyplot as plt #gráficos
import seaborn as sns #gráficos más sotisficadas
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


get_ipython().system('pip install beautifulsoup4')


# In[102]:


get_ipython().system('pip install selenium==3.141.0')


# In[103]:


from selenium import webdriver
from bs4 import BeautifulSoup 
import time

# URL que quiero analizar: Bluebikes trip history data
url = "https://s3.amazonaws.com/hubway-data/index.html"

driver_path = 'C:\\Users\\hp\\Documents\\5º E2 BA\\TFG ADE\\chromedriver-win64\\chromedriver.exe'#driver correcto para mi navegador Chrome

driver = webdriver.Chrome(driver_path) 

# Abro la URL con Selenium
driver.get(url)


time.sleep(5)  # Establezco este tiempo de 5 segundos para que la página cargue correctamente

# Obtengo el código fuente de la página
html = driver.page_source
driver.quit()

# Uso Beautiful Soup para analizar el HTML
soup = BeautifulSoup(html, 'html.parser')

# Encuentro todos los enlaces de la página
links = soup.find_all('a')

# Extraigo y muestro las URLs
for link in links:
    href = link.get('href')
    print(href)


# In[104]:


import os
import requests
import zipfile

# Directorio de descargas
download_dir = 'tripdata/'

# Filtro para obtener solo los enlaces de 2021 que son archivos zip
for link in links:
    href = link.get('href')
    if href.endswith('.zip') and '2021' in href:
        # Descargo el archivo zip
        file_name = os.path.basename(href)
        file_path = os.path.join('directorio/descargas', file_name)
        response = requests.get(href)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Descomprimo el archivo zip
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('directorio/descomprimido')
        
        # Elimino el archivo zip después de descomprimirlo
        os.remove(file_path)


# In[105]:


import pandas as pd


# Directorio donde se han descomprimido los archivos
extracted_dir = 'directorio/descomprimido'

# Obtengo la lista de archivos CSV en el directorio
csv_files = [file for file in os.listdir(extracted_dir) if file.endswith('.csv')]

# leo cada archivo CSV y los almaceno en una lista
dfs = []
for csv_file in csv_files:
    file_path = os.path.join(extracted_dir, csv_file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# concateno todos los dataframes en uno solo
combined_df = pd.concat(dfs, ignore_index=True)

# muestro las cinco primeras filas del dataframe
print(combined_df.head())


# # TRANSFORMACIÓN DE DATOS

# ### Feature Engineering

# In[106]:


# tenemos un par de columnas que contienen fechas. Para poder extraer más información, las pasaremos a tipo 'Datetime'
combined_df['starttime'] = pd.to_datetime(combined_df['starttime'])
combined_df['stoptime'] = pd.to_datetime(combined_df['stoptime'])


# In[107]:


combined_df['weekday_start']=combined_df['starttime'].dt.weekday
combined_df['weekday_stop']=combined_df['stoptime'].dt.weekday

combined_df['weekday_start'] = pd.Categorical(combined_df['weekday_start'], categories=range(7), ordered=True)
combined_df['weekday_stop'] = pd.Categorical(combined_df['weekday_stop'], categories=range(7), ordered=True)


# In[108]:


combined_df['fin_de_semana_start'] = (combined_df['weekday_start'] >= 5).astype(int)


# In[109]:


combined_df['fin_de_semana_stop'] = (combined_df['weekday_stop'] >= 5).astype(int)


# In[110]:


combined_df['usertype_binary'] = (combined_df['usertype'] == 'Subscriber').astype(int)


# $$d_H(p,q)= 2 * R * \arcsin{\sqrt{\sin^2{(\frac{lat1 -lat2}{2})+ \cos{(lat1)* \cos{(lat2) * \sin^2{(\frac{long1 -long2}{2})}}}}}}*1000$$

# In[111]:


#Distancia Haversine
from math import sin, cos, asin, degrees,sqrt, atan2, radians
def funct_dist_Haversine(lat_start, lon_start, lat_end, lon_end):
    R = 6373.0  # radio de La Tierra en Km
    lat_start = radians(lat_start)
    lon_start = radians(lon_start)
    lat_end = radians(lat_end)
    lon_end = radians(lon_end)
    dlon = lon_end - lon_start
    dlat = lat_end - lat_start
    a = sin(dlat/2)**2 + cos(lat_start) * cos(lat_end) * (sin(dlon/2))**2
    c = 2 * atan2( sqrt(a), sqrt(1-a) )
    return R * c * 1000 # pasamos a metros


# In[112]:


# Añado columna al conjunto de datos con la distancia Haversine
combined_df['distance_haversine'] = combined_df.apply(lambda row: funct_dist_Haversine(row['start station latitude'],
                                                                                     row['start station longitude'],
                                                                                     row['end station latitude'],
                                                                                     row['end station longitude']), axis=1)

print(combined_df.head())


# # LIMPIEZA DE DATOS

# ## Analizo si hay duplicados

# In[113]:


combined_df.duplicated().sum() #veo si hay duplicados


# ## Analizo si hay valores nulos

# In[114]:


combined_df.isnull().sum() #veo si hay valores nulos


# In[115]:


#elimino columna porque tiene muchos NAs y no tiene relevancia en la investigación
combined_df.drop('postal code', axis=1, inplace=True)
#compruebo el resultado
combined_df.isnull().sum()


# In[116]:


print(combined_df.head())


# ## Análisis EDA

# In[117]:


combined_df.shape 


# In[118]:


combined_df.dtypes #miro el tipo de las variables


# In[119]:


combined_df.info()


# In[120]:


combined_df.head(5)


# ### Análisis variables numéricas

# In[121]:


combined_df[['tripduration', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude', 'distance_haversine']].describe() #describimos las columnas numéricas


# In[122]:


col_num = ['tripduration', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude','distance_haversine']
print(combined_df[col_num].corr())


# In[123]:


#analizo las correlaciones entre variables numéricas del conjunto de datos
numeric_df = combined_df[['tripduration', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude', 'distance_haversine']]
correlation_matrix = numeric_df.corr()  

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[26]:


# creo matriz de dispersión
sample_df = combined_df.sample(frac=0.1)
sns.pairplot(combined_df[col_num],diag_kind='kde')
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
#dibujo histogramas
num_subgráf = len(col_num)
num_filas = (num_subgráf + 1) // 2  

fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=(15, 5 * num_filas))
fig.subplots_adjust(hspace=0.5)

fig.suptitle("Análisis de Bicicletas Compartidas en Boston", fontsize=16)
fig.subplots_adjust(hspace=0.5)

for i, col in enumerate(col_num):
    fila = i // 2  
    col = i % 2   
    if col_num[i] == "tripduration":
        nbins = 10
    else:
        nbins = 50
    sns.histplot(x=col_num[i], data=combined_df, ax=axes[fila, col], bins=nbins, kde=True)
    axes[fila, col].set_title(col_num[i])
    axes[fila, col].set_xlabel("Valor")  
    axes[fila, col].set_ylabel("Frecuencia") 

plt.tight_layout()
plt.show()


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# histograma para la variable "tripduration" con escala logarítmica en el eje y para afrontar gran variedad de datos
plt.figure(figsize=(10, 6))
sns.histplot(x="tripduration", data=combined_df, bins=50, kde=True)
plt.title("Histograma de Tripduration")
plt.xlabel("Valor")
plt.ylabel("Frecuencia (escala logarítmica)")
plt.yscale('log')  
plt.show()


# ### Análisis Variables Categóricas

# In[124]:


plot = combined_df['usertype'].value_counts().plot(kind='pie', autopct='%.2f', 
                                            figsize=(6, 6),
                                            title='Tipo de Usuario',
                                            colors=['#1f77b4', '#aec7e8'])


# In[125]:


import matplotlib.pyplot as plt

# me quedo con las 15 estaciones más frecuentes mediante .head(15)
startstation_counts = combined_df['start station name'].value_counts()
top_15_stations = startstation_counts.head(15)

# visualizo en un gráfico de barras
plot = top_15_stations.plot(
    kind='bar', title='Las 15 Estaciones Iniciales Más Frecuentes', figsize=(10, 6))
plot.set_xlabel('Nombre de la Estación')
plot.set_ylabel('Frecuencia')
plt.xticks(rotation=90)  # giro etiquetas del eje x para poder visualizarlas mejor
plt.show()


# In[126]:


import matplotlib.pyplot as plt

# calculo recuento de estaciones finales
end_station_counts = combined_df['end station name'].value_counts()

# me quedo con las 15 estaciones más frecuentes mediante .head(15)
top_15_end_stations = end_station_counts.head(15)

# visualizo en un gráfico de barras
plot = top_15_end_stations.plot(
    kind='bar', title='Las 15 Estaciones Finales Más Frecuentes', figsize=(10, 6))
plot.set_xlabel('Nombre de la Estación')
plot.set_ylabel('Frecuencia')
plt.xticks(rotation=90)  # giro etiquetas del eje x para poder visualizarlas mejor
plt.show()


# In[127]:


import matplotlib.pyplot as plt
import seaborn as sns

# recuento de viajes de inicio por día de la semana
weekday_counts = combined_df['weekday_start'].value_counts()

# ordeno el conjunto de datos por el día de la semana
weekday_counts = weekday_counts.sort_index()

# asigno etiquetas a los índices de los días de la semana
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# escojo paleta de colores azul para el gráfico
colores_azules = sns.color_palette("Blues", len(dias_semana))

# Visualizo información en un gráfico de tarta
plt.figure(figsize=(8, 8))
plt.pie(weekday_counts, labels=dias_semana, autopct='%1.1f%%', startangle=140, colors=colores_azules)
plt.title('Recuento de Viajes de Inicio por Día de la Semana')
plt.show()


# In[128]:


import matplotlib.pyplot as plt
import seaborn as sns

# creo un conjunto de datos con el recuento de viajes de fin por día de la semana
weekday_counts = combined_df['weekday_stop'].value_counts()

# Ordeno el conjunto de datos por el día de la semana
weekday_counts = weekday_counts.sort_index()

# asigno etiquetas a los índices de los días de la semana
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# escojo paleta de colores azul para el gráfico
colores_azules = sns.color_palette("Blues", len(dias_semana))

# Visualizo información en un gráfico de tarta
plt.figure(figsize=(8, 8))
plt.pie(weekday_counts, labels=dias_semana, autopct='%1.1f%%', startangle=140, colors=colores_azules)
plt.title('Recuento de Viajes de Fin por Día de la Semana')
plt.show()


# In[129]:


import matplotlib.pyplot as plt
import seaborn as sns

# recuento de 'fin_de_semana_start'
weekend_counts = combined_df['fin_de_semana_start'].value_counts()

# asigno etiquetas 
etiquetas = ['Día de Semana', 'Fin de Semana']

# escojo la paletas de colores azul
colores_azules = sns.color_palette("Blues", len(etiquetas))

# Visualizo en un gráfico de tarta
plt.figure(figsize=(8, 8))
plt.pie(weekend_counts, labels=etiquetas, autopct='%1.1f%%', startangle=140, colors=colores_azules)
plt.title('Distribución de Viajes entre Días de Semana y Fin de Semana')
plt.show()


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns

# calculo recuento de viajes de parada en fin de semana
weekend_counts = combined_df['fin_de_semana_stop'].value_counts()

# asigno las etiquetas para el eje x
etiquetas = ['Día de Semana', 'Fin de Semana']

# creo la paleta de colores degradados de azul
colores_azules = sns.color_palette("Blues", len(etiquetas))

# creo el gráfico de tarta 
plt.figure(figsize=(8, 8))
plt.pie(weekend_counts, labels=etiquetas, autopct='%1.1f%%', startangle=140, colors=colores_azules)
plt.title('Distribución de Viajes entre Días de Semana y Fin de Semana')
plt.show()


# In[37]:


# recuento del número de viajes realizados en cada mes y los ordeno
monthly_trip_counts = combined_df['starttime'].dt.month.value_counts().sort_index()

# asigno etiquetas
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# visualizo en gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(months, monthly_trip_counts, color='skyblue')

plt.title('Número de Viajes por Mes')
plt.xlabel('Mes')
plt.ylabel('Número de Viajes')
plt.xticks(rotation=45, ha='right')  # giro etiquetas eje x para mejor visibilidad
plt.show()


# In[67]:


# cuento el número de viajes en cada mes y los ordeno
monthly_trip_counts = combined_df['stoptime'].dt.month.value_counts().sort_index()

# creo una lista de los meses en orden para asignar estas etiquetas al eje x
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# creo el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(months, monthly_trip_counts, color='skyblue')

plt.title('Número de Viajes por Mes')
plt.xlabel('Mes')
plt.ylabel('Número de Viajes')
plt.xticks(rotation=45, ha='right')  # roto las etiquetas a 45 grados para que se visualicen mejor
plt.show()


# ## Estudiamos outliers

# In[38]:


# ajusto tamaño de la figura
plt.figure(figsize=(16, 8))

# creo los boxplots de las variables numéricas
sns.boxplot(data=combined_df[col_num],orient='h', fliersize=5, linewidth=1, notch=True)

plt.title("Boxplot")

plt.show()


# In[40]:


#regla empírica, según la cual los datos dentro de 3 veces la desviación estándar respecto a la media representan el 99.7% de los datos de la distribución

from scipy.stats import zscore

# calculo z-score para cada dato y los pongo en valor absoluto
z_scores = zscore(combined_df['tripduration'])
abs_z_scores = np.abs(z_scores)

# identifico como outliers aquellos que superen el umbral de 3
outliers = combined_df[abs_z_scores > 3]
outliers.head()

print(f'Número de outliers: {len(outliers)}')


# #### Para tratar con los outliers, vamos a emplear el método de winsorización 
# Técnica que reemplaza los valores atípicos por el valor más cercano que no se considera un outlier según ciertos criterios 

# In[41]:


from scipy.stats.mstats import winsorize

combined_df_winsorized = combined_df.copy()
combined_df_winsorized['tripduration'] = winsorize(combined_df_winsorized['tripduration'],  limits = [0.05, 0.05], inplace = True)


# In[42]:


#boxplot de tripduration antes de emplear el método
sns.boxplot(data=combined_df['tripduration'],orient='h', fliersize=5, linewidth=1, notch=True)


# In[43]:


#boxplot de tripduration después de emplear el método
sns.boxplot(data=combined_df_winsorized['tripduration'],orient='h', fliersize=5, linewidth=1, notch=True)


# #### Me descargo el dataframe en formato excel para hacer la visualización de datos con PowerBI

# In[77]:


ruta_archivo_csv = 'C:\\Users\\hp\\Downloads\\BostonBPS21.csv'
combined_df.to_csv(ruta_archivo_csv, index=False)
try:
    with open(ruta_archivo_csv, 'r') as f:
        print("El archivo CSV se ha guardado correctamente.")
except FileNotFoundError:
    print("Hubo un problema al guardar el archivo CSV.")


# ## Pregunta 1
# ¿Qué bicicletas tienen que ir a mantenimiento?

# #### Paso 1: Cuento la frecuencia de uso de cada bicicleta

# In[44]:


bike_counts = combined_df['bikeid'].value_counts()
print(bike_counts) #cuento cuántos viajes ha realizado cada bicicleta


# In[45]:


#me quedo con las bicicletas que más viajes han realizado
top_bicicletas = bike_counts.head(10)
print(top_bicicletas)


# In[46]:


# filtro el dataset para quedarme con la información de los viajes de esas bicicletas más usadas
viajes_top_bicicletas = combined_df[combined_df['bikeid'].isin(top_bicicletas.index)]
print(viajes_top_bicicletas)


# In[49]:


viajes_top_bicicletas.describe() #para ver un resumen estadístico de los viajes de las 10 bicis más usadas


# In[50]:


# calculo la media y la desviación estándar del número de viajes
media_viajes = bike_counts.mean()
print(media_viajes)
desviacion_viajes = bike_counts.std()
print(desviacion_viajes)
      
# defino el umbral a partir del cual se considera que una bicicleta tiene que ir a mantenimiento
umbral_mantenimiento = media_viajes + 2 * desviacion_viajes  # media más dos desviaciones estándar
print(umbral_mantenimiento)

# identifico las bicicletas que necesitan mantenimiento
bicicletas_mantenimiento = bike_counts[bike_counts > umbral_mantenimiento].index

print("Las bicicletas que tienen que ir a mantenimiento son:")
print(bicicletas_mantenimiento)


# In[51]:


viajes_bicicletas_mantenimiento = combined_df[combined_df['bikeid'].isin(bicicletas_mantenimiento)]
viajes_por_bicicleta_mantenimiento = viajes_bicicletas_mantenimiento['bikeid'].value_counts()
plt.figure(figsize=(10, 6))
grafico = viajes_por_bicicleta_mantenimiento.plot(kind='bar', color='skyblue')
plt.title('Número de viajes de bicicletas que necesitan mantenimiento')
plt.xlabel('ID de bicicleta')
plt.ylabel('Número de viajes')

# Agregar una línea horizontal en el umbral de mantenimiento
plt.axhline(y=umbral_mantenimiento, color='red', linestyle='--', label='Umbral de mantenimiento')

# Agregar etiquetas de datos en las barras
for barra in grafico.patches:
    grafico.annotate(format(barra.get_height(), '.0f'),
                     (barra.get_x() + barra.get_width() / 2,
                      barra.get_height()),
                     ha='center', va='center',
                     xytext=(0, 5),
                     textcoords='offset points')

plt.legend()
plt.show()

print(umbral_mantenimiento)


# In[53]:


# obtengo las estaciones de inicio únicas para las bicicletas de mantenimiento
estacionesinicio_bibicletasmantenimiento = viajes_bicicletas_mantenimiento['start station id'].unique()

# cuento el número de viajes que realizan las bicicletas de mantenimiento por estación de inicio
viajes_por_estacion_inicio = viajes_bicicletas_mantenimiento.groupby('start station id').size()

# ordeno las estaciones por número de viajes en orden descendente
viajes_por_estacion_inicio = viajes_por_estacion_inicio.sort_values(ascending=False)

# selecciono las 10 estaciones con más viajes
top_estacionesinicio_bicicletasmantenimiento = viajes_por_estacion_inicio.head(10)
# elaboro un gráfico de barras
plt.figure(figsize=(12, 6))
top_estacionesinicio_bicicletasmantenimiento.plot(kind='bar', color='skyblue')
plt.title('Top diez estaciones de inicio con mayor número de viajes realizados por bicicletas que requieren mantenimiento')
plt.xlabel('ID de estación')
plt.ylabel('Número de viajes')
for i, v in enumerate(top_estacionesinicio_bicicletasmantenimiento):
    plt.text(i, v + 0.1, str(v), ha='center')

plt.show()


# In[54]:


# obtengo las estaciones de parada únicas para las bicicletas que necesitan mantenimiento
estacionesfin_bibicletasmantenimiento = viajes_bicicletas_mantenimiento['end station id'].unique()

# cuento el número de viajes que realizan las bicicletas de mantenimiento por estación de fin
viajes_por_estacion_fin = viajes_bicicletas_mantenimiento.groupby('end station id').size()

# ordeno las estaciones por número de viajes en orden descendente
viajes_por_estacion_fin = viajes_por_estacion_fin.sort_values(ascending=False)

# selecciono las 10 estaciones con más viajes
top_estacionesfin_bicicletasmantenimiento = viajes_por_estacion_fin.head(10)
# elaboro un gráfico de barras
plt.figure(figsize=(12, 6))
top_estacionesfin_bicicletasmantenimiento.plot(kind='bar', color='skyblue')
plt.title('Top diez estaciones de parada con mayor número de viajes realizados por bicicletas que requieren mantenimiento')
plt.xlabel('ID de estación')
plt.ylabel('Número de viajes')
for i, v in enumerate(top_estacionesfin_bicicletasmantenimiento):
    plt.text(i, v + 0.1, str(v), ha='center')

plt.show()

print(viajes_bicicletas_mantenimiento.loc[viajes_bicicletas_mantenimiento['start station id'] == 67, 'start station name'].iloc[0])


# In[56]:


# obtengo las 10 rutas más realizadas
top_rutas = viajes_bicicletas_mantenimiento.groupby(['start station id', 'end station id']).size().nlargest(10)
# itero sobre las rutas más realizadas
for (start_id, end_id), bike_counts in top_rutas.items():
    # obtengo el nombre de la estación de inicio y fin
    start_name = viajes_bicicletas_mantenimiento[viajes_bicicletas_mantenimiento['start station id'] == start_id]['start station name'].iloc[0]
    end_name = viajes_bicicletas_mantenimiento[viajes_bicicletas_mantenimiento['end station id'] == end_id]['end station name'].iloc[0]
    
    print(f'Ruta: {start_name} -> {end_name}, Viajes registrados: {bike_counts}')


# In[94]:


import folium
from folium.plugins import HeatMap
from folium.vector_layers import PolyLine, CircleMarker

# obtengo las 10 rutas más realizadas
top_rutas = viajes_bicicletas_mantenimiento.groupby(['start station id', 'end station id', 'start station latitude', 'start station longitude', 'end station latitude', 'end station longitude']).size().nlargest(10).index

# creo un mapa centrado en Boston introduciendo como input sus coordenadas de latitud y longitud
mapa = folium.Map(location=[42.3601, -71.0589], zoom_start=1)

# agrego una capa de calor para mostrar la densidad de viajes
HeatMap(data=viajes_bicicletas_mantenimiento[['start station latitude', 'start station longitude']]).add_to(mapa)

# itero sobre las 10 rutas más realizadas
for i, (start_station, stop_station, start_lat, start_lng, end_lat, end_lng) in enumerate(top_rutas):
    # creo una línea entre las estaciones de inicio y fin para que se visualice bien la ruta
    polyline = PolyLine(locations=[(start_lat, start_lng), (end_lat, end_lng)], color='blue', weight=5, opacity=0.8).add_to(mapa)
    
    # agrego iconos para identificar las estaciones, con color verde las de inicio y con color rojo las de parada
    folium.Marker([start_lat, start_lng], icon=folium.Icon(color='green')).add_to(mapa)
    folium.Marker([end_lat, end_lng], icon=folium.Icon(color='red')).add_to(mapa)

mapa

mapa.save('mapa_rutas_boston.html')


# ## Pregunta 2: ¿Existe relación entre distancia entre estaciones de Blue Bikes en Boston y uso del servicio por parte de los usuarios?

# In[131]:


import matplotlib.pyplot as plt
import seaborn as sns

# histograma variable "tripduration"
plt.figure(figsize=(8, 6))
sns.histplot(x="distance_haversine", data=combined_df, bins=50, kde=True)
plt.title("Histograma de Distancia Haversine")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.show()


# In[132]:


combined_df['distance_haversine'].head()


# In[133]:


combined_df['distance_haversine'].describe()


# In[134]:


# regla empírica, según la cual los datos dentro de 3 veces la desviación estándar respecto a la media representan el 99.7% de los datos de la distribución. Sabiendo esto, podemos concluir con bastante seguridad que los datos que caen más allá de este umbral son atípicos, pues son distintos al 99.7% de los datos
from scipy.stats import zscore

# Calculo z-score de cada dato y los pongo en valor absoluto
z_scores = zscore(combined_df['distance_haversine'])
abs_z_scores = np.abs(z_scores)

# identifico como outliers aquellos valores que superen el umbral de 3
outliers = combined_df[abs_z_scores > 3]
outliers.head()

print(f'Número de outliers: {len(outliers)}')


# In[135]:


combined_df.describe()


# In[136]:


combined_df.head()


# In[138]:


from scipy.stats.mstats import winsorize
import seaborn as sns

# empleo método winsorización 
combined_df['distance_haversine'] = winsorize(combined_df['distance_haversine'], limits=[0.0025, 0.0025])
combined_df['tripduration'] = winsorize(combined_df['tripduration'], limits=[0.0025, 0.0025])


# In[67]:


#este es el boxplot de tripduration antes de emplear el método
sns.boxplot(data=combined_df['tripduration'],orient='h', fliersize=5, linewidth=1, notch=True)


# In[68]:


#este es el boxplot de tripduration antes de emplear el método
sns.boxplot(data=combined_df_winsorized['tripduration'],orient='h', fliersize=5, linewidth=1, notch=True)


# In[139]:


import seaborn as sns

sns.regplot(data=combined_df_winsorized, x='distance_haversine', y='tripduration',
            scatter_kws={"color": "blue", "alpha": 0.5},
            line_kws={"color": "red"})


# In[140]:


# establezco los intervalos de distancia 
intervalos = [0, 1000, 2000, 40000]

# asigno las etiquetas para los grupos de distancia 
etiquetas = ['Distancia Corta', 'Distancia Media', 'Distancia Larga']

# añado una nueva columna al conjunto de datos con la clasificación de la variable distancia en las categorías que hemos definido
combined_df['Grupo_distancia'] = pd.cut(combined_df['distance_haversine'], bins=intervalos, labels=etiquetas, include_lowest=True)

# imprimo las cinco primeras filas del DataFrame con la nueva columna de grupos de distancia
print(combined_df[['start station id', 'distance_haversine', 'Grupo_distancia']].head())


# In[141]:


metrics_by_distance = combined_df.groupby('Grupo_distancia').agg({
    'tripduration': 'mean',  # duración promedio del viaje
    'bikeid': 'count'  # número de viajes
})
print(metrics_by_distance)


# In[142]:


import matplotlib.pyplot as plt

# recuento de cada grupo de distancia
frecuencia_por_grupo = combined_df['Grupo_distancia'].value_counts()

# histograma con viajes realizados por grupo de distancia
plt.figure(figsize=(10, 6))
ax=sns.barplot(x=frecuencia_por_grupo.index, y=frecuencia_por_grupo.values, palette="Blues_d")
for p in ax.patches:
    ax.annotate('{:,.0f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                textcoords='offset points')


# In[143]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# establezco los intervalos para los días de la semana 
intervalos = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

# asigno las etiquetas para los días de la semana
etiquetas = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# agrego una nueva columna al conjunto de datos con la clasificación de los días en los días de la semana, en lugar de tener el índice de 0 a 6
combined_df['Días'] = pd.cut(combined_df['starttime'].dt.weekday, bins=intervalos, labels=etiquetas, include_lowest=True)

# cuento el número de viajes por día de la semana y grupo de distancia 
day_counts = combined_df.groupby(['Días', 'Grupo_distancia']).size().unstack()
print(day_counts)
# visualizo los resultados en un gráfico de barras
ax = day_counts.plot(kind='bar', stacked=True)
plt.xlabel('Día de la semana')
plt.ylabel('Número de viajes')
plt.title('Número de viajes por grupo de distancia y día de la semana')
plt.legend(title='Grupo de distancia')
plt.xticks(rotation=45)

# asigno etiquetas de datos para cada grupo de distancia en su correspondiente segmento de la barra acumulada 
for p in ax.containers:
    ax.bar_label(p, label_type='center')

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[76]:


import matplotlib.pyplot as plt

# establezco intervalos de estaciones
intervalos = [-1, 2, 5, 8, 11]

# asigno etiquetas de las estaciones del año
etiquetas = ['Invierno', 'Primavera', 'Verano', 'Otoño']

# añado una nueva columna al conjunto de datos con la clasificación de los meses en estaciones del año
combined_df['Estaciones'] = pd.cut(combined_df['starttime'].dt.month, bins=intervalos, labels=etiquetas, include_lowest=True)

# cuento el número de viajes por estación del año y grupo de distancia 
station_counts = combined_df.groupby(['Estaciones', 'Grupo_distancia']).size().unstack()

# visualizo los resultados en un gráfico de barras
ax = station_counts.plot(kind='bar', stacked=True)
plt.xlabel('Estación del año')
plt.ylabel('Número de viajes')
plt.title('Número de viajes por grupo de distancia y estación del año')
plt.legend(title='Grupo de distancia')
plt.xticks(rotation=45)

# asigno etiquetas de datos para cada grupo de distancia en su correspondiente segmento de la barra acumulada 
for p in ax.containers:
    ax.bar_label(p, label_type='center')

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.show()


# In[144]:


# calculo la cantidad de viajes realizados por tipo de usuario para cada grupo de distancia
viajes_por_grupo_y_usuario = combined_df.groupby(['Grupo_distancia', 'usertype']).size()
print(viajes_por_grupo_y_usuario)


# In[145]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# calculo el número de viajes realizados por tipo de usuario para cada grupo de distancia
counts = combined_df.groupby(['Grupo_distancia', 'usertype']).size().unstack()

sns.set_palette("Greens")

# visualizo el número de viajes por grupo de distancia y tipo de usuario
ax = counts.plot(kind='bar', stacked=True)

plt.xlabel('Grupo de distancia')
plt.ylabel('Número de viajes')
plt.title('Número de viajes por grupo de distancia y tipo de usuario')
plt.legend(title='Tipo de usuario', loc='upper right')

# asigno etiquetas de datos para cada tipo de usuario en su correspondiente segmento de la barra acumulada 
for p in ax.patches:
    ax.annotate(format(int(p.get_height()), ','), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

plt.show()


# ## Pregunta 3: ¿Cómo afecta la hora del día, la latitud de origen y la latitud de destino a la duración de los viajes en bicicleta compartida?

# In[82]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


# In[83]:


# creo variables hora de inicio y hora de parada
combined_df['HoradeldíaInicio'] = combined_df['starttime'].dt.hour
combined_df['HoradeldíaParada'] = combined_df['stoptime'].dt.hour

X = combined_df[['start station latitude', 'end station latitude', 'HoradeldíaInicio','HoradeldíaParada','distance_haversine']]
y = combined_df['tripduration']


# In[84]:


# se ha reducido el tamaño del conjunto de entrenamiento para agilizar la carga pues son demasiadas observaciones
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100000, test_size=0.2, random_state=125)

print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]}")
print(f"Tamaño conjunto test: {X_test.shape[0]}")


# In[85]:


# se normalizan los datos de entrenamiento y test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[87]:


# creo el modelo de Random Forest
random_forest = RandomForestRegressor(n_estimators=10, criterion = 'squared_error',max_depth = None,max_features = 1,oob_score= False,n_jobs= -1, random_state=123)


# In[88]:


# entreno el modelo
random_forest.fit(X_train_scaled, y_train)


# In[89]:


# predicciones en el conjunto de prueba
y_pred = random_forest.predict(X_test_scaled)


# In[92]:


mse = mean_squared_error(y_test, y_pred)
print("El error cuadrático medio:", mse)


# In[93]:


# Búsqueda de hiperparámetro óptimo de número de árboles mediante la validación con Out-of-Bag error
# ==============================================================================
warnings.filterwarnings('ignore')
train_scores = []
oob_scores   = []
estimator_range = range(1, 150, 20)

for n_estimators in estimator_range:
    modelo = RandomForestRegressor(
                n_estimators = n_estimators,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = 1,
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )
    modelo.fit(X_train, y_train)
    train_scores.append(modelo.score(X_train, y_train))
    oob_scores.append(modelo.oob_score_)


# In[94]:


# visualizo en un gráfico la evolución del error out-of-bag con diferentes valores de número de árboles
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
        marker='o', color = "red", label="max score")
ax.set_ylabel("R^2")
ax.set_xlabel("número de árboles")
ax.set_title("Evolución del out-of-bag-error vs número árboles")
plt.legend();
print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(oob_scores)]}")
warnings.filterwarnings('default')


# In[96]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# In[97]:


train_scores = []
cv_scores    = []

estimator_range = range(1, 150, 20)

for n_estimators in estimator_range:
    
    modelo = RandomForestRegressor(
                n_estimators = n_estimators,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = 1,
                oob_score    = False,
                n_jobs       = -1,
                random_state = 123
             )
    # error de entrenamiento
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X=X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, predicciones))
    train_scores.append(rmse_train)
    
    # error de validación cruzada
    scores = cross_val_score(
                estimator = modelo,
                X         = X_train,
                y         = y_train,
                scoring   = 'neg_root_mean_squared_error',
                cv        = 5
             )
    cv_scores.append(-1*scores.mean())


# In[85]:


# visualizo en un gráfico la evolución del error de validación cruzada con diferentes valores de número de árboles
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
        marker='o', color = "red", label="min score")
ax.set_ylabel("root_mean_squared_error")
ax.set_xlabel("n_estimators")
ax.set_title("Evolución del cv-error vs número árboles")
plt.legend();
print(f"Valor óptimo de n_estimators: {estimator_range[np.argmin(cv_scores)]}")


# In[86]:


# Búsqueda de hiperparámetro óptimo de número de predictores mediante la validación con Out-of-Bag error
train_scores = []
oob_scores   = []

max_features_range = range(1, X_train.shape[1] + 1, 1)

for max_features in max_features_range:
    modelo = RandomForestRegressor(
                n_estimators = 100,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = max_features,
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )
    modelo.fit(X_train, y_train)
    train_scores.append(modelo.score(X_train, y_train))
    oob_scores.append(modelo.oob_score_)


# In[87]:


# visualizo en un gráfico la evolución del error out-of-bag con diferentes valores de número de predictores
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(max_features_range, train_scores, label="train scores")
ax.plot(max_features_range, oob_scores, label="out-of-bag scores")
ax.plot(max_features_range[np.argmax(oob_scores)], max(oob_scores),
        marker='o', color = "red")
ax.set_ylabel("R^2")
ax.set_xlabel("max_features")
ax.set_title("Evolución del out-of-bag-error vs número de predictores")
plt.legend();
print(f"Valor óptimo de max_features: {max_features_range[np.argmax(oob_scores)]}")


# In[88]:


from sklearn.metrics import mean_squared_error
import numpy as np

train_scores = []
cv_scores    = []

max_features_range = range(1, X_train.shape[1] + 1, 1)


for max_features in max_features_range:
    
    modelo = RandomForestRegressor(
                n_estimators = 100,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = max_features,
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )
    
    # error de train
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X = X_train)
    mse_train = mean_squared_error(y_true=y_train, y_pred=predicciones)
    rmse_train = np.sqrt(mse_train)
    train_scores.append(rmse_train)
    
    # error de validación cruzada
    scores = cross_val_score(
                estimator = modelo,
                X         = X_train,
                y         = y_train,
                scoring   = 'neg_root_mean_squared_error',
                cv        = 5
             )
    cv_scores.append(-1*scores.mean())


# In[144]:


# visualizo en un gráfico la evolución del error de validación cruzada con diferentes valores de número de predictores
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(max_features_range, train_scores, label="train scores")
ax.plot(max_features_range, cv_scores, label="cv scores")
ax.plot(max_features_range[np.argmin(cv_scores)], min(cv_scores),
        marker='o', color = "red", label="min score")
ax.set_ylabel("root_mean_squared_error")
ax.set_xlabel("max_features")
ax.set_title("Evolución del cv-error vs número de predictores")
plt.legend();
print(f"Valor óptimo de max_features: {max_features_range[np.argmin(cv_scores)]}")


# ### Grid Search

# In[91]:


from sklearn.model_selection import ParameterGrid


# In[92]:


# Grid search de combinación de hiperparámetros óptimos
param_grid = ParameterGrid(
                {'n_estimators': [145],
                 'max_features': [1,5,7],
                 'max_depth'   : [None,10, 20]
                }
             )

# bucle para entrenar un modelo con cada combinación de hiperparámetros
resultados = {'params': [], 'oob_r2': []}

for params in param_grid:
    
    modelo = RandomForestRegressor(
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123,
                ** params
             )
    
    modelo.fit(X_train, y_train)
    
    resultados['params'].append(params)
    resultados['oob_r2'].append(modelo.oob_score_)
    print(f"Modelo: {params} ✓")

resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('oob_r2', ascending=False)
resultados.head(4)


# In[145]:


print("Mejores hiperparámetros encontrados (oob-r2)")
print(resultados.iloc[0,0:])


# ## Grid Search basado en validación cruzada

# In[98]:


from multiprocessing import cpu_count
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {'n_estimators': [145],
              'max_features': [1,5, 7],
              'max_depth'   : [None, 3, 10]
             }

grid = GridSearchCV(
            estimator  = RandomForestRegressor(random_state = 123),
            param_grid = param_grid,
            scoring    = 'neg_root_mean_squared_error',
            n_jobs     = cpu_count() - 1,
            cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123), 
            refit      = True,
            verbose    = 0,
            return_train_score = True
       )

grid.fit(X=X_train, y=y_train)

resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)')     .drop(columns = 'params')     .sort_values('mean_test_score', ascending = False)     .head(4)


# In[98]:


# Mejores hiperparámetros encontrados mediante validación cruzada
print("Mejores hiperparámetros encontrados (cv)")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)


# In[102]:


from sklearn.metrics import mean_squared_error
import numpy as np

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# In[103]:


modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X=X_test)
rmse = root_mean_squared_error(y_true=y_test, y_pred=predicciones)
print(f"El error cuadrático medio (RMSE) de test es: {rmse}")


# ## Bayesian search

# In[107]:


get_ipython().system('pip install optuna')
import optuna


# In[108]:


# Bayesian search de combinación de hiperparámetros óptimos
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'max_features': trial.suggest_float('max_features', 0.2, 1.0),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 1),
        'n_jobs': -1,
        'random_state': 4576688,
        
    }

    modelo = RandomForestRegressor(**params)
    cross_val_scores = cross_val_score(
        estimator = modelo,
        X = X_train,
        y = y_train,
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
        scoring = 'neg_root_mean_squared_error',
        n_jobs=-1
    )
    score = np.mean(cross_val_scores) 
    return score

study = optuna.create_study(direction='maximize') # Se maximiza por que el score es negativo
study.optimize(objective, n_trials=30, show_progress_bar=True, timeout=60*5)

print('Mejores hiperparámetros:', study.best_params)
print('Mejor puntuación:', study.best_value)


# In[109]:


modelo_final = RandomForestRegressor(**study.best_params)
modelo_final.fit(X_train, y_train)
predicciones = modelo_final.predict(X=X_test)
rmse = root_mean_squared_error(y_true=y_test, y_pred=predicciones)
print(f"El error (rmse) de test es: {rmse}")


# ### Pregunta 4. ¿Cómo influyen la hora del día y la ubicación en la probabilidad de que un viaje en bicicleta compartida sea corto (menos de 29 minutos) o largo (más de 29 minutos)?

# In[125]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# calculo el promedio de duración de viajes
promedio_trip_duration = combined_df['tripduration'].mean()
print(promedio_trip_duration)

# defino ese valor del promedio como umbral para la clasificación 
umbral_tripduration = promedio_trip_duration

# creo nueva columna en el conjunto de datos de tipo binaria para determinar si un viaje es largo (1) o corto (0)
combined_df['Duración Larga'] = (combined_df['tripduration'] > umbral_tripduration).astype(int)
print(combined_df['Duración Larga'].head(5))

X = combined_df[['start station latitude', 'end station latitude', 'HoradeldíaInicio', 'distance_haversine']]
y = combined_df['Duración Larga']  # Cambiar 'tripduration' por 'Duración Larga'


# In[126]:


# se ha reducido el tamaño del conjunto de entrenamiento para agilizar la carga pues son demasiadas observaciones
X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(X, y, train_size=100000, test_size=0.2, random_state=125)
print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]}")
print(f"Tamaño conjunto test: {X_test.shape[0]}")


# In[129]:


# entreno modelo Random Forest de clasificación  
model = RandomForestClassifier()  
model.fit(X_train_clas, y_train_clas)


# In[130]:


# calculo las probabilidades de las clases positivas
y_prob = model.predict_proba(X_test_clas)[:, 1]


# In[131]:


#calculo el área bajo la curva ROC
roc_auc = roc_auc_score(y_test_clas, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

print("Área bajo la curva ROC:", roc_auc)


# In[132]:


y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print(f"Área bajo la curva ROC: {auc}")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# visualizo la curva ROC
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()


# In[133]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)

# creo la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

