# Análisis de datos de Airbnb sobre tendencias de vivienda en ciudades importantes

# Importar las bibliotecas necesarias
import pandas as pd                # Para manipulación y análisis de datos
import matplotlib.pyplot as plt     # Para visualización de datos
import seaborn as sns               # Para visualizaciones estadísticas
import geopandas as gpd             # Para trabajar con datos geoespaciales
from datetime import datetime       # Para manejar objetos de fecha y hora

# Configurar el estilo de los gráficos de Seaborn
sns.set(style="whitegrid")

# Cargar los conjuntos de datos para dos ciudades (Nueva York y San Francisco)

# Cargar los datasets para Nueva York
listings_nyc = pd.read_csv('assets/data/listings_nyc.csv.gz', compression='gzip', low_memory=False)
calendar_nyc = pd.read_csv('assets/data/calendar_nyc.csv.gz', compression='gzip', low_memory=False)

# Cargar los datasets para San Francisco
listings_sf = pd.read_csv('assets/data/listings_sf.csv.gz', compression='gzip', low_memory=False)
calendar_sf = pd.read_csv('assets/data/calendar_sf.csv.gz', compression='gzip', low_memory=False)

# Exploración inicial de los datos

# Ver las primeras filas de los listados de Nueva York
print(listings_nyc.head())

# Ver las primeras filas del calendario de Nueva York
print(calendar_nyc.head())

# Ver las primeras filas de los listados de San Francisco
print(listings_sf.head())

# Ver las primeras filas del calendario de San Francisco
print(calendar_sf.head())

# Limpieza y preprocesamiento de datos

# Verificar valores faltantes en listings_nyc
print(listings_nyc.isnull().sum())

# Eliminar filas con valores faltantes críticos en listings_nyc
listings_nyc.dropna(subset=['price', 'availability_365', 'latitude', 'longitude'], inplace=True)

# Repetir el proceso para listings_sf
print(listings_sf.isnull().sum())
listings_sf.dropna(subset=['price', 'availability_365', 'latitude', 'longitude'], inplace=True)

# Conversión de tipos de datos

# Convertir la columna 'price' a tipo numérico en listings_nyc
listings_nyc['price'] = listings_nyc['price'].replace('[\\$,]', '', regex=True).astype(float)

# Repetir para listings_sf
listings_sf['price'] = listings_sf['price'].replace('[\\$,]', '', regex=True).astype(float)

# Convertir 'price' en calendar_nyc y calendar_sf
calendar_nyc['price'] = calendar_nyc['price'].replace('[\\$,]', '', regex=True).astype(float)
calendar_sf['price'] = calendar_sf['price'].replace('[\\$,]', '', regex=True).astype(float)

# Manejo de fechas

# Convertir la columna 'date' a tipo datetime en calendar_nyc y calendar_sf
calendar_nyc['date'] = pd.to_datetime(calendar_nyc['date'])
calendar_sf['date'] = pd.to_datetime(calendar_sf['date'])

# Manejo de disponibilidad

# Convertir 'available' a valores binarios en calendar_nyc ('t' para disponible, 'f' para no disponible)
calendar_nyc['occupied'] = calendar_nyc['available'].apply(lambda x: 0 if x == 't' else 1)

# Repetir para calendar_sf
calendar_sf['occupied'] = calendar_sf['available'].apply(lambda x: 0 if x == 't' else 1)

# Análisis de tendencias de precios en el tiempo

# Agregar columna de mes en calendar_nyc
calendar_nyc['month'] = calendar_nyc['date'].dt.to_period('M')

# Calcular precio promedio y tasa de ocupación por mes en Nueva York
nyc_monthly = calendar_nyc.groupby('month').agg({'price': 'mean', 'occupied': 'mean'}).reset_index()

# Renombrar columnas para mayor claridad
nyc_monthly.rename(columns={'price': 'avg_price', 'occupied': 'occupancy_rate'}, inplace=True)

# Agregar columna de mes en calendar_sf
calendar_sf['month'] = calendar_sf['date'].dt.to_period('M')

# Calcular precio promedio y tasa de ocupación por mes en San Francisco
sf_monthly = calendar_sf.groupby('month').agg({'price': 'mean', 'occupied': 'mean'}).reset_index()

# Renombrar columnas para mayor claridad
sf_monthly.rename(columns={'price': 'avg_price', 'occupied': 'occupancy_rate'}, inplace=True)

# Análisis de correlación entre precios y tasas de ocupación

# Calcular correlación en Nueva York
nyc_correlation = nyc_monthly['avg_price'].corr(nyc_monthly['occupancy_rate'])
print(f"Correlación entre precio y tasa de ocupación en Nueva York: {nyc_correlation}")

# Calcular correlación en San Francisco
sf_correlation = sf_monthly['avg_price'].corr(sf_monthly['occupancy_rate'])
print(f"Correlación entre precio y tasa de ocupación en San Francisco: {sf_correlation}")

# Visualización de tendencias

# Graficar precios promedio mensuales para ambas ciudades
plt.figure(figsize=(12,6))
plt.plot(nyc_monthly['month'].astype(str), nyc_monthly['avg_price'], label='Nueva York')
plt.plot(sf_monthly['month'].astype(str), sf_monthly['avg_price'], label='San Francisco')
plt.xlabel('Mes')
plt.ylabel('Precio Promedio')
plt.title('Tendencia de Precios Promedios Mensuales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Graficar tasas de ocupación promedio mensuales para ambas ciudades
plt.figure(figsize=(12,6))
plt.plot(nyc_monthly['month'].astype(str), nyc_monthly['occupancy_rate'], label='Nueva York')
plt.plot(sf_monthly['month'].astype(str), sf_monthly['occupancy_rate'], label='San Francisco')
plt.xlabel('Mes')
plt.ylabel('Tasa de Ocupación')
plt.title('Tendencia de Tasas de Ocupación Promedio Mensuales')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Visualización geográfica de los listados

# Convertir listings_nyc a GeoDataFrame para visualización geoespacial
geometry_nyc = gpd.points_from_xy(listings_nyc['longitude'], listings_nyc['latitude'])
geo_listings_nyc = gpd.GeoDataFrame(listings_nyc, geometry=geometry_nyc)

# Repetir para listings_sf
geometry_sf = gpd.points_from_xy(listings_sf['longitude'], listings_sf['latitude'])
geo_listings_sf = gpd.GeoDataFrame(listings_sf, geometry=geometry_sf)

# Graficar los listados de Nueva York en un mapa
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

geo_listings_nyc.plot(ax=ax[0], markersize=5, color='blue', alpha=0.5)
ax[0].set_title('Listados de Airbnb en Nueva York')
ax[0].set_xlabel('Longitud')
ax[0].set_ylabel('Latitud')

# Graficar los listados de San Francisco en un mapa
geo_listings_sf.plot(ax=ax[1], markersize=5, color='red', alpha=0.5)
ax[1].set_title('Listados de Airbnb en San Francisco')
ax[1].set_xlabel('Longitud')
ax[1].set_ylabel('Latitud')

plt.tight_layout()
plt.show()

# Cargar los límites administrativos de Nueva York y San Francisco (en formato GeoJSON o shapefile)
nyc_neighborhoods = gpd.read_file('assets/data/neighbourhoods_nyc.geojson')
sf_neighborhoods = gpd.read_file('assets/data/neighbourhoods_sf.geojson')

# Asegurar que ambos GeoDataFrames estén en el mismo CRS
geo_listings_nyc = geo_listings_nyc.set_crs("EPSG:4326", allow_override=True)
nyc_neighborhoods = nyc_neighborhoods.to_crs("EPSG:4326")

geo_listings_sf = geo_listings_sf.set_crs("EPSG:4326", allow_override=True)
sf_neighborhoods = sf_neighborhoods.to_crs("EPSG:4326")

# Realizar la unión espacial usando el nuevo parámetro `predicate`
nyc_listings_with_neighborhoods = gpd.sjoin(geo_listings_nyc, nyc_neighborhoods, how="left", predicate="within")
sf_listings_with_neighborhoods = gpd.sjoin(geo_listings_sf, sf_neighborhoods, how="left", predicate="within")

# Revisar las columnas disponibles después de la unión para confirmar el nombre correcto de la columna de barrios
print("Columnas en nyc_listings_with_neighborhoods:", nyc_listings_with_neighborhoods.columns)
print("Columnas en sf_listings_with_neighborhoods:", sf_listings_with_neighborhoods.columns)

# Calcular la distribución de listados por barrio usando 'neighbourhood_right'
nyc_neighborhood_counts = nyc_listings_with_neighborhoods['neighbourhood_right'].value_counts()
sf_neighborhood_counts = sf_listings_with_neighborhoods['neighbourhood_right'].value_counts()

print("Distribución de listados por barrio en Nueva York:")
print(nyc_neighborhood_counts)

print("\nDistribución de listados por barrio en San Francisco:")
print(sf_neighborhood_counts)


# Análisis de asequibilidad de vivienda

# Calcular el precio mediano en Nueva York
median_price_nyc = listings_nyc['price'].median()
print(f"Precio mediano en Nueva York: ${median_price_nyc}")

# Calcular el precio mediano en San Francisco
median_price_sf = listings_sf['price'].median()
print(f"Precio mediano en San Francisco: ${median_price_sf}")

# Comparación de distribución de precios entre las dos ciudades

# Crear un DataFrame para la comparación de precios
price_comparison = pd.DataFrame({
    'Nueva York': listings_nyc['price'],
    'San Francisco': listings_sf['price']
})

# Graficar un boxplot para comparar la distribución de precios
plt.figure(figsize=(10,6))
sns.boxplot(data=price_comparison)
plt.ylabel('Precio')
plt.title('Comparación de Distribución de Precios')
plt.show()
