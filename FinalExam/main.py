import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
import xarray as xr

# Coordenadas de Rionegro
RIONEGRO_BOUNDS = {
    "min_lat": 6.067,
    "max_lat": 6.250,
    "min_lon": -75.450,
    "max_lon": -75.350
}

# Cargar los datos de hotspots
df_fire_archive = pd.read_csv("./data/viirs-snpp_2021_Colombia.csv")
# Cargar los datos de AOD
aod_data = xr.open_dataset("./data/AOD_Regional.nc")

# Filtrar los datos
def filter_geographical_data(df, bounds):
    return df[
        (df['latitude'] >= bounds['min_lat']) &
        (df['latitude'] <= bounds['max_lat']) &
        (df['longitude'] >= bounds['min_lon']) &
        (df['longitude'] <= bounds['max_lon'])
    ]


def visualice_hotspots(df_fire_rionegro):
    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_fire_rionegro,
        geometry=gpd.points_from_xy(df_fire_rionegro.longitude, df_fire_rionegro.latitude),
        crs="EPSG:4326"
    )

    # Gráfico 1: Mapa de Colombia
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-80, -65, -5, 15], crs=ccrs.PlateCarree())  # Límites para Colombia

    # Agregar características del mapa base
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.LAND, edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.LAKES, edgecolor="black", alpha=0.3)

    # Añadir gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Visualización de datos en Rionegro
    scatter = ax.scatter(
        df_fire_rionegro["longitude"], df_fire_rionegro["latitude"],
        c=df_fire_rionegro["bright_ti4"],
        s=df_fire_rionegro["frp"] * 20,
        cmap="Reds", edgecolor="k", alpha=0.7, transform=ccrs.PlateCarree(),
        label="Hotspots"
    )

    # Agregar límites de Colombia
    world = gpd.read_file("./data/ne_110m_admin_0_countries.shp")
    colombia = world[world['ADMIN'] == 'Colombia']
    colombia.plot(ax=ax, color='none', edgecolor='blue', linewidth=1, transform=ccrs.PlateCarree())

    # Barra de colores
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Brightness")

    # Títulos y etiquetas
    ax.set_title("Hotspots en Rionegro y Mapa de Colombia", fontsize=14)

    # Mostrar el primer gráfico
    plt.show()

    # Gráfico 2: Mapa centrado en Rionegro
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-75.450, -75.350, 6.067, 6.250], crs=ccrs.PlateCarree())

    # Agregar características del mapa base
    ax.add_feature(cfeature.LAND, edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.LAKES, edgecolor="black", alpha=0.3)

    # Añadir gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # Visualización de datos en Rionegro
    scatter = ax.scatter(
        df_fire_rionegro["longitude"], df_fire_rionegro["latitude"],
        c=df_fire_rionegro["brightness"],
        s=df_fire_rionegro["frp"] * 20,
        cmap="Reds", edgecolor="k", alpha=0.7, transform=ccrs.PlateCarree(),
        label="Hotspots"
    )

    # Barra de colores
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Brightness")

    # Títulos y etiquetas
    ax.set_title("Hotspots en Rionegro", fontsize=17)

    # Mostrar el segundo gráfico
    plt.show()

    # Gráfico 3: Mapa detallado con fondo de OpenStreetMap
    gdf = gdf.to_crs(epsg=3857)  # Cambiar CRS para usar contextily
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        ax=ax,
        column="brightness",
        cmap="Reds",
        legend=True,
        markersize=gdf["frp"] * 20,
        edgecolor="k",
        alpha=0.7
    )

    # Agregar fondo de mapa
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Títulos y etiquetas
    ax.set_title("Hotspots en Rionegro con Fondo de Mapa", fontsize=17)

    # Mostrar el tercer gráfico
    plt.show()

def visualice_AOD(aod_data):
    """
    Visualiza el valor de AOD (Aerosol Optical Depth) en Rionegro con un fondo de mapa detallado.
    """
    import contextily as ctx

    # Seleccionar la variable AOD y una fecha específica
    aod = aod_data["aod550"]
    aod_date = aod.sel(time="2021-03-18")  # Cambia la fecha según sea necesario

    # Filtrar datos de AOD para Rionegro
    aod_rionegro = aod_date.sel(
        latitude=slice(RIONEGRO_BOUNDS["max_lat"], RIONEGRO_BOUNDS["min_lat"]),
        longitude=slice(RIONEGRO_BOUNDS["min_lon"], RIONEGRO_BOUNDS["max_lon"])
    )

    # Obtener el valor de AOD
    aod_value = aod_rionegro.values[0, 0]  # Extrae el único valor disponible
    lon = aod_rionegro.longitude.values[0]
    lat = aod_rionegro.latitude.values[0]

    # Crear un GeoDataFrame para el punto de AOD
    gdf_aod = gpd.GeoDataFrame(
        {"AOD": [aod_value]},
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326"
    )

    # Convertir el CRS a EPSG:3857 para usar contextily
    gdf_aod = gdf_aod.to_crs(epsg=3857)

    # Visualizar el mapa con el fondo de OpenStreetMap
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf_aod.plot(
        ax=ax,
        color="red", markersize=200, alpha=0.7,
        label=f"AOD: {aod_value:.3f}"
    )

    # Agregar el fondo del mapa
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Configurar el título y los límites del mapa
    ax.set_title("AOD en Rionegro (2021-03-18)", fontsize=16)
    ax.set_xlim(gdf_aod.total_bounds[[0, 2]])  # Límites para la longitud
    ax.set_ylim(gdf_aod.total_bounds[[1, 3]])  # Límites para la latitud

    # Agregar leyenda
    plt.legend(loc="upper left", fontsize=10)

    # Mostrar el gráfico
    plt.show()



def main():

    # Filtrar los datos de hotsports en Rionegro
    df_fire_rionegro = filter_geographical_data(df_fire_archive, RIONEGRO_BOUNDS)
    pd.set_option('display.max_columns', None)
    print(df_fire_rionegro.head())
    df_fire_rionegro = df_fire_rionegro[df_fire_rionegro["acq_date"] == "2021-03-18"]
    print(df_fire_rionegro.head())

    # Visualizar los hotspots en Rionegro
    visualice_hotspots(df_fire_rionegro)

    # Visualizar los datos de AOD
    visualice_AOD(aod_data)



if __name__ == "__main__":
    main()