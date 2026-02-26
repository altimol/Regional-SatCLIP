import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
from sklearn.decomposition import PCA
from scipy.interpolate import Rbf

INPUT_DIR = "results_inference"
SPAIN_SHP = "Data/study_area/ne_10m_admin_0_countries.shp"

GEOM_COL = "geometry"
EMBED_PREFIX = "dim_"

plot_boundaries = False

# helper for geometry loading
def safe_load_geom(x):
    if isinstance(x, str):
        return wkt.loads(x.strip())
    if isinstance(x, Point):
        return x
    return None

# load Spain polygon once
world = gpd.read_file(SPAIN_SHP)

if world.crs is None:
    world = world.set_crs(4326)

# CRS of the embeddings will define the target CRS, so reproject later
spain_raw = world[world["ADMIN"] == "Spain"]

# loop through all CSV files
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for csv_name in csv_files:

    csv_path = os.path.join(INPUT_DIR, csv_name)
    base = os.path.splitext(csv_name)[0]
    output_fig = f"pca_map_{base}.png"

    print(f"Processing: {csv_name}")


    # load embeddings

    df = pd.read_csv(csv_path)
    df[GEOM_COL] = df[GEOM_COL].apply(safe_load_geom)
    df = df.dropna(subset=[GEOM_COL])

    gdf = gpd.GeoDataFrame(df, geometry=GEOM_COL)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    if gdf.crs is None:
        gdf = gdf.set_crs(25830)

    # Reproject Spain to match
    spain = spain_raw.to_crs(gdf.crs)

    # Union and keep mainland only
    spain_poly = spain.geometry.union_all()
    if spain_poly.geom_type == "MultiPolygon":
        spain_poly = max(spain_poly.geoms, key=lambda g: g.area)

    xmin, xmax = gdf["x"].min(), gdf["x"].max()
    ymin, ymax = gdf["y"].min(), gdf["y"].max()


    # PCA
    embed_cols = [c for c in gdf.columns if c.startswith(EMBED_PREFIX)]
    X = gdf[embed_cols].values

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)
    expl_var = pca.explained_variance_ratio_.sum()

    def rescale(v):
        return (v - v.min()) / (v.max() - v.min())

    R = rescale(pcs[:, 0])
    G = rescale(pcs[:, 1])
    B = rescale(pcs[:, 2])

    # create grid inside Spain
    res = 400
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    grid_x, grid_y = np.meshgrid(xs, ys)

    mask = np.array([
        spain_poly.contains(Point(px, py))
        for px, py in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)

    # RBF interpolation
    px = gdf["x"].values
    py = gdf["y"].values

    rbf_r = Rbf(px, py, R, function="linear")
    rbf_g = Rbf(px, py, G, function="linear")
    rbf_b = Rbf(px, py, B, function="linear")

    R_grid = rbf_r(grid_x, grid_y)
    G_grid = rbf_g(grid_x, grid_y)
    B_grid = rbf_b(grid_x, grid_y)

    rgb = np.stack([R_grid, G_grid, B_grid], axis=2)
    rgb = np.clip(rgb, 0, 1)

    rgb[~mask] = np.nan

    # plot with transparent background and proper alpha mask
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Create alpha mask
    alpha = np.ones(mask.shape)
    alpha[~mask] = 0

    rgba = np.dstack([rgb, alpha])

    ax.imshow(
        rgba,
        extent=(xmin, xmax, ymin, ymax),
        origin="lower"
    )

    plt.savefig(
        output_fig,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True
    )
    plt.close()

    print("Saved:", output_fig)

