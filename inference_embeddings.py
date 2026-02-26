import os
import json
import torch
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt, Point
from satclip.model import SatCLIP  # SatCLIP model class

# config
RESULTS_DIR = "Results"
DATA_DIR = "Data"
OUT_DIR = "results_inference"
os.makedirs(OUT_DIR, exist_ok=True)

APPLY_NORMALIZATION = False

DEFAULT_MODEL_ARGS = dict(
    embed_dim=64,
    image_resolution=224,
    vision_layers="moco_resnet50",
    vision_width=64,
    vision_patch_size=16,
    in_channels=3,
    le_type="grid",  # change based on the training setup
    pe_type="siren", # change based on the training setup
    frequency_num=10,
    max_radius=10,
    min_radius=0.1,
    harmonics_calculation="closed-form",
    legendre_polys=40,
    num_hidden_layers=2,
    capacity=512,
    dropout_rate=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}\n")

# helper functions
def load_coordinates(csv_path: str) -> gpd.GeoDataFrame:
    print(f"Loading coordinates from {csv_path}")
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=None)
    elif {"x", "y"} <= set(df.columns):
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["x"], df["y"])], crs=None)
    else:
        raise ValueError("CSV must contain a 'geometry' column or x/y columns.")

    if gdf.crs is None:
        x_range = gdf.geometry.x.max() - gdf.geometry.x.min()
        y_range = gdf.geometry.y.max() - gdf.geometry.y.min()
        if x_range > 1000 and y_range > 1000:
            gdf.set_crs("EPSG:25830", inplace=True)
        else:
            gdf.set_crs("EPSG:4326", inplace=True)

    print(f"Loaded {len(gdf)} coordinates (CRS: {gdf.crs})")
    return gdf


def load_model_args_from_config(run_folder: str) -> dict:
    args = DEFAULT_MODEL_ARGS.copy()
    cfg_path = os.path.join(run_folder, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

            if "embed_dim" in cfg:
                args["embed_dim"] = cfg["embed_dim"]
            if "capacity" in cfg:
                args["capacity"] = cfg["capacity"]
            if "dropout" in cfg:
                args["dropout"] = cfg["dropout"]
            if "hidden_layers" in cfg:
                args["num_hidden_layers"] = cfg["hidden_layers"]
            if "legendre_polys" in cfg:
                args["legendre_polys"] = cfg["legendre_polys"]
            if "min_radius" in cfg:
                args["min_radius"] = cfg["min_radius"]

            print(
                f"Using model args for run folder {os.path.basename(run_folder)}: "
                f"embed_dim={args['embed_dim']}, "
                f"capacity={args['capacity']}, "
                f"dropout_rate={args['dropout']}, "
                f"legendre_polys={args['legendre_polys']}, "
                f"num_hidden_layers={args['num_hidden_layers']}, "
                f"min_radius={args['min_radius']}"
            )

        except Exception as e:
            print(f"Failed to parse config.json ({e}). Falling back to defaults.")
    else:
        print(f"No config.json found in {run_folder} — using default model args.")
    return args


def load_satclip_state_dict(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        print("Detected Lightning .ckpt file — extracting state_dict.")
        state_dict = checkpoint["state_dict"]
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        return cleaned_state_dict
    else:
        print("Detected regular .pth file.")
        return checkpoint


# load the data
print(f"Scanning for run folders inside: {RESULTS_DIR}")

run_folders = [
    f for f in os.listdir(RESULTS_DIR)
    if os.path.isdir(os.path.join(RESULTS_DIR, f))
]

if not run_folders:
    raise RuntimeError(f"No run folders found in {RESULTS_DIR}")

# Load coordinate CSV once
coord_csv = os.path.join(DATA_DIR, "resnet_moco_20k.csv")
if not os.path.exists(coord_csv):
    raise FileNotFoundError(f"Coordinate CSV not found: {coord_csv}")

gdf_native = load_coordinates(coord_csv)
gdf_4326 = gdf_native.to_crs("EPSG:4326")
gdf_25830 = gdf_native.to_crs("EPSG:25830")

coords_4326 = torch.tensor(
    np.vstack((gdf_4326.geometry.x.values, gdf_4326.geometry.y.values)).T,
    dtype=torch.float64,
    device=device,
)

# initiate processing loop
for run_name in run_folders:
    print(f"Processing run: {run_name}")

    run_path = os.path.join(RESULTS_DIR, run_name)

    # find checkpoint file
    ckpt_files = [
        f for f in os.listdir(run_path)
        if f.endswith((".pth", ".ckpt"))
    ]

    if not ckpt_files:
        print(f"No checkpoint found in {run_name}, skipping.")
        continue

    ckpt_path = os.path.join(run_path, ckpt_files[0])
    print(f"Using checkpoint: {ckpt_path}")

    # load model args from correct run folder
    MODEL_ARGS = load_model_args_from_config(run_path)

    # load model + weights
    model = SatCLIP(**MODEL_ARGS).to(device)
    state_dict = load_satclip_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # encode location embeddings
    print("Encoding location embeddings...")
    with torch.no_grad():
        loc_features = model.encode_location(coords_4326)

    if APPLY_NORMALIZATION:
        norms = torch.linalg.norm(loc_features, dim=1, keepdim=True)
        loc_features = loc_features / (norms + 1e-8)

    loc_np = loc_features.cpu().numpy()
    embed_cols = [f"dim_{i}" for i in range(loc_np.shape[1])]

    out_df = pd.DataFrame(loc_np, columns=embed_cols)
    if "CODMUNI" in gdf_25830.columns:
        out_df["CODMUNI"] = gdf_25830["CODMUNI"].values
    out_df["x"] = gdf_25830.geometry.x
    out_df["y"] = gdf_25830.geometry.y
    out_df["geometry"] = gdf_25830.geometry.to_wkt()

    cols_order = (
        ["CODMUNI"] + embed_cols + ["x", "y", "geometry"]
        if "CODMUNI" in out_df.columns
        else embed_cols + ["x", "y", "geometry"]
    )
    out_df = out_df[cols_order]

    # correct output name per run
    out_path = os.path.join(OUT_DIR, f"inference_{run_name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved embeddings → {out_path}")

print("\nAll runs processed successfully!")
