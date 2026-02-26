import os
import json
import torch
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt, Point
from satclip.model import SatCLIP
from RAG_retriever2 import RAGRetriever

RESULTS_DIR = "../Results"
DATA_DIR = "../Data"
OUT_DIR = "../results_inference"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}\n")

DEFAULT_MODEL_ARGS = dict(
    embed_dim=1000,
    image_resolution=224,
    vision_layers="moco_resnet50",
    vision_width=64,
    vision_patch_size=16,
    in_channels=3,
    le_type="grid", #changed based on the embedding setup
    pe_type="siren",
    frequency_num=10,
    max_radius=10,
    min_radius=0.1,
    harmonics_calculation="closed-form",
    legendre_polys=40,
    num_hidden_layers=2,
    capacity=512,
    dropout_rate=0.0,
)

# Retrieval configuration
topk = 256
retrieval_mode = "top1"       # "top1" or "topm_mean"
topm = 2                      # only used if topm_mean
normalize_img = True

BATCH_SIZE = 4096

def load_coordinates(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25830")
    elif {"x", "y"} <= set(df.columns):
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["x"], df["y"])], crs="EPSG:25830")
    else:
        raise ValueError("CSV must contain 'geometry' WKT or x,y.")

    return gdf

def load_model_args_from_config(run_folder: str) -> dict:
    args = DEFAULT_MODEL_ARGS.copy()
    cfg_path = os.path.join(run_folder, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if "embed_dim" in cfg:
            args["embed_dim"] = cfg["embed_dim"]
        if "capacity" in cfg:
            args["capacity"] = cfg["capacity"]
        if "dropout" in cfg:
            args["dropout_rate"] = cfg["dropout"]
        if "hidden_layers" in cfg:
            args["num_hidden_layers"] = cfg["hidden_layers"]
        if "legendre_polys" in cfg:
            args["legendre_polys"] = cfg["legendre_polys"]
        if "min_radius" in cfg:
            args["min_radius"] = cfg["min_radius"]
    return args

def load_satclip_state_dict(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        return {k.replace("model.", ""): v for k, v in sd.items()}
    return checkpoint

run_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]
if not run_folders:
    raise RuntimeError(f"No run folders found in {RESULTS_DIR}")

# point set for inference
coord_csv = os.path.join(DATA_DIR, "resnet_moco_20k.csv")
gdf_25830 = load_coordinates(coord_csv)
gdf_4326 = gdf_25830.to_crs("EPSG:4326")

coords_lonlat = np.vstack((gdf_4326.geometry.x.values, gdf_4326.geometry.y.values)).T
coords_t = torch.tensor(coords_lonlat, dtype=torch.float64, device=DEVICE)

for run_name in run_folders:
    print(f"Processing run: {run_name}")

    run_path = os.path.join(RESULTS_DIR, run_name)
    ckpt_files = [f for f in os.listdir(run_path) if f.endswith((".pth", ".ckpt"))]
    if not ckpt_files:
        print(f"No checkpoint found in {run_name}, skipping.")
        continue

    ckpt_path = os.path.join(run_path, ckpt_files[0])

    MODEL_ARGS = load_model_args_from_config(run_path)
    model = SatCLIP(**MODEL_ARGS).to(DEVICE)
    model.load_state_dict(load_satclip_state_dict(ckpt_path), strict=False)
    model.eval()

    retriever = RAGRetriever(
        data_dir=DATA_DIR,
        model_names=[
            "resnet18_moco_20k",
            "resnet_decur_20k",
            "resnet_dino_20k",
            "resnet_moco_20k",
            "vit_dino_20k",
            "vit_moco_20k",
        ],
        satclip_model=model,
        device=DEVICE,
        topk=topk,
        mode=retrieval_mode,
        topm=topm,
        normalize_img=normalize_img,
        embedding_prefix="dim_",
    )

    print("Encoding query locations...")
    with torch.no_grad():
        q_loc = model.encode_location(coords_t).to(torch.float32)
    q_loc = q_loc / (torch.linalg.norm(q_loc, dim=1, keepdim=True) + 1e-8)

    print("Retrieval in batches...")
    retrieved_chunks = []
    with torch.no_grad():
        for start in range(0, q_loc.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, q_loc.shape[0])
            r_img = retriever.retrieve_img(q_loc[start:end], run_name=run_name)
            retrieved_chunks.append(r_img.detach().cpu().numpy())

    r_img_np = np.vstack(retrieved_chunks)
    q_loc_np = q_loc.detach().cpu().numpy()

    # concatenate: [image_features, location_features]
    combined = np.concatenate([r_img_np, q_loc_np], axis=1)

    embed_cols = [f"dim_{i}" for i in range(combined.shape[1])]
    out_df = pd.DataFrame(combined, columns=embed_cols)

    if "CODMUNI" in gdf_25830.columns:
        out_df["CODMUNI"] = gdf_25830["CODMUNI"].values

    out_df["x"] = gdf_25830.geometry.x.values
    out_df["y"] = gdf_25830.geometry.y.values
    out_df["geometry"] = gdf_25830.geometry.to_wkt()

    cols_order = ["CODMUNI"] + embed_cols + ["x", "y", "geometry"] if "CODMUNI" in out_df.columns else embed_cols + ["x", "y", "geometry"]
    out_df = out_df[cols_order]

    out_path = os.path.join(OUT_DIR, f"inference_{run_name}_AlexRAG.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved embeddings to {out_path}")
