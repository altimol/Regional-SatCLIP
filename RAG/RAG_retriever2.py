import os
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from shapely import wkt


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)


def _read_bank_coords_4326_from_csv(csv_path: str) -> np.ndarray:
    """
    Reads a CSV with a WKT geometry column that is in EPSG:25830 (UTM),
    converts it to EPSG:4326, returns Nx2 [lon, lat] in degrees.
    """
    df = pd.read_csv(csv_path)

    if "geometry" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'geometry' WKT column.")

    geom = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:25830")
    gdf_4326 = gdf.to_crs("EPSG:4326")

    lon = gdf_4326.geometry.x.to_numpy(dtype=np.float64)
    lat = gdf_4326.geometry.y.to_numpy(dtype=np.float64)
    return np.stack([lon, lat], axis=1)


def _read_embeddings_from_csv(csv_path: str, prefix: str = "dim_") -> np.ndarray:
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    if len(emb_cols) == 0:
        raise ValueError(f"No embedding columns starting with '{prefix}' found in {csv_path}")
    emb = df[emb_cols].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(emb).all():
        raise ValueError(f"Non-finite values in embeddings: {csv_path}")
    return emb


class RAGRetriever:
    """
    RAG retriever but with retrieval (no softmax):

    Bank per model:
      db_loc: SatCLIP location embeddings for bank points (EPSG:4326 input), normalized
      db_img: raw image embeddings from CSV (optionally normalized)

    Query:
      q_loc -> topk by cosine similarity vs db_loc -> select top1 or mean topm
      returns retrieved image embedding
    """

    def __init__(
        self,
        data_dir: str,
        model_names: list[str],
        satclip_model,
        device,
        topk: int = 128,
        mode: str = "top1",          # "top1" or "topm_mean"
        topm: int = 2,               # only for topm_mean
        normalize_img: bool = False,
        embedding_prefix: str = "dim_",
    ):
        self.data_dir = data_dir
        self.model_names = model_names
        self.device = device
        self.topk = int(topk)
        self.mode = mode.lower()
        self.topm = int(topm)
        self.normalize_img = bool(normalize_img)
        self.embedding_prefix = embedding_prefix

        self.banks = {}
        self._build_banks(satclip_model)

    def _build_banks(self, satclip_model):
        satclip_model.eval()
        print("Building retrieval banks")

        for name in self.model_names:
            csv_path = os.path.join(self.data_dir, f"{name}.csv")
            if not os.path.exists(csv_path):
                print(f"CSV not found for {name}, skipping.")
                continue

            coords_4326 = _read_bank_coords_4326_from_csv(csv_path)  # lon,lat degrees
            img_emb = _read_embeddings_from_csv(csv_path, prefix=self.embedding_prefix)

            coords_t = torch.tensor(coords_4326, device=self.device, dtype=torch.float64)
            with torch.no_grad():
                db_loc = satclip_model.encode_location(coords_t).to(torch.float32)
            db_loc = _l2_normalize(db_loc, dim=1)

            db_img = torch.tensor(img_emb, device=self.device, dtype=torch.float32)
            if self.normalize_img:
                db_img = _l2_normalize(db_img, dim=1)

            self.banks[name] = {"db_loc": db_loc, "db_img": db_img}
            print(f"Bank ready for {name} | N={db_loc.shape[0]} | Dloc={db_loc.shape[1]} | Dimg={db_img.shape[1]}")

    def select_bank(self, run_name: str):
        for name in self.model_names:
            if name in run_name:
                return name, self.banks[name]
        raise ValueError(f"No bank matches run_name {run_name}")

    @torch.no_grad()
    def retrieve_img(self, q_loc: torch.Tensor, run_name: str) -> torch.Tensor:
        """
        q_loc: (B, Dloc) float32
        returns: (B, Dimg) float32
        """
        _, bank = self.select_bank(run_name)
        db_loc = bank["db_loc"]
        db_img = bank["db_img"]

        q = _l2_normalize(q_loc, dim=1)
        sims = q @ db_loc.T  # (B, N)

        k = min(self.topk, sims.shape[1])
        vals, idx = torch.topk(sims, k=k, dim=1, largest=True)

        if self.mode == "top1":
            picked = db_img[idx[:, 0]]  # (B, Dimg)
            return picked

        if self.mode == "topm_mean":
            m = max(1, min(self.topm, k))
            gathered = db_img[idx[:, :m]]  # (B, m, Dimg)
            return gathered.mean(dim=1)

        raise ValueError(f"Unknown mode: {self.mode}")
