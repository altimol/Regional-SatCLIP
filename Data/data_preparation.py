import os
import pandas as pd
from shapely import wkt
import geopandas as gpd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class DataPrepare:
    """
    DataPrepare implements the full preprocessing pipeline for SatCLIP training data.

    This class is responsible for loading a CSV file containing precomputed image
    embeddings and WKT geometries, cleaning unnecessary columns, parsing the
    geometry column, transforming coordinates to geographic space, and converting
    the resulting embeddings and coordinates into PyTorch tensors.

    Assumptions:
        - A column named "geometry" exists containing WKT POINT strings.
        - Embedding columns are prefixed with "dim".
        - Input geometries are in EPSG:25830 (ETRS89 / UTM zone 30N).
        - Coordinates are reprojected to EPSG:4326 (WGS84 lon/lat).

    Attributes created during processing:
        - data: raw pandas DataFrame
        - cleaned_data: cleaned and transformed GeoDataFrame
        - projected_data: GeoDataFrame in EPSG:4326
        - embeddings_tensor: torch.FloatTensor of embeddings
        - coords_tensor: torch.FloatTensor of longitude and latitude
    """

    def __init__(self, csv_path, verbose=True):
        self.csv_path = csv_path
        self.verbose = verbose

    def csv_info(self):
        self.data = pd.read_csv(self.csv_path)
        if self.verbose:
            print(f"\nLoaded file: {os.path.basename(self.csv_path)}")
            print("Shape:", self.data.shape)
        return self.data

    def csv_clean(self):
        if "Unnamed: 0" in self.data.columns:
            self.cleaned_data = self.data.drop("Unnamed: 0", axis="columns")
        else:
            self.cleaned_data = self.data
        return self.cleaned_data

    def extract_coords(self):
        self.cleaned_data["geometry"] = self.cleaned_data["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(self.cleaned_data, geometry="geometry", crs="EPSG:25830")
        gdf = gdf.to_crs("EPSG:4326")
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        self.cleaned_data = gdf
        self.projected_data = gdf
        return gdf

    def to_tensors(self):
        mask = self.cleaned_data.columns.str.contains("^dim")
        select_embeddings = self.cleaned_data.loc[:, mask]

        if self.verbose:
            print("\n [DEBUG] Column selection summary")
            print(f"Total columns in file: {len(self.cleaned_data.columns)}")
            print(f"Embedding columns detected: {len(select_embeddings.columns)}")
            print(f"Coordinate columns used: ['lon', 'lat']")
            print(f"First 5 embedding columns: {list(select_embeddings.columns[:5])}")
            print(f"Embedding tensor shape: {select_embeddings.shape}")
            print(f"Coordinate tensor shape: {self.cleaned_data[['lon', 'lat']].shape}")
            print("-" * 60)

        self.embeddings_tensor = torch.tensor(
            select_embeddings.values, dtype=torch.float32
        )
        self.coords_tensor = torch.tensor(
            self.cleaned_data[["lon", "lat"]].values, dtype=torch.float32
        )
        return self.coords_tensor, self.embeddings_tensor


class SatCLIPSpainDataset(Dataset):
    """
    PyTorch Dataset for Regional SatCLIP experiments over Spain.

    Each sample consists of:
        - A coordinate tensor of shape (2,) representing (lon, lat)
        - A corresponding image embedding vector

    This dataset is used to train the contrastive alignment between
    geographic coordinates and image embeddings in the SatCLIP framework.
    """

    def __init__(self, coords_tensor, embeddings_tensor):
        self.coords = coords_tensor
        self.embs = embeddings_tensor

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.embs[idx]


class ConstructDataset:
    """
    Helper wrapper for constructing a SatCLIPSpainDataset instance
    from coordinate and embedding tensors.

    This abstraction allows a clean separation between preprocessing
    and dataset instantiation within the training pipeline.
    """

    def __init__(self, coords_tensor, embeddings_tensor):
        self.coords_tensor = coords_tensor
        self.embeddings_tensor = embeddings_tensor

    def build_dataset(self):
        dataset = SatCLIPSpainDataset(self.coords_tensor, self.embeddings_tensor)
        return dataset


class ConstructDataloader:
    """
    Constructs train, validation, and test DataLoaders
    for SatCLIP training and evaluation.

    The dataset is randomly split into:
        - Training set
        - Validation set
        - Test set

    DataLoaders are created with configurable batch size and
    number of worker processes, supporting efficient GPU training.
    """

    def __init__(self, dataset, batch_size=64, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        total = len(self.dataset)
        n_train = int(train_ratio * total)
        n_val = int(val_ratio * total)
        n_test = total - n_train - n_val
        train_ds, val_ds, test_ds = random_split(self.dataset, [n_train, n_val, n_test])
        return train_ds, val_ds, test_ds

    def dataloaders(self):
        train_ds, val_ds, test_ds = self.split_dataset()
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return train_dl, val_dl, test_dl