import os
from Data.data_preparation import DataPrepare, ConstructDataset, ConstructDataloader
from tqdm import tqdm

def handle_multiple_csvs(folder_path, batch_size=64, num_workers=0):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    loaders_dict = {}

    for file_name in tqdm(csv_files, desc="Loading embeddings", unit="file"):
        csv_path = os.path.join(folder_path, file_name)

        # load and prepare data
        dp = DataPrepare(csv_path, verbose=False) # set to True, to print name and structure of embeddings while loading
        dp.csv_info()
        dp.csv_clean()
        dp.extract_coords()
        coords_tensor, emb_tensor = dp.to_tensors()

        # build dataset and dataloaders
        dataset = ConstructDataset(coords_tensor, emb_tensor).build_dataset()
        dataloader_builder = ConstructDataloader(dataset, batch_size, num_workers)
        train_dl, val_dl, test_dl = dataloader_builder.dataloaders()

        loaders_dict[file_name.replace(".csv", "")] = {
            "train": train_dl,
            "val": val_dl,
            "test": test_dl,
            "img_dim": emb_tensor.shape[1],
        }

    print("\n All CSV files have been processed successfully.")
    return loaders_dict
