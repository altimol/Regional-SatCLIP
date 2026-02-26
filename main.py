from training import SatCLIPTrainer
import yaml, itertools, json, os
import torch
from Data.handle_multiple_csv import handle_multiple_csvs

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available! Using GPU: {device_name}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA is NOT available, running on CPU")

# load config
with open("training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

keys_list = [
    "capacity",
    "learning_rate",
    "weight_decay",
    "num_epochs",
    "dropout",
    "min_lr",
    "warmup_value",
    "hidden_layers",
    "grad_clipping",
    "harmonics_calculation",
    "min_radius",
    "max_radius",
    "legendre_polys"
]

# prepare all config combinations
configs_to_run = []

values_list = list(itertools.product(
    config["capacity"],
    config["learning_rate"],
    config["weight_decay"],
    config["num_epochs"],
    config["dropout"],
    config["min_lr"],
    config["warmup_value"],
    config["hidden_layers"],
    config["grad_clipping"],
    config["harmonics_calculation"],
    config["min_radius"],
    config["max_radius"],
    config["legendre_polys"]
))

for values in values_list:
    new_config = dict(zip(keys_list, values))
    new_config["batch_size"] = config["batch_size"][0]
    new_config["capacity"] = config["capacity"][0]
    configs_to_run.append(new_config)

# load all CSV data once
all_loaders = handle_multiple_csvs("Data/", batch_size=config["batch_size"][0])

# select the embeddings which SatCLIP will be trained on, do not include .csv at the end
selected_embeddings = [
    "vit_dino_20k"
]

results = {}

for name in selected_embeddings:

    print(f"\nStarting embedding: {name}")

    if name not in all_loaders:
        print(f"No loader found for {name}, skipping")
        continue

    loaders = all_loaders[name]
    train_dl = loaders["train"]
    val_dl = loaders["val"]
    test_dl = loaders["test"]
    img_dim = loaders["img_dim"]

    # loop through all config combinations
    for i, new_config in enumerate(configs_to_run, start=1):

        # each embedding gets its own independent folder structure
        result_folder = os.path.join("Results", f"{name}_{i}")

        if os.path.exists(result_folder):
            print(f"Skipping run {i} for {name}, folder already exists")
            continue

        os.makedirs(result_folder, exist_ok=True)

        # save config
        config_path = os.path.join(result_folder, "config.json")
        with open(config_path, "w") as f:
            json.dump(new_config, f, indent=4)

        # run training
        trainer = SatCLIPTrainer(
            data_folder="Data/",
            result_folder=result_folder,
            config=new_config
        )

        test_acc = trainer.train_embedding(
            model_name=name,
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            img_dim=img_dim
        )

        # store results
        if name not in results:
            results[name] = []
        results[name].append(test_acc)

print("\nFinished all embeddings and config sweeps")
