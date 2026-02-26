import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import math

from satclip.loss import SatCLIPLoss
from satclip.model import SatCLIP

import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")


class SatCLIPTrainer:
    """
    SatCLIPTrainer implements the full training pipeline for aligning image
    embeddings with geographic coordinates using the SatCLIP contrastive objective.

    The trainer is designed for regional SatCLIP experiments where:
        - A pretrained visual backbone is frozen
        - The image projection MLP and location encoder are trained
        - A contrastive InfoNCE-style loss (SatCLIPLoss) is used
        - Cosine learning rate scheduling with warmup is applied
        - Retrieval accuracy is monitored during training

    Core responsibilities:
        1. Initialize and configure the SatCLIP model
        2. Freeze and unfreeze appropriate components
        3. Run training, validation, and test loops
        4. Save the best model checkpoint based on validation loss
        5. Generate training/validation loss curves
        6. Optionally log metrics to Weights & Biases (opt-in only)

    Weights & Biases behavior:
        - Disabled by default
        - Enabled only if config["use_wandb"] = True
        - If wandb is not installed, logging is automatically disabled
        - The script will never prompt for authentication unless explicitly enabled

    Parameters
    ----------
    data_folder : str
        Path to the dataset directory.

    result_folder : str
        Directory where model checkpoints and training plots are saved.

    config : dict
        Dictionary containing hyperparameters and model configuration.
        Expected keys typically include:

        Optimization:
            - learning_rate
            - weight_decay
            - grad_clipping
            - num_epochs
            - warmup_value
            - min_lr

        Location encoder:
            - hidden_layers
            - capacity
            - dropout
            - max_radius
            - min_radius
            - harmonics_calculation
            - legendre_polys

        Optional:
            - use_wandb (bool)
            - wandb_project (str)
            - wandb_name (str)

    Returns
    -------
    float
        Final test retrieval accuracy from train_embedding().
    """
    def __init__(self, data_folder, result_folder, config):
        self.data_folder = data_folder
        self.config = config
        self.result_folder = result_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Opt-in W&B, default disabled
        self.use_wandb = bool(self.config.get("use_wandb", False))

        # If wandb is not installed, force disable
        if self.use_wandb and wandb is None:
            print("[INFO] use_wandb=True but wandb is not installed. Disabling wandb.")
            self.use_wandb = False

        # Make sure wandb never prompts or logs unless explicitly enabled
        # If you want wandb, set config["use_wandb"]=True.
        if not self.use_wandb:
            os.environ.setdefault("WANDB_MODE", "disabled")

        self._wandb_inited = False

    def _wandb_init(self, model_name: str):
        if not self.use_wandb:
            return
        if wandb is None:
            return

        # You can optionally allow passing these via config
        project = self.config.get("wandb_project", "satclip-tracker")
        run_name = self.config.get("wandb_name", f"{model_name}_run")

        wandb.init(
            project=project,
            name=run_name,
            config=self.config,
        )

        # Extra fields you were updating
        wandb.config.update(
            {
                "harmonics_calculation": self.config.get("harmonics_calculation"),
                "min_radius": self.config.get("min_radius"),
                "max_radius": self.config.get("max_radius"),
                "legendre_polys": self.config.get("legendre_polys"),
            },
            allow_val_change=True,
        )

        self._wandb_inited = True

    def _wandb_log(self, data: dict):
        if not self.use_wandb:
            return
        if wandb is None:
            return
        if not self._wandb_inited:
            return
        wandb.log(data)

    def _wandb_finish(self):
        if not self.use_wandb:
            return
        if wandb is None:
            return
        if not self._wandb_inited:
            return
        wandb.finish()
        self._wandb_inited = False

    def compute_retrieval_accuracy(self, logits_img, logits_coord):
        sim_matrix = logits_img @ logits_coord.T  # (B, B)
        preds = sim_matrix.argmax(dim=1)
        correct = torch.sum(
            preds == torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        )
        return (correct.float() / sim_matrix.size(0)).item()

    def train_embedding(self, model_name, train_dl, val_dl, test_dl, img_dim):
        print("\nInitializing training loop...")

        # Initialize SatCLIP model
        model = SatCLIP(
            embed_dim=img_dim,
            image_resolution=224,
            vision_layers="moco_resnet50",
            vision_width=64,
            vision_patch_size=16,
            in_channels=3,
            le_type="grid",  # Positional encoding
            pe_type="siren",  # Neural network
            frequency_num=10,
            max_radius=self.config["max_radius"],
            min_radius=self.config["min_radius"],
            harmonics_calculation=self.config["harmonics_calculation"],
            legendre_polys=self.config["legendre_polys"],
            num_hidden_layers=self.config["hidden_layers"],
            capacity=self.config["capacity"],
            dropout_rate=self.config["dropout"],
        ).to(self.device)

        # Freeze components
        for p in model.visual.parameters():
            p.requires_grad = False
        for p in model.image_mlp.parameters():
            p.requires_grad = True
        for p in model.location.parameters():
            p.requires_grad = True

        # Optimizer (AdamW with your config weight_decay)
        loss_fn = SatCLIPLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Cosine LR scheduler with warmup
        num_epochs = self.config["num_epochs"]
        warmup_epochs = int(self.config["warmup_value"] * num_epochs)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            if num_epochs == warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            min_lr_factor = self.config["min_lr"]
            return min_lr_factor + 0.5 * (1 - min_lr_factor) * (
                1 + math.cos(math.pi * progress)
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float("inf")
        save_dir = self.result_folder
        os.makedirs(save_dir, exist_ok=True)

        # wandb setup (opt-in)
        self._wandb_init(model_name)

        train_losses, val_losses = [], []

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_acc = 0.0, 0.0

            for coords, image_embs in tqdm(
                train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
            ):
                coords, image_embs = coords.to(self.device), image_embs.to(self.device)

                logits_img, logits_coord = model(image_embs, coords)
                loss = loss_fn(logits_img, logits_coord)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config["grad_clipping"]
                )
                optimizer.step()

                train_loss += loss.item()
                train_acc += self.compute_retrieval_accuracy(logits_img, logits_coord)

            train_loss /= len(train_dl)
            train_acc /= len(train_dl)

            # Validation
            model.eval()
            val_loss, val_acc = 0.0, 0.0
            with torch.no_grad():
                for coords, image_embs in val_dl:
                    coords, image_embs = coords.to(self.device), image_embs.to(
                        self.device
                    )
                    logits_img, logits_coord = model(image_embs, coords)
                    val_loss += loss_fn(logits_img, logits_coord).item()
                    val_acc += self.compute_retrieval_accuracy(logits_img, logits_coord)

            val_loss /= len(val_dl)
            val_acc /= len(val_dl)

            # Scheduler step
            scheduler.step()

            # Debug info
            with torch.no_grad():
                coords_sample, img_sample = next(iter(val_dl))
                coords_sample, img_sample = coords_sample.to(
                    self.device
                ), img_sample.to(self.device)

                logits_img, logits_coord = model(img_sample, coords_sample)
                img_emb = model.encode_image(img_sample)
                loc_emb = model.encode_location(coords_sample)

                cos_sim = F.cosine_similarity(img_emb, loc_emb, dim=1)
                mean_cos = cos_sim.mean().item()
                med_cos = cos_sim.median().item()

                raw_logit_scale = model.logit_scale.exp().item()

                print(
                    f"[DEBUG] LR: {scheduler.get_last_lr()[0]:.6f}, "
                    f"logit_scale_raw: {raw_logit_scale:.2f}, "
                    f"Mean cos: {mean_cos:.4f}, "
                    f"Median cos: {med_cos:.4f}"
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1:02d} | LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, f"best_{model_name}.pth")
                torch.save(model.state_dict(), save_path)
                print(
                    f"Saved new best model at epoch {epoch + 1} with val loss {val_loss:.4f}"
                )

            # wandb logging (opt-in)
            self._wandb_log(
                {
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            )

        # Plot loss curves
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Training loss", color="tab:blue", linewidth=2)
        plt.plot(
            epochs,
            val_losses,
            label="Validation loss",
            color="tab:orange",
            linestyle="--",
            linewidth=2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SatCLIP training, {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{model_name}_loss_curve.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved loss curve: {plot_path}")

        # Final test evaluation
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for coords, image_embs in test_dl:
                coords, image_embs = coords.to(self.device), image_embs.to(self.device)
                logits_img, logits_coord = model(image_embs, coords)
                test_acc += self.compute_retrieval_accuracy(logits_img, logits_coord)
        test_acc /= len(test_dl)

        print(f"\nFinal Test Retrieval Accuracy for {model_name}: {test_acc:.4f}\n")
        self._wandb_log({"test_acc": test_acc})
        self._wandb_finish()

        return test_acc