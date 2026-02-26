# Regional-SatCLIP – A Region Specific Geographic Location Encoder 
**MSc Thesis – Geo-Information Science, Wageningen University & Research**

This repository contains the implementation of **Regional-SatCLIP**, developed as part of the MSc Geo-Information Science programme at Wageningen University & Research.

Regional-SatCLIP is built upon SatCLIP (Klemmer, K., Rolf, E., Robinson, C., Mackey, L., & Rußwurm, M. (2024). SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery. arXiv:2311.17179) and retrains the objective on region-specific data to improve representation learning for localized geospatial patterns. It also integrates a Retrieval Augmented Generation (RAG) pipeline, to fuze embedding modalities during inference. The framework supports training, inference, PCA visualization, and retrieval-augmented extensions.

---

## Project Overview

SatCLIP learns joint embeddings of satellite imagery and geographic coordinates through a contrastive objective. In this project, the architecture is initialized locally instead of loading pretrained weights from Hugging Face.

This implementation:

- Retrains SatCLIP on region-specific data  
- Supports configurable location encoder capacity  
- Enables embedding inference  
- Generates PCA visualization maps  
- Integrates a retrieval augmented extension  
- Supports downstream evaluation (code not included)

⚠️ This pipeline is not fully reproducible in a strict end to end manner without further modifications and access to the necessary data. The repository provides an overview of the code used to develop and train Regional SatCLIP, together with the RAG components. Reproduction and execution are possible by meeting the requirements listed below. The downstream evaluation pipeline builds upon the work of a previous MSc student (Levin van Krieken) and was adapted for this thesis. Owing to authorship restrictions, the downstream evaluation code is not distributed within this repository.

---

## Repository Structure
    .
    ├── Data/
    │ ├── data_preparation.py
    │ ├── handle_multiple.py
    │
    ├── Rag/
    │ ├── RAG_retriever.py
    │ ├── inference_with_RAG.py
    │ ├── inference_with_simple_RAG.py
    │
    ├── training.py
    ├── main.py
    ├── inference_embeddings.py
    ├── pca_maps.py
    │
 
    │
    ├── satclip/
    ├── locationencoder/
    ├── utils/


### `data/`
- **data_preparation.py**  
  Converts CSV embedding files into the correct coordinate reference system and transforms them into PyTorch tensors.
- **handle_multiple.py**  
  Loads and processes multiple embedding CSV files for training.

### `training.py`  
Contains the Regional-SatCLIP training logic.

### `main.py`  
Initializes configuration parameters and launches training.

### `inference_embeddings.py`  
Generates embeddings using a trained Regional-SatCLIP model.

### `satclip/`, `locationencoder/`, `utils/`  
Contain the core components of the SatCLIP architecture, including:
- Contrastive objective
- Model definition
- Location encoders
- Utility functions

The architecture is initialized locally rather than loaded from Hugging Face. All the code in these files is adopted from the original SatCLIP repository and was slightly modified to suit this implementation (https://github.com/microsoft/satclip/tree/main)

### `pca_maps.py`  
Generates PCA maps from learned embeddings for visualization.

### `rag/`  
Implements retrieval augmented components:
- RAG retriever module  
- Inference with retrieval  
- Simplified RAG inference  

---

## Requirements

The pipeline can be reproduced and executed, if the following requirements are met. 
### Embedding File Structure

Training requires precomputed embedding CSV files.

Embedding files **must**:

- Contain embedding dimensions as columns, e.g. `dim_0`, `dim_1`, ..., `dim_n`
- Include a `geometry` column
- The `geometry` column must contain **WKT POINT strings**

Example:
    POINT (-3.7038 40.4168)


Each row represents:
- An image embedding vector  
- Its associated geographic coordinate in WKT format  

### Important Notes

- Batch size, capacity, learning rate, and other hyperparameters must be configured manually.
- Downstream evaluation code is not included.
- Some scripts assume specific folder structures and file naming conventions.

---
