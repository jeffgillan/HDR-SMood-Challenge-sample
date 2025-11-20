# Imageomics HDR Scientific Mood Challenge - Sample Code

[![Challenge](https://img.shields.io/badge/Challenge-CodaBench-blue)](https://www.codabench.org/competitions/9854/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD700)](https://huggingface.co/datasets/imageomics/sentinel-beetles)

This repository contains baseline training code and submission templates for the **2025 HDR Scientific Mood (Modeling out of distribution) Challenge: Beetles as Sentinel Taxa**. Use this as a reference for understanding the challenge and developing your own solutions.


<br>
<br>

## Repository Structure

```
.
├── README.md                 # This file
├── LICENSE                   # MIT License
├── pyproject.toml           # Project metadata (Python 3.12+)
├── requirements.txt         # Dependencies
├── uv.lock                  # uv lock file for reproducibility
│
├── baselines/
│   ├── training/            # Training code and trained models
│   │   ├── BioClip2/        # Frozen BioCLIP baseline
│   │   │   ├── train.py     # Training script
│   │   │   ├── model.py     # Model architecture (BioClip2_DeepRegressor)
│   │   │   ├── evaluation.py # Evaluation on validation set
│   │   │   ├── utils.py     # Utilities (data loading, metrics)
│   │   │   └── model.pth    # Trained weights (created by train.py)
│   │   │
│   │   ├── BioClip2-ft/     # Fine-tuned BioCLIP
│   │   │   ├── train.py     # Fine-tuning script
│   │   │   ├── model.py     # Split forward pass implementation
│   │   │   ├── utils.py
│   │   │   └── evaluation.py
│   │   ├── BioClip2-ft-did/ # Fine-tuned + domain ID embeddings
│   │   │   ├── train.py     # Domain-aware training
│   │   │   ├── model.py     # Domain embedding module
│   │   │   ├── utils.py
│   │   │   └── evaluation.py
│   │   └── Dino2/           # DINOv2 baseline
│   │       ├── train.py
│   │       ├── model.py     # Spatial convolution processing
│   │       ├── utils.py
│   │       └── evaluation.py
│   └── submissions/         # Submission-ready models (inference only)
│       ├── BioClip2/
│       │   ├── model.py     # Complete model with Model class
│       │   ├── model.pth    # Trained weights
│       │   └── requirements.txt
│       ├── BioClip2-ft/
│       ├── BioClip2-ft-did/
│       └── Dino2/
```
<br>
<br>

## The Dataset
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD700)](https://huggingface.co/datasets/imageomics/sentinel-beetles)
The Sentinel Beetles Dataset is hosted on HuggingFace and is **publicly accessible** - no authentication required:

- **Dataset**: https://huggingface.co/datasets/imageomics/sentinel-beetles
- **Access**: Downloads automatically on first run of _train.py_

**Optional:** If you encounter HuggingFace API rate limits or need access to private datasets, create a token:
1. Create a HuggingFace account: https://huggingface.co/join
2. Create an access token: https://huggingface.co/settings/tokens
3. Pass it via `--hf_token YOUR_TOKEN` when running scripts (see examples below)

More information on the dataset [here](https://www.codabench.org/competitions/9854/).
<br>
<br>

## The Pretrained Vision Models

**[BioCLIP v2](https://huggingface.co/imageomics/bioclip-2)**: A vision-language model trained on 150M+ biological images paired with scientific names using contrastive learning. It has a Vision Transformer backbone and outputs 768-dimensional embeddings. Trained specifically on biological/natural history data, making it well-suited for specimen images.

**[DINOv2-base](https://huggingface.co/facebook/dinov2-base)**: A self-supervised vision transformer trained on 1.4B general images using self-distillation. Produces rich spatial features (16×16 grid of 768-dim patch tokens) without requiring labeled data.

<br>
<br>

## Installation
If you a running the scripts in a docker container in Cyverse, the sofware is already totally installed. 

If you need install the python libraries, you will do so during the _training_ command. 


<br>
<br>

## Training

If you are running the scripts within a Cyverse container and all the software has already been installed:

```
python HDR-SMood-Challenge-sample/baselines/training/BioClip2/train.py --batch_size 16 --num_workers 4 --epochs 100
```
<br>

If you need to install all the software: 

with `uv` type:
```
uv run python HDR-SMood-Challenge-sample/baselines/training/BioClip2/train.py --batch_size 8 --num_workers 4 --epochs 100
```

### Command flags

| Flag | Default | Used in Code? | Purpose | Common Tweaks |
|------|---------|---------------|---------|---------------|
| `--lr` | `1e-4` | Yes | Adam learning rate for regression head | `5e-5` (slower), `2e-4` (faster) |
| `--batch_size` | `64` | Yes | Batch size for feature extraction & training | Lower for memory (e.g. 8 / 16) |
| `--num_workers` | `4` | Yes | Parallel workers for `DataLoader` | `0` if low shared memory |
| `--epochs` | `500` | Yes | Total training epochs (loop count) | Start with `100–200` |
| `--hf_token` | `None` | Yes | HF access token (private/gated datasets) | Set for private access |


#### `--lr` (Learning Rate)
Controls the step size for Adam updates on `model.regressor.parameters()`. Default `1e-4` is conservative for a small head on frozen encoder features.
- Lower (`5e-5`) → more stable, slower convergence.
- Higher (`2e-4`/`5e-4`) → faster early progress; risk of oscillation or overfitting.

#### `--batch_size`
Number of samples per batch for BOTH:
1. Initial image `DataLoader` used in `extract_bioclip_features`.
2. Secondary `DataLoader` over the in-memory feature tensor dataset.

Effect:
- Larger batch → better GPU utilization, smoother gradients, higher memory.
- Smaller batch → reduced memory footprint, slower per epoch, noisier gradients.

Memory Issues:
- If you encounter bus / shared memory errors, try `--batch_size 8 --num_workers 0`.

#### `--num_workers`
Worker processes for PyTorch `DataLoader` multiprocessing.
- Default `4` leverages parallel decoding / transforms.
- Set to `0` to disable multiprocessing (useful when `/dev/shm` is constrained or seeing `Bus error`).

#### `--epochs`
Total training iterations over the feature dataset. The code saves the best model by average validation R² but **still runs through all epochs**.
- Large value (500) risks overfitting after plateau.
- Suggested start: `100–200` and monitor validation R².
- Consider adding early stopping (patience) for efficiency (not yet implemented).


#### `--hf_token`
Hugging Face token passed to `datasets.load_dataset` for gated/private datasets or to avoid rate limits.
- Default `None` works for fully public datasets.
- Provide via environment variable: `export HF_TOKEN=...` then `--hf_token "$HF_TOKEN"`.

<br>
<br>

## Evaluation
Aftering training, you can locally evaluate your model by running the following:
```
python HDR-SMood-Challenge-sample/baselines/training/BioClip2/evaluation.py --batch_size 16 --num_workers 4
```

with `uv` do:
```
uv run python HDR-SMood-Challenge-sample/baselines/training/BioClip2/evaluation.py --batch_size 16 --num_workers 4
```

<br>

### Command Flags

| Flag          | Default | Purpose                                                                 |
  |---------------|---------|-------------------------------------------------------------------------|
  | --batch_size  | 64      | Batch size for evaluation DataLoader                                    |
  | --num_workers | 4       | Number of parallel workers for data loading (set to 0 if memory issues) |
  | --hf_token    | None    | HuggingFace token (only needed for rate limits or private datasets)     |

<br>
<br>

## Resource Use

Shows GPU compute utilization % and memory
`watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'`

## Python Scripts Explainer
### utils.py 
A toolkit of helper functions that the training and evaluation scripts use repeatedly. Think of it as a shared
   library that prevents code duplication. Instead of writing the same code in multiple places, these common operations are
  centralized here so all the baseline models can use them.

  Key Functions:

  - get_bioclip() and get_DINO(): Downloads and loads the pre-trained vision models from HuggingFace so you can use them for feature
  extraction.
  - extract_bioclip_features() and extract_dino_features(): These run beetle images through the pre-trained vision models (BioCLIP or
  DINOv2) and save the numerical representations (features). This is done once at the start of training and cached, so you don't waste
   time re-computing features every epoch.
  - compile_event_predictions(): Since each drought event has multiple beetle images, this function groups predictions by event ID and
   averages them. For example, if event #42 has 5 beetle photos, it averages the 5 predictions into one final prediction for that
  event.
  - get_collate_fn(): Creates a function that formats batches of data for PyTorch. It stacks the images together and pairs them with
  their target values (the 3 SPEI climate indices).
  - evalute_spei_r2_scores(): Calculates how accurate your predictions are using R² scores (a statistical measure where 1.0 = perfect,
   0.0 = as good as guessing the average).
  - get_training_args(): Reads the command-line arguments (learning rate, batch size, etc.) that you pass when running the training
  script.
  - save_results(): Writes the final evaluation metrics (MAE and R² scores) to a JSON file so you can check how well your model
  performed.

### Model.py
model.py defines the actual neural network architecture - it's the blueprint
  that describes how data flows through the model from beetle image input to
  SPEI prediction output. Each baseline has a different architecture tailored
  to its specific approach (frozen vs fine-tuned, with/without domain info,
  etc.).

  All models follow the same basic pattern: Pre-trained Vision Model (Bioclipv2 or DINOv2) → 
  Regression Head → 3 SPEI Predictions

  The Four Different Architectures:

  1. BioClip2 (BioClip2_DeepRegressor) - Simplest version
  - Takes a beetle image → runs it through frozen BioCLIP (no updates) →
  extracts 768 numbers
  - Passes those 768 numbers through a 4-layer neural network (the
  "regressor"): 768 → 512 → 128 → 32 → 3
  - Outputs 3 predictions (30-day, 1-year, 2-year SPEI)
  - Only the 4-layer network gets trained; BioCLIP stays frozen

  2. BioClip2-ft (BioClip2_DeepFeatureRegressor) - Fine-tuning version
  - Split forward pass: Divides BioCLIP into frozen part + trainable part
  - forward_frozen(): Processes through the first 10 transformer blocks
  (frozen)
  - forward_unfrozen(): Processes through last 2 blocks (trainable) + regressor
  - Has a special get_trainable_parameters() function that sets different 
  learning rates:
    - Last 2 transformer blocks: 0.01× learning rate (gentle updates to
  pre-trained weights)
    - Regressor: 1× learning rate (normal updates)
  - Same 4-layer regressor at the end

  3. BioClip2-ft-did (BioClip2_DeepFeatureRegressorWithDomainID) - Domain-aware
   version
  - Same split architecture as BioClip2-ft
  - Adds domain ID embeddings: Learns a 768-number representation for each
  collection site
  - When predicting, it adds the domain embedding to the image features:
  image_features + domain_embedding
  - This helps the model adapt to different geographic locations
  - If domain is unknown, uses a special "padding" embedding

  4. Dino2 (DINO_DeepRegressor) - Alternative backbone
  - Uses DINOv2 instead of BioCLIP (frozen)
  - Gets spatial patch tokens: 256 patches × 768 features, reshaped to
  768×16×16 grid
  - Adds convolutional layers (tokens_to_linear) to process spatial
  information:
    - Conv layer 1: Reduces 16×16 → 12×12 (still 768 channels)
    - Conv layer 2: Reduces 12×12 → 1×1 and expands to 1024 channels
  - Then same 4-layer regressor (but starts at 1024 instead of 768)

### train.py
train.py is the orchestrator - it's the main script you run to train a model. It
  handles everything from downloading data to saving the final trained model. Think
  of it as the "control center" that brings together all the pieces (data, model,
  training loop, evaluation).

  The Main Workflow (all baselines follow this):

  1. Parse arguments: Get hyperparameters from command line (learning rate, batch
  size, epochs, etc.)
  2. Load dataset: Download Sentinel Beetles from HuggingFace
  3. Load pre-trained model: Get BioCLIP or DINOv2
  4. Create model instance: Build the full architecture (vision model + regressor)
  5. Transform images: Apply preprocessing (resize, normalize, etc.)
  6. Extract & cache features: Run all images through the frozen part of the model
  once and save results (huge speedup!)
  7. Create data loaders: Make new loaders using the cached features
  8. Train: Run the training loop with validation after each epoch
  9. Save best model: Keep the weights that achieved the highest validation R²

  The train() Function:

  - Training loop: For each epoch, process batches, compute loss (MSE),
  backpropagate, update weights
  - Validation loop: After each training epoch, evaluate on validation set (no
  gradient updates)
  - Early stopping: Track best validation R² (average of 3 SPEI targets), save model
  when it improves
  - Logging: Display training/validation loss and R² scores in progress bars

**Notes**
- Downloaded training data is cached in memory and not stored on disk
- your newly trained models are saved as model.pth in their respective directory (`e.g., /baselines/training/BioClip2/model.pth`)

<br>

### Evaluation.py 
evaluation.py is the testing script - you run it after training to see how well
  your model performs. It loads the trained model weights, runs predictions on the
  validation/test dataset, computes performance metrics, and saves the results. This
  is how you measure if your model is any good!

  The Main Workflow:

  1. Parse arguments: Get configuration (batch size, etc.) from command line
  2. Load trained model:
    - Initialize the model architecture
    - Load saved weights from model.pth (saved during training)
  3. Load dataset: Download validation split from HuggingFace
  4. Transform images: Apply same preprocessing as training
  5. Create data loader: Include eventID for aggregation
  6. Evaluate: Run the evaluate() function
  7. Save results: Write metrics to results.json

  The evaluate() Function:

  The heart of the script - processes all test images and computes metrics:

  1. Loop through batches: Process images in batches with torch.inference_mode() (no
  gradient tracking)
  2. Make predictions: Run images through model → get 3 SPEI predictions per image
  3. Compute MAE: Calculate Mean Absolute Error per batch, average across all batches
  4. Collect predictions: Store all predictions, ground truths, and event IDs
  5. Event aggregation: Call compile_event_predictions() to average multiple images
  per event
  6. Compute R² scores: Calculate R² for each SPEI target (30d, 1y, 2y) on
  event-level data
  7. Print results: Display MAE and R² for all 3 targets
  8. Return metrics: Pass back to be saved as JSON

  Output Example:

  test loss MAE SPEI_30d 1.23
  test loss MAE SPEI_1y 0.98
  test loss MAE SPEI_2y 0.87
  test r2 SPEI_30d 0.456
  test r2 SPEI_1y 0.523
  test r2 SPEI_2y 0.589

  Saved to results.json:
  {
    "SPEI_30d": {"MAE": 1.23, "r2": 0.456},
    "SPEI_1y": {"MAE": 0.98, "r2": 0.523},
    "SPEI_2y": {"MAE": 0.87, "r2": 0.589}
  }

<br>
<br>

## Submission

For your repository, you will want to complete the structure information below and add other files (e.g., training code):
```
submission
  <model weights>
  model.py
  requirements.txt
```
We also recommend that you include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.

**Note:** If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-SMood-challenge/blob/main/ingestion_program/whitelist.txt), please check the [issues](https://github.com/Imageomics/HDR-SMood-challenge/issues) on the challenge GitHub to see if someone else has requested it before making your own issue.


---


## Notes on Computing Resource Use
For training a linear prob on BioCLIP, ~3 GB of GPU was used for a batch size of 64.

For fine-tuning the last two layers of BioCLIP with a linear layer, ~6 GB of GPU was used for a batch size of 64



## References

**Vision Models:**
- BioCLIP v2: Gu et al., "Bioclip 2: Emergent properties from scaling hierarchical contrastive learning." *arXiv preprint arXiv:2505.23883* (2025)
- DINOv2: Oquab et al., "Dinov2: Learning robust visual features without supervision." *arXiv preprint arXiv:2304.07193* (2023)

**Challenge:**
- [CodaBench Platform](https://www.codabench.org/competitions/9854/)
- [Sentinel Beetles Dataset](https://huggingface.co/datasets/imageomics/sentinel-beetles)
- [Challenge GitHub](https://github.com/Imageomics/HDR-SMood-challenge)

---

## Citation

If you use this sample code in your research, please cite:

```bibtex
@misc{imageomics2025smood,
  title={HDR Scientific Mood Challenge: Beetles as Sentinel Taxa},
  author={Imageomics},
  year={2025},
  howpublished={\url{https://www.codabench.org/competitions/9854/}}
}
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

