# Copilot Instructions for AI Coding Agents

## Project Overview
This repository implements neural network models for cat vs. non-cat image classification. It includes both utility code and training/evaluation scripts, with datasets and model artifacts stored locally.

## Key Components
- `application.py`: Main script for model training and evaluation. Entry point for most workflows.
- `dnn_app_utils.py`: Utility functions for deep neural network operations (forward/backward propagation, initialization, etc.).
- `datasets/`: Contains HDF5 files for training and test data (`train_catvnoncat.h5`, `test_catvnoncat.h5`).
- `images/`: Reference images and test samples.
- `trainingDatasetL-LayerNN.pickle`: Serialized model weights for L-layer neural network.
- `pytorch_tests/`: Experimental PyTorch scripts (e.g., `model_eval.py`).

## Developer Workflows
- **Run Training/Evaluation:**
  - Execute `application.py` directly to train or evaluate models.
  - Example: `python application.py`
- **Jupyter Experiments:**
  - Use `test.ipynb` for interactive exploration and testing.
- **PyTorch Experiments:**
  - See `pytorch_tests/model_eval.py` for alternative model evaluation approaches.

## Data Flow
- Data is loaded from `datasets/` (HDF5 format) in `application.py` and processed via utilities in `dnn_app_utils.py`.
- Model weights may be saved/loaded as `.pickle` files.

## Conventions & Patterns
- Utility functions are centralized in `dnn_app_utils.py`.
- Model training and evaluation logic is in `application.py`.
- Use explicit relative imports within the `00_catDog/` directory.
- Datasets are not downloaded automatically; ensure `datasets/` exists with required files.
- No external configuration management—edit scripts directly for parameter changes.

## Integration Points
- No external APIs or services; all data and models are local.
- PyTorch code is isolated in `pytorch_tests/` and does not affect main workflow.

## Examples
- To train and evaluate: `python application.py`
- To run a Jupyter notebook: open `test.ipynb` in VS Code or Jupyter Lab.

## Additional Notes
- No build system or test automation is present.
- For new utilities, add to `dnn_app_utils.py` and import as needed.
- For new experiments, create separate scripts or notebooks to avoid cluttering main files.
