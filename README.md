# Image Captioning using Knowledge Distillation and Early Exit

## Project structure

```text
ImageCap/
├── data/                  # raw and processed datasets (not committed)
├── artifacts/             # checkpoints and generated files (not committed)
├── notebooks/
├── src/image_captioning/
│   ├── data/              # Flickr8k readers and repositories
│   ├── preprocessing/     # caption normalization and tokenization
│   ├── models/            # VGG16-LSTM and Transformer/KD models
│   ├── training/          # batch generators and training services
│   └── evaluation/        # BLEU and other evaluation metrics
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- GPU with CUDA recommended for training
- Flickr8k dataset
- VS Code with the Jupyter extension, or Google Colab/Kaggle Notebook

## Installation on Windows

Open PowerShell at the project root:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks script activation, run the following once for the current user, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In VS Code, select `.venv` as the Python and Jupyter kernel. Verify the package installation:

```powershell
python -c "from image_captioning.config import CaptioningConfig; print('image_captioning is ready')"
```

## Prepare Flickr8k

Place the extracted dataset in this layout:

```text
data/raw/flickr8k/
├── captions.txt
└── Images/
	├── 1000268201_693b08cb0e.jpg
	└── ...
```

Do not commit images, credentials, checkpoints, or generated pickle files. The repository expects these files under `data/` and `artifacts/` locally.

## Run the notebooks

The notebooks are experiment entry points and should be run in this order:

1. Open [notebooks/Img_caption.ipynb](notebooks/Img_caption.ipynb). Run all cells to extract VGG16 features, train the LSTM captioner, and save the teacher model under the configured working/artifact directory.
2. Open [notebooks/Img_cap_KD_EE.ipynb](notebooks/Img_cap_KD_EE.ipynb). Update its teacher checkpoint and dataset paths, then run all cells to train/evaluate the KD early-exit heads.

The original notebooks were written for hosted runtimes. Before running locally, replace these hosted paths:

| Notebook | Hosted paths to replace | Local paths |
| --- | --- | --- |
| `Img_caption.ipynb` | `/content/sample_data/img_cap/` | `data/raw/flickr8k/` |
| `Img_caption.ipynb` | `/content/sample_data/working/` | `artifacts/baseline/` |
| `Img_cap_KD_EE.ipynb` | `/kaggle/input/datasets/...` | `data/raw/flickr8k/` |
| `Img_cap_KD_EE.ipynb` | `/kaggle/working/` | `artifacts/kd_ee/` |

For Colab or Kaggle, upload/mount the dataset and model checkpoints first, then keep the hosted paths used by that runtime. Training both models is resource-intensive; use a CUDA kernel and run one notebook at a time.

## Use the OOP package

Reusable code belongs in `src/image_captioning/`. New notebook cells should import it instead of defining reusable functions inline:

```python
from pathlib import Path

from image_captioning.config import DatasetConfig
from image_captioning.data import Flickr8kDatasetBuilder, PickleDatasetRepository

config = DatasetConfig(root=Path("data/raw/flickr8k"))
builder = Flickr8kDatasetBuilder(config.image_path, config.train_ratio, config.seed)
train, valid = builder.build(config.captions_path)
PickleDatasetRepository.save(train, Path("data/processed/train.pkl"))
PickleDatasetRepository.save(valid, Path("data/processed/valid.pkl"))
```

## Run tests

From the project root:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The tests cover caption cleaning, deterministic Flickr8k splitting, and dataset persistence without loading a deep-learning model.

Dataset images, checkpoints, Kaggle credentials, and notebook outputs should stay outside the source package.
