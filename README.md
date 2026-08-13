# Ontology Robustness Evaluation

A tool for evaluating the robustness of a computer vision ontology by computing inter-annotator agreement metrics on YOLO-format annotations.

## Overview

When building a computer vision dataset, ensuring that annotators agree on how to apply ontology classes is critical. This tool compares annotations from multiple annotators on the same images and produces:

- **Mean IoU matrix** — how consistently annotators place bounding boxes
- **Cohen Kappa matrix** — how consistently annotators assign class labels
- **IoU distribution** — per-pair boxplots of IoU values
- **Confusion matrices** — per-pair class agreement/disagreement
- **Synthetic report** — identifies divergent annotator pairs and problematic classes

## Repository Structure

```
eval_ontologie/
├── data/
│   ├── annotator_1/
│   │   ├── labels.txt
│   │   └── labels/
│   │       ├── image1.txt
│   │       └── ...
│   ├── annotator_2/
│   │   ├── labels.txt
│   │   └── labels/
│   │       ├── image1.txt
│   │       └── ...
│   └── ...
├── src/
│   ├── evaluate_ontology.ipynb
│   └── evaluate_ontology.py
├── output/
├── requirements.txt
└── README.md
```

### Annotation format

Each annotation file follows the standard YOLO format — one line per bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized between 0 and 1.

### Labels file format

Each annotator folder must contain a `labels.txt` file mapping class IDs to class names:

```
'0': 'class_name_a'
'1': 'class_name_b'
'2': 'class_name_c'
```

> **Note:** Class IDs do not need to be consistent across annotators — the tool handles harmonization automatically by matching class names.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chaouabti/ontology-robustness-eval.git
cd eval_ontologie
pip install -r requirements.txt
```

## Usage

### Script

```bash
# Minimal — figures displayed only
python src/evaluate_ontology.py --eval_folder data/

# Full — figures saved to output folder
python src/evaluate_ontology.py \
    --eval_folder data/ \
    --output_folder output/ \
    --iou_threshold 0.5 \
    --kappa_threshold 0.6
```

#### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--eval_folder` | str | required | Path to the folder containing one subfolder per annotator |
| `--output_folder` | str | None | Path to save figures. If not provided, figures are only displayed |
| `--iou_threshold` | float | 0.5 | Minimum IoU to consider a bounding box match valid |
| `--kappa_threshold` | float | 0.6 | Cohen Kappa below which a class is considered problematic |

### Notebook

Open `src/evaluate_ontology.ipynb` in JupyterLab and run the last cell:

```python
run_eval_ontologie(
    eval_folder='../data/',
    output_folder='../output/',
    iou_threshold=0.5,
    kappa_threshold=0.6
)
```

## Metrics

### Mean IoU
Measures the spatial agreement between annotators on bounding box placement. A high IoU indicates that annotators consistently draw boxes around the same regions.

### Agreement Rate
Proportion of matched bounding boxes (IoU ≥ threshold) where both annotators assigned the same class.

### Cohen Kappa (κ)
Measures class label agreement corrected for chance. Computed globally and per class.

| κ | Interpretation |
|---|---|
| < 0.2 | Poor agreement |
| 0.2 – 0.4 | Moderate agreement |
| 0.4 – 0.6 | Fair agreement |
| 0.6 – 0.8 | Good agreement |
| 0.8 – 1.0 | Excellent agreement |

A low per-class Kappa suggests the class definition in the ontology may be ambiguous or insufficiently described.

### Bounding Box Matching
Inter-annotator matching uses the **Hungarian algorithm** (optimal 1-to-1 assignment), which ensures each bounding box is matched at most once. This is more appropriate than greedy matching for comparing human annotators who are expected to annotate each object exactly once.

## Requirements

```
numpy
pandas
scipy
matplotlib
scikit-learn
jupyterlab
ipykernel
```
<!--
## Author

[Chaouabti](https://github.com/Chaouabti)
-->
