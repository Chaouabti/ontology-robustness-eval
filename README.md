# Ontology Robustness Evaluation

A tool for evaluating the robustness of a computer vision ontology by computing inter-annotator agreement metrics on YOLO-format annotations.

## Overview

When building a computer vision dataset, ensuring that annotators agree on how to apply ontology classes is critical. This tool compares annotations from multiple annotators on the same images and produces:

- **Mean IoU matrix**: how closely annotators place the bounding boxes they both draw
- **Cohen Kappa matrix**: how consistently annotators assign class labels and agree on which regions are boxed (boxes drawn by one annotator only are taken into account)
- **IoU distribution**: per-pair boxplots of the IoU of every matched box
- **Confusion matrices**: per-pair class agreement and disagreement, including unmatched boxes
- **Summary report** (printed and saved as a `.txt` file): matched and unmatched boxes per pair, per-class kappa with the number of boxes behind it, classes used by only one annotator, problematic classes and divergent pairs

## Repository Structure

```
ontology-robustness-eval/
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

Each annotation file follows the standard YOLO format, one line per bounding box:

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

> **Note:** Class IDs do not need to be consistent across annotators: the tool harmonizes them by matching **class names**. Two annotators who use different names for the same class are therefore treated as using two different classes (see *Classes used by only one annotator* below).

## Installation

Python 3.10 or later is required.

Clone the repository and install dependencies:

```bash
git clone https://github.com/Chaouabti/ontology-robustness-eval.git
cd ontology-robustness-eval
pip install -r requirements.txt
```

## Usage

### Script

```bash
# Minimal: figures displayed only, report saved as ./summary_report.txt
python src/evaluate_ontology.py --eval_folder data/

# Full: figures and report saved to the output folder
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
| `--output_folder` | str | None | Path to save figures and `summary_report.txt`. If not provided, figures are only displayed and the report is written to the current directory |
| `--iou_threshold` | float | 0.5 | Minimum IoU to consider a bounding box match valid |
| `--kappa_threshold` | float | 0.6 | Per-class Cohen Kappa below which a class is considered problematic |

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

For every pair of annotators, the evaluation is run on the images annotated by **both** of them. Bounding boxes are matched first; agreement is measured afterwards.

### Bounding box matching

The boxes of the two annotators are matched one-to-one with the **Hungarian algorithm** (optimal assignment, each box matched at most once). This is more appropriate than greedy matching for comparing annotators who are each expected to annotate every object exactly once.

A pair of boxes is **matched** when its IoU is at least `iou_threshold` (default `0.5`). A box without a counterpart is **unmatched**.

### Metrics reported for each pair

| Metric | Definition |
|---|---|
| **Images annotated by both** | Number of images annotated by the two annotators of the pair. |
| **Boxes matched / unmatched** | Number of matched box pairs, and number of boxes (from either annotator) left unmatched. |
| **Mean IoU** | Mean IoU over all matched boxes, all common images pooled. Matched boxes reach the IoU threshold by construction, so unmatched boxes do not enter this value. |
| **Same class among matched** | Number of matched box pairs to which both annotators gave the same class, out of the number of matched pairs. |
| **Agreement rate** | Matched box pairs with the same class, divided by (matched box pairs + unmatched boxes). Every unmatched box counts as a disagreement. |
| **Cohen's κ** | Cohen's kappa computed over the class labels of all matched pairs and unmatched boxes. An unmatched box receives an extra label, `unmatched`, for the annotator who did not draw it. |

κ therefore reflects two kinds of disagreement at once: disagreement on the **class** of a box, and disagreement on **whether and how a region is boxed**. Comparing *Same class among matched* with the number of unmatched boxes tells which of the two drives a low κ.

κ is interpreted with the scale of Landis & Koch (1977):

| κ | Interpretation |
|---|---|
| < 0 | Poor |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost perfect |

> Landis, J. R. and Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159–174.

κ measures how consistently the annotators apply the categories. It does not show whether the ontology is complete or appropriate for a given use.

### Per-class kappa

For each class, a one-versus-rest κ is computed over the same entries as the global κ. Next to each value the report gives the number of boxes of that class drawn by each annotator, and the number of matched pairs where both gave that class (`same label`).

**Read per-class κ together with these counts.** With few boxes it is very unstable: with 4 boxes per annotator, a single disagreement moves κ from 1.0 to about 0.7.

### Classes used by only one annotator

If, in a pair, one annotator never used a class, the per-class κ of that class is always 0. It then says nothing about how the class is defined: the class was not annotated by that annotator, or was named differently (classes are aligned across annotators **by name**). The report lists these cases separately, and they are left out of the *problematic classes* below. Look at the `labels.txt` of the annotators to tell the two cases apart: two classes with mirrored counts (0 vs *n* and *n* vs 0) suggest the same class under two names.

### Problematic classes

A class is reported as problematic when its mean per-class κ, over the pairs in which **both** annotators used it, is below `kappa_threshold` (default `0.6`). A low κ suggests that the definition of the class in the ontology is ambiguous or insufficiently described.

### Divergent pairs

A pair is reported as divergent when its mean IoU is below `iou_threshold`. Since matched boxes always reach that threshold, this only happens when a pair has **no matched box at all**. The matched / unmatched counts and the agreement rate are the indicators to use for weaker disagreement.

### Overall assessment

The report ends with the mean IoU and the mean κ across pairs (unweighted mean of the pairwise values), and a one-line verdict based on the mean κ:

| Mean κ | Verdict |
|---|---|
| ≥ 0.8 | Almost perfect agreement: categories were applied consistently |
| ≥ 0.6 | Substantial agreement: some classes may need clarification |
| ≥ 0.4 | Moderate agreement: ontology review recommended |
| < 0.4 | Low agreement: ontology requires significant revision |

The cut-offs of the verdict are rounded (0.8, 0.6, 0.4) compared with the scale above.

## Outputs

With `output_folder` set, the evaluation saves:

| File | Content |
|---|---|
| `summary_report.txt` | The report printed in the notebook. |
| `kappa_matrix.png` | Cohen's κ between annotators. |
| `mean_iou_matrix.png` | Mean IoU between annotators. |
| `iou_distribution.png` | Distribution of the IoU of every matched box, for each pair (all common images pooled). The red dot is the mean, equal to the *Mean IoU* of the report. Matched boxes have an IoU of at least `iou_threshold`, so unmatched boxes are not shown: see the matched / unmatched counts of the report. |
| `confusion_<A>_vs_<B>.png` | Confusion matrix of the classes for a pair, including the `unmatched` label. |

The report also lists, for each pair, the images where at least one annotator drew a box but no box could be matched. An image where neither annotator drew a box is ignored.

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
## Author
[Chaouabti](https://github.com/Chaouabti)