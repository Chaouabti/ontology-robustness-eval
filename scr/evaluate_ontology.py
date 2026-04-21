"""
evaluate_ontology.py
--------------------
Script for evaluating the robustness of a computer vision ontology
by computing inter-annotator agreement metrics.

Usage
-----
    # Minimal (no output saved)
    python evaluate_ontology.py --eval_folder /path/to/data

    # Full
    python evaluate_ontology.py --eval_folder /path/to/data \
                                 --output_folder /path/to/output \
                                 --iou_threshold 0.5 \
                                 --kappa_threshold 0.6

Expected folder structure
-------------------------
    data/
    ├── annotator_1/
    │   ├── labels.txt
    │   └── labels/
    │       ├── image1.txt
    │       └── ...
    ├── annotator_2/
    │   ├── labels.txt
    │   └── labels/
    │       ├── image1.txt
    │       └── ...
    └── ...
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
import matplotlib.pyplot as plt


# ============================================================
# 1. LOADING & SETUP
# ============================================================

def scan_eval_folder(eval_folder: str) -> dict:
    """
    Scans the eval_ontologie folder and returns the structure of annotator folders.

    Parameters
    ----------
    eval_folder : str
        Path to the root evaluation folder containing one subfolder per annotator.

    Returns
    -------
    dict : {annotator_name: {'labels_file': Path, 'labels_dir': Path}}
    """
    eval_data_folder = Path(eval_folder)
    result = {}

    for ann_folder in eval_data_folder.iterdir():
        # Skip any loose files at the root level
        if not ann_folder.is_dir():
            continue

        annotator = ann_folder.name
        # Convert generator to list so we can iterate multiple times
        contents = list(ann_folder.iterdir())

        # Find the labels mapping file and the annotations directory
        labels_file = next((f for f in contents if f.name == "labels.txt"), None)
        labels_dir = next((f for f in contents if f.is_dir()), None)

        result[annotator] = {
            'labels_file': labels_file,
            'labels_dir': labels_dir
        }

    return result


def get_labels(labels_file) -> dict:
    """
    Reads a labels.txt file and returns a dict mapping class IDs to class names.

    Parameters
    ----------
    labels_file : str or Path
        Path to the labels.txt file.

    Returns
    -------
    dict : {class_id_str: class_name}
    """
    labels_dict = {}
    with open(labels_file, 'r') as labels:
        for line in labels:
            key, value = line.strip().split(': ')
            key = key.strip("'")
            value = value.strip("'\n")
            labels_dict[key] = value
    return labels_dict


def load_annotator_data(all_data: dict) -> dict:
    """
    Loads annotations and class mappings for all annotators.

    Parameters
    ----------
    all_data : dict
        Output of scan_eval_folder:
        {annotator_name: {'labels_file': Path, 'labels_dir': Path}}

    Returns
    -------
    dict : {annotator_name: (annotations, labels)}
        - annotations : dict {stem: [[class_id, x, y, w, h], ...]}
        - labels : dict {class_id_str: class_name}
    """
    result = {}

    for annotator, data in all_data.items():
        labels_file = data['labels_file']
        labels_dir = data['labels_dir']

        # Load class ID → class name mapping
        labels = get_labels(str(labels_file))

        annotations = {}
        for txt_file in labels_dir.iterdir():
            # Skip non-annotation files
            if txt_file.suffix != '.txt':
                continue

            stem = txt_file.stem
            boxes = []
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    boxes.append([class_id] + coords)

            annotations[stem] = boxes

        result[annotator] = (annotations, labels)

    return result


def harmonize_labels(all_data: dict) -> dict:
    """
    Builds a global class mapping across all annotators.

    Since each annotator may have different numeric IDs for the same class,
    this function aligns them by class name.

    Parameters
    ----------
    all_data : dict
        {annotator_name: {'labels_file': Path, 'labels_dir': Path}}

    Returns
    -------
    dict : {class_name: {annotator_name: class_id_int}}
    """
    mapping = {}

    for annotator, data in all_data.items():
        labels = get_labels(str(data['labels_file']))

        for class_id, class_name in labels.items():
            if class_name not in mapping:
                # First annotator encountered for this class
                mapping[class_name] = {annotator: int(class_id)}
            else:
                # Add this annotator's ID for the existing class
                mapping[class_name][annotator] = int(class_id)

    return mapping


def get_common_files(all_data: dict) -> dict:
    """
    Returns the stems of images annotated by at least two annotators.

    Parameters
    ----------
    all_data : dict
        {annotator_name: {'labels_file': Path, 'labels_dir': Path}}

    Returns
    -------
    dict : {stem: [annotator1, annotator2, ...]}
        Only stems annotated by at least 2 annotators are included.
    """
    common_files = {}

    for annotator, data in all_data.items():
        for file in Path(data['labels_dir']).iterdir():
            stem = file.stem
            if stem not in common_files:
                # First annotator encountered for this stem
                common_files[stem] = [annotator]
            else:
                # Add this annotator to the existing list
                common_files[stem].append(annotator)

    # Collect stems annotated by only one annotator
    exclusive_stems = [
        stem for stem, annotators in common_files.items()
        if len(annotators) < 2
    ]

    # Remove exclusive stems from the result
    for stem in exclusive_stems:
        common_files.pop(stem)

    return common_files


# ============================================================
# 2. MATCHING & METRICS
# ============================================================

def calculate_iou(box1: list, box2: list) -> float:
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes
    in YOLO normalized format.

    Adapted from the 'bb_intersection_over_union' function on PyImageSearch.
    Coordinates are converted from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max) before computing the IoU.

    Parameters
    ----------
    box1 : list [class_id, x_center, y_center, width, height]
        First bounding box in YOLO normalized format.
    box2 : list [class_id, x_center, y_center, width, height]
        Second bounding box in YOLO normalized format.

    Returns
    -------
    float : IoU value between 0 and 1.
        0 indicates no overlap, 1 indicates perfect overlap.
    """
    # Convert coordinates (x, y, w, h) to (x_min, y_min, x_max, y_max)
    box1_x_min = box1[1] - box1[3] / 2
    box1_y_min = box1[2] - box1[4] / 2
    box1_x_max = box1[1] + box1[3] / 2
    box1_y_max = box1[2] + box1[4] / 2

    box2_x_min = box2[1] - box2[3] / 2
    box2_y_min = box2[2] - box2[4] / 2
    box2_x_max = box2[1] + box2[3] / 2
    box2_y_max = box2[2] + box2[4] / 2

    # Calculate coordinates of the overlap
    x_min = max(box1_x_min, box2_x_min)
    y_min = max(box1_y_min, box2_y_min)
    x_max = min(box1_x_max, box2_x_max)
    y_max = min(box1_y_max, box2_y_max)

    # Calculate the area of the overlap
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate the area of the two bounding boxes
    box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
    box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

    # Calculate the Intersection over Union (IoU)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def get_hungarian_matches(boxes_a: list, boxes_b: list, iou_threshold: float = 0.5) -> dict:
    """
    Finds the optimal 1-to-1 matching between two sets of bounding boxes
    using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

    Unlike greedy matching, each box can only be matched to one other box,
    which is more appropriate for inter-annotator comparison where each
    annotator should have annotated each object once.

    Parameters
    ----------
    boxes_a : list of [class_id, x, y, w, h]
    boxes_b : list of [class_id, x, y, w, h]
    iou_threshold : float
        Minimum IoU to consider a match valid (default 0.5)

    Returns
    -------
    dict : {
        'matched': [(box_a, box_b, iou), ...],
        'unmatched_a': [box_a, ...],
        'unmatched_b': [box_b, ...]
    }
    """
    # Edge case : one of the lists is empty
    if not boxes_a:
        return {'matched': [], 'unmatched_a': [], 'unmatched_b': boxes_b}
    if not boxes_b:
        return {'matched': [], 'unmatched_a': boxes_a, 'unmatched_b': []}

    # Build IoU matrix (rows = boxes_a, cols = boxes_b)
    iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            iou_matrix[i, j] = calculate_iou(box_a, box_b)

    # Hungarian algorithm minimizes cost → we pass (1 - IoU) as cost matrix
    row_indices, col_indices = linear_sum_assignment(1 - iou_matrix)

    matched = []
    matched_a = set()
    matched_b = set()

    for i, j in zip(row_indices, col_indices):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            # Valid match : IoU above threshold
            matched.append((boxes_a[i], boxes_b[j], iou))
            matched_a.add(i)
            matched_b.add(j)
        # If IoU below threshold, both boxes remain unmatched

    # Collect unmatched boxes
    unmatched_a = [boxes_a[i] for i in range(len(boxes_a)) if i not in matched_a]
    unmatched_b = [boxes_b[j] for j in range(len(boxes_b)) if j not in matched_b]

    return {
        'matched': matched,
        'unmatched_a': unmatched_a,
        'unmatched_b': unmatched_b
    }


def compute_pair_metrics(all_data: dict, mapping: dict, common_files: dict,
                         ann_a: str, ann_b: str, iou_threshold: float = 0.5) -> dict:
    """
    Calculates all metrics for a pair of annotators on their common images.

    Parameters
    ----------
    all_data : dict
        {annotator_name: (annotations, labels)}
    mapping : dict
        {class_name: {annotator_name: class_id_int}}
    common_files : dict
        {stem: [annotator1, annotator2, ...]}
    ann_a : str
        Name of the first annotator.
    ann_b : str
        Name of the second annotator.
    iou_threshold : float
        Minimum IoU to consider a match valid (default 0.5)

    Returns
    -------
    dict : {
        'mean_iou': float,
        'agreement_rate': float,
        'kappa': float,
        'per_class_kappa': {class_name: float},
        'per_file_iou': {stem: float}
    }
    """
    # Retrieve annotations for each annotator
    annotations_a, _ = all_data[ann_a]
    annotations_b, _ = all_data[ann_b]

    # Build inverted mapping {class_id_int: class_name} for each annotator
    id_to_name_a = {mapping[k][ann_a]: k for k in mapping if ann_a in mapping[k]}
    id_to_name_b = {mapping[k][ann_b]: k for k in mapping if ann_b in mapping[k]}

    all_iou = []          # all IoU values of matched boxes
    all_classes_a = []    # classes of ann_a for Kappa (matched + unmatched)
    all_classes_b = []    # classes of ann_b for Kappa
    per_file_iou = {}     # mean IoU per image

    for stem, annotators in common_files.items():
        # Only process images annotated by both
        if ann_a not in annotators or ann_b not in annotators:
            continue

        boxes_a = annotations_a.get(stem, [])
        boxes_b = annotations_b.get(stem, [])

        # Optimal 1-to-1 matching
        result = get_hungarian_matches(boxes_a, boxes_b, iou_threshold)

        # Collect IoU and class pairs for matched boxes
        file_ious = []
        for box_a, box_b, iou in result['matched']:
            all_iou.append(iou)
            file_ious.append(iou)
            # Convert IDs to class names
            all_classes_a.append(id_to_name_a.get(box_a[0], 'unknown'))
            all_classes_b.append(id_to_name_b.get(box_b[0], 'unknown'))

        # Unmatched boxes → total disagreement, 'unmatched' class for Kappa
        for box_a in result['unmatched_a']:
            all_classes_a.append(id_to_name_a.get(box_a[0], 'unknown'))
            all_classes_b.append('unmatched')

        for box_b in result['unmatched_b']:
            all_classes_a.append('unmatched')
            all_classes_b.append(id_to_name_b.get(box_b[0], 'unknown'))

        per_file_iou[stem] = np.mean(file_ious) if file_ious else 0.0

    # Global metrics
    mean_iou = np.mean(all_iou) if all_iou else 0.0

    # Agreement rate : proportion of matched boxes with IoU >= threshold AND same class
    agreements = sum(1 for a, b in zip(all_classes_a, all_classes_b) if a == b and a != 'unmatched')
    total = len(all_classes_a)
    agreement_rate = agreements / total if total > 0 else 0.0

    # Global Kappa
    kappa = cohen_kappa_score(all_classes_a, all_classes_b) if len(set(all_classes_a)) > 1 else 1.0

    # Per-class Kappa : binarize for each class
    per_class_kappa = {}
    all_class_names = set(all_classes_a + all_classes_b) - {'unmatched'}
    for class_name in all_class_names:
        binary_a = [1 if c == class_name else 0 for c in all_classes_a]
        binary_b = [1 if c == class_name else 0 for c in all_classes_b]
        if len(set(binary_a)) > 1 or len(set(binary_b)) > 1:
            per_class_kappa[class_name] = cohen_kappa_score(binary_a, binary_b)
        else:
            # Both always agree → kappa = 1
            per_class_kappa[class_name] = 1.0

    return {
        'mean_iou': mean_iou,
        'agreement_rate': agreement_rate,
        'kappa': kappa,
        'per_class_kappa': per_class_kappa,
        'per_file_iou': per_file_iou
    }


# ============================================================
# 3. MATRICES & VISUALIZATIONS
# ============================================================

def build_annotator_matrix(all_pair_metrics: dict, annotators: list, metric: str) -> pd.DataFrame:
    """
    Builds a symmetric NxN matrix for a given metric across all annotator pairs.

    Parameters
    ----------
    all_pair_metrics : dict
        {(ann_a, ann_b): {'mean_iou': float, 'agreement_rate': float,
                          'kappa': float, 'per_class_kappa': dict, 'per_file_iou': dict}}
    annotators : list
        List of annotator names, used as row and column labels.
    metric : str
        Metric to display : 'mean_iou', 'agreement_rate' or 'kappa'

    Returns
    -------
    pd.DataFrame : symmetric NxN matrix with annotator names as index and columns.
        Diagonal is set to 1.0 (perfect agreement with oneself).
    """
    # Initialize NxN matrix with zeros
    matrix = pd.DataFrame(
        np.zeros((len(annotators), len(annotators))),
        index=annotators,
        columns=annotators
    )

    # Fill diagonal with 1.0 (perfect agreement with oneself)
    for ann in annotators:
        matrix.loc[ann, ann] = 1.0

    # Fill upper and lower triangle (symmetric matrix)
    for (ann_a, ann_b), pair_metrics in all_pair_metrics.items():
        value = pair_metrics[metric]
        matrix.loc[ann_a, ann_b] = value
        matrix.loc[ann_b, ann_a] = value

    return matrix


def plot_annotator_matrix(matrix: pd.DataFrame, title: str, output_path: Path = None) -> None:
    """
    Displays an inter-annotator matrix as a heatmap.

    Parameters
    ----------
    matrix : pd.DataFrame
        Symmetric NxN matrix with annotator names as index and columns.
        Output of build_annotator_matrix.
    title : str
        Title of the plot.
    output_path : Path, optional
        If provided, the figure is saved to this path before display.
        If None, the figure is only displayed (default None).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw heatmap with values annotated in each cell
    im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap='RdYlGn')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set axis labels
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(matrix.index)

    # Annotate each cell with its value
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            # Dark text on light cells, light text on dark cells
            color = 'black' if value > 0.5 else 'white'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color)

    ax.set_title(title)
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.show()


def plot_iou_distribution(all_pair_metrics: dict, output_path: Path = None) -> None:
    """
    Displays the distribution of IoU values per annotator pair as boxplots.

    Parameters
    ----------
    all_pair_metrics : dict
        {(ann_a, ann_b): {'mean_iou': float, 'agreement_rate': float,
                          'kappa': float, 'per_class_kappa': dict,
                          'per_file_iou': {stem: float}}}
    output_path : Path, optional
        If provided, the figure is saved to this path before display.
        If None, the figure is only displayed (default None).

    Returns
    -------
    None
    """
    labels = []
    values = []

    for (ann_a, ann_b), pair_metrics in all_pair_metrics.items():
        pair_label = f'{ann_a}\nvs\n{ann_b}'
        labels.append(pair_label)
        values.append(list(pair_metrics['per_file_iou'].values()))

    fig, ax = plt.subplots(figsize=(len(labels) * 2, 6))

    # Boxplot : one box per annotator pair
    ax.boxplot(values, labels=labels)

    # Add mean IoU as a red dot for each pair
    for i, iou_values in enumerate(values):
        mean = np.mean(iou_values) if iou_values else 0
        ax.plot(i + 1, mean, 'ro', markersize=8, label='Mean IoU' if i == 0 else '')

    ax.set_ylim(0, 1)
    ax.set_ylabel('IoU')
    ax.set_title('IoU distribution per annotator pair')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.show()


def plot_confusion_matrix_pair(all_data: dict, mapping: dict, common_files: dict,
                                ann_a: str, ann_b: str, iou_threshold: float = 0.5,
                                output_path: Path = None) -> None:
    """
    Displays the confusion matrix of classes for a pair of annotators.

    Only matched boxes (IoU >= threshold) are considered.
    Unmatched boxes are mapped to 'unmatched' class.

    Parameters
    ----------
    all_data : dict
        {annotator_name: (annotations, labels)}
    mapping : dict
        {class_name: {annotator_name: class_id_int}}
    common_files : dict
        {stem: [annotator1, annotator2, ...]}
    ann_a : str
        Name of the first annotator.
    ann_b : str
        Name of the second annotator.
    iou_threshold : float
        Minimum IoU to consider a match valid (default 0.5)
    output_path : Path, optional
        If provided, the figure is saved to this path before display.
        If None, the figure is only displayed (default None).

    Returns
    -------
    None
    """
    annotations_a, _ = all_data[ann_a]
    annotations_b, _ = all_data[ann_b]

    # Build inverted mapping {class_id_int: class_name} for each annotator
    id_to_name_a = {mapping[k][ann_a]: k for k in mapping if ann_a in mapping[k]}
    id_to_name_b = {mapping[k][ann_b]: k for k in mapping if ann_b in mapping[k]}

    all_classes_a = []
    all_classes_b = []

    for stem, annotators in common_files.items():
        if ann_a not in annotators or ann_b not in annotators:
            continue

        boxes_a = annotations_a.get(stem, [])
        boxes_b = annotations_b.get(stem, [])

        result = get_hungarian_matches(boxes_a, boxes_b, iou_threshold)

        for box_a, box_b, iou in result['matched']:
            all_classes_a.append(id_to_name_a.get(box_a[0], 'unknown'))
            all_classes_b.append(id_to_name_b.get(box_b[0], 'unknown'))

        for box_a in result['unmatched_a']:
            all_classes_a.append(id_to_name_a.get(box_a[0], 'unknown'))
            all_classes_b.append('unmatched')

        for box_b in result['unmatched_b']:
            all_classes_a.append('unmatched')
            all_classes_b.append(id_to_name_b.get(box_b[0], 'unknown'))

    # All unique class labels
    all_labels = sorted(set(all_classes_a + all_classes_b))

    # Compute confusion matrix
    cm = metrics.confusion_matrix(all_classes_a, all_classes_b, labels=all_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')

    ax.set_title(f'Confusion matrix : {ann_a} (y) vs {ann_b} (x)')
    ax.set_ylabel(ann_a)
    ax.set_xlabel(ann_b)

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.show()


# ============================================================
# 4. SYNTHETIC REPORT
# ============================================================

def get_problematic_classes(all_pair_metrics: dict, kappa_threshold: float = 0.6) -> pd.DataFrame:
    """
    Identifies classes with a mean Cohen Kappa below the threshold across all pairs.

    A low kappa for a class suggests its definition in the ontology is ambiguous
    or not clearly understood by annotators.

    Parameters
    ----------
    all_pair_metrics : dict
        {(ann_a, ann_b): {'mean_iou': float, 'agreement_rate': float,
                          'kappa': float, 'per_class_kappa': dict,
                          'per_file_iou': dict}}
    kappa_threshold : float
        Kappa below which a class is considered problematic (default 0.6)

    Returns
    -------
    pd.DataFrame : problematic classes sorted by mean kappa ascending
        Columns : ['class_name', 'mean_kappa', 'min_kappa', 'nb_pairs_below_threshold']
    """
    # Collect kappa values per class across all pairs
    class_kappas = {}

    for (ann_a, ann_b), pair_metrics in all_pair_metrics.items():
        for class_name, kappa in pair_metrics['per_class_kappa'].items():
            if class_name not in class_kappas:
                class_kappas[class_name] = []
            class_kappas[class_name].append(kappa)

    # Build summary rows
    rows = []
    for class_name, kappas in class_kappas.items():
        mean_kappa = np.mean(kappas)
        if mean_kappa < kappa_threshold:
            rows.append({
                'class_name': class_name,
                'mean_kappa': round(mean_kappa, 3),
                'min_kappa': round(min(kappas), 3),
                'nb_pairs_below_threshold': sum(1 for k in kappas if k < kappa_threshold)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"✅ No problematic classes found (kappa threshold = {kappa_threshold})")
        return df

    return df.sort_values('mean_kappa').reset_index(drop=True)


def get_divergent_pairs(all_pair_metrics: dict, iou_threshold: float = 0.5) -> pd.DataFrame:
    """
    Identifies annotator pairs with a mean IoU below the threshold.

    A divergent pair suggests systematic disagreement on bounding box placement,
    which may indicate different interpretations of the annotation guidelines.

    Parameters
    ----------
    all_pair_metrics : dict
        {(ann_a, ann_b): {'mean_iou': float, 'agreement_rate': float,
                          'kappa': float, 'per_class_kappa': dict,
                          'per_file_iou': dict}}
    iou_threshold : float
        Mean IoU below which a pair is considered divergent (default 0.5)

    Returns
    -------
    pd.DataFrame : divergent pairs sorted by mean IoU ascending
        Columns : ['pair', 'mean_iou', 'agreement_rate', 'kappa']
    """
    rows = []

    for (ann_a, ann_b), pair_metrics in all_pair_metrics.items():
        if pair_metrics['mean_iou'] < iou_threshold:
            rows.append({
                'pair': f'{ann_a} vs {ann_b}',
                'mean_iou': round(pair_metrics['mean_iou'], 3),
                'agreement_rate': round(pair_metrics['agreement_rate'], 3),
                'kappa': round(pair_metrics['kappa'], 3)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"✅ No divergent pairs found (IoU threshold = {iou_threshold})")
        return df

    return df.sort_values('mean_iou').reset_index(drop=True)


def generate_report(annotators: list, all_pair_metrics: dict,
                    iou_threshold: float = 0.5, kappa_threshold: float = 0.6) -> None:
    """
    Generates a synthetic text report summarizing inter-annotator agreement
    and ontology robustness.

    Parameters
    ----------
    annotators : list
        List of annotator names.
    all_pair_metrics : dict
        {(ann_a, ann_b): {'mean_iou': float, 'agreement_rate': float,
                          'kappa': float, 'per_class_kappa': dict,
                          'per_file_iou': dict}}
    iou_threshold : float
        IoU threshold used for divergent pair detection (default 0.5)
    kappa_threshold : float
        Kappa threshold used for problematic class detection (default 0.6)

    Returns
    -------
    None
    """
    print("=" * 60)
    print("ONTOLOGY ROBUSTNESS EVALUATION — SUMMARY REPORT")
    print("=" * 60)

    # Global overview
    print(f"\n📋 Annotators ({len(annotators)}) : {', '.join(annotators)}")
    print(f"📊 Number of pairs evaluated : {len(all_pair_metrics)}")

    # Global metrics per pair
    print("\n--- Global metrics per pair ---")
    for (ann_a, ann_b), m in all_pair_metrics.items():
        print(f"\n  {ann_a} vs {ann_b}")
        print(f"    Mean IoU       : {m['mean_iou']:.3f}")
        print(f"    Agreement rate : {m['agreement_rate']:.3f}")
        print(f"    Cohen Kappa    : {m['kappa']:.3f}")

    # Divergent pairs
    print("\n--- Divergent pairs ---")
    divergent = get_divergent_pairs(all_pair_metrics, iou_threshold)
    if not divergent.empty:
        print(divergent.to_string(index=False))

    # Problematic classes
    print("\n--- Problematic classes ---")
    problematic = get_problematic_classes(all_pair_metrics, kappa_threshold)
    if not problematic.empty:
        print(problematic.to_string(index=False))

    # Overall assessment
    mean_kappa_global = np.mean([m['kappa'] for m in all_pair_metrics.values()])
    mean_iou_global = np.mean([m['mean_iou'] for m in all_pair_metrics.values()])

    print("\n--- Overall assessment ---")
    print(f"  Mean IoU across all pairs   : {mean_iou_global:.3f}")
    print(f"  Mean Kappa across all pairs : {mean_kappa_global:.3f}")

    if mean_kappa_global >= 0.8:
        print("\n  ✅ Excellent inter-annotator agreement — ontology is robust.")
    elif mean_kappa_global >= 0.6:
        print("\n  🟡 Good agreement — some classes may need clarification.")
    elif mean_kappa_global >= 0.4:
        print("\n  🟠 Moderate agreement — ontology review recommended.")
    else:
        print("\n  🔴 Poor agreement — ontology requires significant revision.")

    print("\n" + "=" * 60)


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_eval_ontologie(eval_folder: str, output_folder: str = None,
                       iou_threshold: float = 0.5, kappa_threshold: float = 0.6) -> None:
    """
    Full pipeline : scan → load → harmonize → metrics → visualizations → report.

    Parameters
    ----------
    eval_folder : str
        Path to the root evaluation folder containing one subfolder per annotator.
    output_folder : str, optional
        Path to the folder where figures will be saved.
        If None, figures are only displayed and not saved (default None).
    iou_threshold : float
        Minimum IoU to consider a match valid (default 0.5)
    kappa_threshold : float
        Kappa below which a class is considered problematic (default 0.6)

    Returns
    -------
    None
    """
    # 1. Scan the folder structure
    annotators_dict = scan_eval_folder(eval_folder)

    # 2. Load all annotator data
    all_data = load_annotator_data(annotators_dict)

    # 3. Harmonize labels across annotators
    mapping = harmonize_labels(annotators_dict)

    # 4. Get common files
    common_files = get_common_files(annotators_dict)

    # 5. Compute metrics for all pairs
    annotators = list(all_data.keys())
    all_pair_metrics = {}
    for ann_a, ann_b in combinations(annotators, 2):
        all_pair_metrics[(ann_a, ann_b)] = compute_pair_metrics(
            all_data, mapping, common_files, ann_a, ann_b, iou_threshold
        )

    # 6. Build IoU and Kappa matrices
    iou_matrix = build_annotator_matrix(all_pair_metrics, annotators, 'mean_iou')
    kappa_matrix = build_annotator_matrix(all_pair_metrics, annotators, 'kappa')

    # Create output folder if provided
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

    # 7. Plot IoU matrix
    plot_annotator_matrix(iou_matrix, title='Mean IoU between annotators',
                          output_path=output_folder / 'mean_iou_matrix.png' if output_folder else None)

    # 8. Plot Kappa matrix
    plot_annotator_matrix(kappa_matrix, title='Cohen Kappa between annotators',
                          output_path=output_folder / 'kappa_matrix.png' if output_folder else None)

    # 9. Plot IoU distribution
    plot_iou_distribution(all_pair_metrics,
                          output_path=output_folder / 'iou_distribution.png' if output_folder else None)

    # 10. Plot confusion matrix for each pair
    for ann_a, ann_b in combinations(annotators, 2):
        plot_confusion_matrix_pair(
            all_data, mapping, common_files, ann_a, ann_b, iou_threshold,
            output_path=output_folder / f'confusion_{ann_a}_vs_{ann_b}.png' if output_folder else None
        )

    # 11. Generate report
    generate_report(annotators, all_pair_metrics, iou_threshold, kappa_threshold)


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the robustness of a computer vision ontology "
                    "by computing inter-annotator agreement metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        required=True,
        help="Path to the root evaluation folder containing one subfolder per annotator."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to the folder where figures will be saved. "
             "If not provided, figures are only displayed."
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Minimum IoU to consider a bounding box match valid."
    )
    parser.add_argument(
        "--kappa_threshold",
        type=float,
        default=0.6,
        help="Cohen Kappa below which a class is considered problematic."
    )

    args = parser.parse_args()

    run_eval_ontologie(
        eval_folder=args.eval_folder,
        output_folder=args.output_folder,
        iou_threshold=args.iou_threshold,
        kappa_threshold=args.kappa_threshold
    )
