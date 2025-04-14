# Video Filter Metrics Pipeline

This repository provides a modular framework for computing and analyzing various filter-based metrics on video datasets. It includes tools for evaluating video frames based on aesthetics, motion, semantic content, and other characteristics, with a focus on flexibility and extensibility.

---

## Installation

1. **Install dependencies**  
   Use the provided `requirements.txt` to install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install `video_filters` as a local editable package**  
   From the root directory:
   ```bash
   pip install -e ./video_filters
   ```

3. **Set the root directory**  
   Edit the `ROOT_DIR` variable in `code/configs.py` to reflect the correct root path for your system.

---

## Usage

To run the pipeline, execute the following script:

```bash
python code/compute_metrics_ds.py --dataset <DATASET_NAME> --filters <FILTER_LIST>
```

### Arguments

- `--dataset`: Name of the dataset to process.
- `--filters`: Comma-separated list of filters to apply.

The results will be saved as Pandas DataFrames in the `results/` directory.

---

## Project Structure

```
.
├── assets/                 # Contains datasets, keyframes, and other resources
├── notebooks/              # Jupyter notebooks for analysis and visualization
├── video_filters/          # Modular implementations of video filters
│   ├── shot_categorizer/shot_categorizer.py
│   ├── laion_aesthetics/laion_aesthetics.py
│   ├── laion_nsfw/laion_nsfw.py
│   ├── motion_score/motion_score.py
│   ├── watermark_laion_score/watermark_laion_score.py
│   └── subject_presence/subject_presence.py
├── code/
│   ├── compute_metrics_ds.py   # Main script to run metrics
│   └── configs.py              # Configuration file (ROOT_DIR path)
├── results/               # Output DataFrames from metric evaluations
└── requirements.txt       # Python package dependencies
```

---

## Available Filters

### Implemented

- **`shot_categorizer`**  
  Analyzes keyframes for visual diversity across attributes such as color, lighting, lighting type, and composition.

- **`laion_aesthetics`**  
  Predicts aesthetic scores for frames using a model trained on LAION data. Useful for filtering high-quality video content.

- **`laion_nsfw`**  
  Identifies NSFW content on a per-frame basis. Videos exceeding a defined threshold can be excluded.

- **`motion_score`**  
  Calculates motion metrics (`motion_fb`, `motion_lk`) from keyframes.

- **`watermark_laion_score`**  
  Assigns watermark detection scores to individual frames.

- **`subject_presence`**  
  Estimates the number of visual objects (via segmentation masks) present per frame.

---

### Planned

The following filters are under development or planned:

- `tv_clip_score`
- `dover_tech_prespective`
- `dover_aesthetic_prespective`
- `average_flow_score_RAFT`
- `amplitude_classification_score` (from EvalCrafter)
- `warping_error` (from EvalCrafter)
- `clip_based_semantic_consistency`
- `exact_dedup`
- `semantic_dedup`
- `avg_spatial_relationships` (caption-based)
- `avg_temporal_action` (caption-based)

---

## Output

All computed metrics are saved as Pandas DataFrames in the `results/` directory. These outputs can be used for further processing, visualization, or downstream tasks.