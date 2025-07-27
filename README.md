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
   pip install -e .
   ```

3. **Set the root directory**  
   Edit the `ROOT_DIR` variable in `configs.py` to reflect the correct root path for your system.

4. **Setup video-folders path in run_metrics_cinepile.py**  
   Update the `scenes_dir_parent` variable in `run_metrics_cinepile.py` to point to your video dataset directories.
   If you are using a custom video dataset, you might need to make a few changes in lines 147-173 in `run_metrics_cinepile.py` to ensure that esentially your inner-loop is processing over all the video-files in your downlaoded video folder structure. The current code assumes strucutre of type `assets/assets/split_scenes/<video_name>/<scene_name>.mp4


---

## Usage

To run the pipeline, execute the following script. Start-index and end-index are optional parameters that allow you to specify a range of videos to process, which can be useful for large datasets:

```bash
python compute_metrics_ds.py --dataset <DATASET_NAME> --filters <FILTER_LIST> --start_idx <START_INDEX> --end_idx <END_INDEX>
```

The results will be saved as Pandas DataFrames in the `results/` directory.

## Available Filters

- **Aesthetic Metrics**: `laion_aesthetics` (Runs with the base requirements.txt enviornment, automatically downloads everything)
- **motion_score**: `motion_score` (Runs with the base requirements.txt enviornment, automatically downloads everything)
- **shot_categorizer**: `shot_categorizer` (Need to setup new enviornment with `shot_categorizer_requrements.txt`)
- **subject_presence**: `subject_presence` (Need to setup new enviornment with `subject_presence_requirements.txt`)
   - Additionally; in video_filters/subject_presence/subject_presence.py, you need to:
      - download SAM weights (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place them in `video_filters/sam_weights/` folder
      - check `video_filters/subject_presence/subject_presence.py` for the correct path to the SAM weights in line-40
- **vision_reward**: `vision_reward` (Works with the `subject_presence` environment, automatically downloads everything)