import importlib

ROOT_DIR = "/BRAIN/adv-robustness/work/t2v_curate"

FILTER_MODULES = {
    "laion_aesthetics": ("video_filters.laion_aesthetics", False),
    "laion_nsfw": ("video_filters.laion_nsfw", False),
    "laion_watermark": ("video_filters.laion_watermark", False),
    "motion_score": ("video_filters.motion_score", False),
    "shot_categorizer": ("video_filters.shot_categorizer", False),
    "subject_presence": ("video_filters.subject_presence", False),
}


def get_filter_module(filter_name):
    """Dynamically import only the needed model module based on model_name."""

    if filter_name not in FILTER_MODULES.keys():
        raise NotImplementedError(f"Model {filter_name} not implemented")

    module_name, needs_video = FILTER_MODULES[filter_name]
    return importlib.import_module(module_name), needs_video
