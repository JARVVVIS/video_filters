import importlib

ROOT_DIR = "/fs/nexus-projects/mt_sec/video_filters"

FILTER_MODULES = {
    "laion_aesthetics": "video_filters.laion_aesthetics",
    "motion_score": "video_filters.motion_score",
    "shot_categorizer": "video_filters.shot_categorizer",
    "subject_presence": "video_filters.subject_presence",
    "vision_reward": "video_filters.vision_reward",
}

def get_filter_module(filter_name):
    """Dynamically import only the needed model module based on model_name."""

    if filter_name not in FILTER_MODULES.keys():
        raise NotImplementedError(f"Model {filter_name} not implemented")

    module_name = FILTER_MODULES[filter_name]
    return importlib.import_module(module_name)
