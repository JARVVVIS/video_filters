import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import traceback, queue as pyqueue
from PIL import Image
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import threading
import queue
import time
from tqdm import tqdm
import traceback
import warnings
import logging
from datetime import datetime

import random
random.seed(2025)

warnings.filterwarnings("ignore")

from video_filters.frame_utils import return_frames
from configs import ROOT_DIR, FILTER_MODULES, get_filter_module

ARTIFACT_LOAD_ERROR = "__ARTIFACT_LOAD_ERROR__"


# Simple logging setup
def setup_logging(output_dir):
    """Setup basic logging to file and console"""
    log_file = os.path.join(
        output_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure logger
    logger = logging.getLogger("video_processor")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class GPUWorkerPool:
    """Manages a pool of GPU workers for parallel processing"""

    def __init__(self, num_gpus, workers_per_gpu, filter_name, args):
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.filter_name = filter_name
        self.args = args
        self.work_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        self.stop_event = mp.Event()

    def start(self):
        """Start all worker processes"""
        for gpu_id in range(self.num_gpus):
            for worker_id in range(self.workers_per_gpu):
                p = mp.Process(
                    target=gpu_worker,
                    args=(
                        self.work_queue,
                        self.result_queue,
                        self.stop_event,
                        gpu_id,
                        worker_id,
                        self.filter_name,
                        self.args,
                    ),
                )
                p.start()
                self.workers.append(p)

    def add_work(self, work_item):
        """Add work to the queue"""
        self.work_queue.put(work_item)

    def get_result(self):
        """Get a result from the queue"""
        return self.result_queue.get()

    def shutdown(self):
        """Shutdown all workers and clean up MP queues to avoid atexit hangs."""
        self.stop_event.set()
        # Wake workers
        for _ in self.workers:
            try:
                self.work_queue.put(None)
            except Exception:
                pass

        # Join workers
        for p in self.workers:
            p.join(timeout=10)
        # Force-kill any stragglers
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        # **IMPORTANT**: close & join MP queues so feeder threads exit cleanly
        try:
            self.work_queue.close()
            self.work_queue.join_thread()
        except Exception:
            pass
        try:
            self.result_queue.close()
            self.result_queue.join_thread()
        except Exception:
            pass


def gpu_worker(
    work_queue, result_queue, stop_event, gpu_id, worker_id, filter_name, args
):
    """Worker process that runs on a specific GPU with batched inference per loop."""

    # Pin this process to a single visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load filter artifacts once per worker
    filter_module = get_filter_module(filter_name)
    args_copy = argparse.Namespace(**vars(args))
    args_copy.device = f"cuda:{0}"  # Always 0 since we set CUDA_VISIBLE_DEVICES
    args_copy.filter_prefix = f"{ROOT_DIR}/video_filters/{filter_name}"

    try:
        filter_artifacts = filter_module.load_artifacts(args_copy)
    except Exception as e:
        err = f"Worker {gpu_id}-{worker_id}: Failed to load artifacts: {e}\n{traceback.format_exc()}"
        try:
            result_queue.put((ARTIFACT_LOAD_ERROR, None, err))
        except Exception:
            pass
        return

    batch_size = max(1, getattr(args, "batch_size", 1))
    print(
        f"Worker {gpu_id}-{worker_id} started on GPU {gpu_id} (batch_size={batch_size})"
    )

    while not stop_event.is_set():
        try:
            # Block for the first item so we don't busy-wait
            first_item = work_queue.get(timeout=1)
            if first_item is None:  # sentinel
                break

            batch = [first_item]
            # Try to fill the batch without blocking
            while len(batch) < batch_size:
                try:
                    nxt = work_queue.get_nowait()
                    if nxt is None:
                        # Put sentinel back for other workers and stop filling
                        work_queue.put(None)
                        break
                    batch.append(nxt)
                except pyqueue.Empty:
                    break

            # Prepare inputs for this batch
            video_paths = [it[0] for it in batch]

            try:
                # Run batched inference per filter
                if filter_name == "vision_reward":
                    # Our batched VisionReward API takes a list of video sources
                    outputs = filter_module.infer_batch(
                        video_sources=video_paths, artifacts=filter_artifacts
                    )
                else:
                    # Most other filters accept a list of frame lists
                    frames_list = [
                        return_frames(p, discard_last=False) for p in video_paths
                    ]
                    outputs = filter_module.infer_batch(
                        frames_list=frames_list, artifacts=filter_artifacts
                    )

                # Emit one result per scene in order
                for sid, out in zip(video_paths, outputs):
                    result_queue.put((sid, out, None))
            except Exception as e_batch:
                print(e_batch)
                print(traceback.format_exc())
                err = (
                    f"Batch error then item error for {video_paths[0]}:\n"
                    f"BATCH EXC: {str(e_batch)}\n"
                    f"ITEM  EXC: {str(e_batch)}\n{traceback.format_exc()}"
                )
                for sid in video_paths:
                    result_queue.put((sid, None, err))

        except pyqueue.Empty:
            continue
        except Exception as e:
            print(f"Worker {gpu_id}-{worker_id} error: {e}")
            break

    print(f"Worker {gpu_id}-{worker_id} stopped")


def cpu_worker(work_item, filter_name, args):
    """CPU worker for filters that don't require GPU"""

    video_path, _filter_name = work_item
    scene_video_id = video_path

    # Load filter artifacts
    filter_module = get_filter_module(filter_name)
    args_copy = argparse.Namespace(**vars(args))
    args_copy.filter_prefix = f"{ROOT_DIR}/video_filters/{filter_name}"

    try:
        filter_artifacts = filter_module.load_artifacts(args_copy)

        # Process the video
        if filter_name == "vision_reward":
            filter_results = filter_module.infer(
                artifacts=filter_artifacts, video_source=video_path
            )
        else:
            frames_list = return_frames(video_path, discard_last=False)
            filter_results = filter_module.infer(
                frames_list=frames_list, artifacts=filter_artifacts
            )

        return scene_video_id, filter_results, None

    except Exception as e:
        error_msg = (
            f"Error processing {scene_video_id}: {str(e)}\n{traceback.format_exc()}"
        )
        return scene_video_id, None, error_msg


def get_openvid_paths(root_dir):
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    video_paths = list(root_dir.glob("*.mp4"))
    return sorted(video_paths)


def save_checkpoint(df, checkpoint_csv, checkpoint_pkl):
    """Save checkpoint with thread safety"""
    try:
        df.to_csv(checkpoint_csv, index=False)
        df.to_pickle(checkpoint_pkl)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def process_filter_with_multiprocessing(
    filter_name,
    video_items_subset,
    args,
    processed_video_ids,
    checkpoint_csv,
    checkpoint_pkl,
    logger,
):
    """Process a filter using multiprocessing"""
    # Filter out already processed videos
    scenes_to_process = [
        (video_path, filter_name)
        for video_path in video_items_subset
        if video_path not in processed_video_ids
    ]

    if not scenes_to_process:
        logger.info(f"No new scenes to process for filter {filter_name}")
        return pd.DataFrame()

    logger.info(f"Processing {len(scenes_to_process)} scenes for filter {filter_name}")

    results_list = []
    errors_list = []

    # Performance tracking
    start_time = time.time()
    last_checkpoint_time = start_time
    videos_since_last_checkpoint = 0
    processing_times = []

    # Determine if we should use GPU or CPU processing
    use_gpu = args.num_gpus > 0 and torch.cuda.is_available()

    if use_gpu:
        total_workers = args.num_gpus * args.workers_per_gpu
        logger.info(
            f"Using {args.num_gpus} GPUs with {args.workers_per_gpu} workers per GPU (total: {total_workers} workers)"
        )

        # Create GPU worker pool
        worker_pool = GPUWorkerPool(
            args.num_gpus, args.workers_per_gpu, filter_name, args
        )
        worker_pool.start()

        # For speed_run, only submit what we need
        if args.speed_run:
            scenes_to_submit = scenes_to_process[
                : args.checkpoint_freq + 50
            ]  # Add buffer
        else:
            scenes_to_submit = scenes_to_process

        # Submit work
        for work_item in scenes_to_submit:
            worker_pool.add_work(work_item)

        # Add sentinel values to signal workers to stop after processing
        for _ in range(total_workers):
            worker_pool.add_work(None)

        # Collect results with progress bar
        num_to_collect = (
            len(scenes_to_submit)
            if not args.speed_run
            else min(args.checkpoint_freq, len(scenes_to_submit))
        )

        with tqdm(
            total=len(scenes_to_process), desc=f"Processing {filter_name}"
        ) as pbar:
            for idx in range(len(scenes_to_submit)):
                try:
                    video_start_time = time.time()
                    scene_video_id, filter_results, error = worker_pool.get_result()

                    ## Skip if artifact load error
                    if scene_video_id == ARTIFACT_LOAD_ERROR:
                        logger.error("Artifact load failure reported by a worker:")
                        logger.error(error if error else "Unknown artifact load error")
                        worker_pool.shutdown()
                        if getattr(args, "abort_on_artifact_error", False):
                            logger.error(
                                "Exiting entire run due to --abort_on_artifact_error."
                            )
                            sys.exit(2)
                        logger.error(
                            f"Skipping filter '{filter_name}' due to artifact load error."
                        )
                        return pd.DataFrame()

                    video_process_time = time.time() - video_start_time
                    processing_times.append(video_process_time)

                    if error:
                        errors_list.append((scene_video_id, error))
                        logger.debug(
                            f"Error processing {scene_video_id}: {error[:100]}..."
                        )
                    else:
                        result_dict = {"video_id": scene_video_id}
                        result_dict.update(filter_results)
                        results_list.append(result_dict)

                    pbar.update(1)
                    videos_since_last_checkpoint += 1

                    # Save checkpoint and log performance metrics
                    if len(results_list) % args.checkpoint_freq == 0 and results_list:
                        checkpoint_time = time.time()
                        time_for_checkpoint = checkpoint_time - last_checkpoint_time
                        avg_time_per_video = (
                            time_for_checkpoint / videos_since_last_checkpoint
                        )
                        throughput = videos_since_last_checkpoint / time_for_checkpoint

                        logger.info(f"\n{'=' * 60}")
                        logger.info(
                            f"CHECKPOINT REACHED - {len(results_list)} videos processed"
                        )
                        logger.info(
                            f"Time for last {videos_since_last_checkpoint} videos: {time_for_checkpoint:.2f}s"
                        )
                        logger.info(
                            f"Average time per video: {avg_time_per_video:.2f}s"
                        )
                        logger.info(f"Throughput: {throughput:.2f} videos/second")
                        logger.info(
                            f"Estimated speedup vs sequential: {1.0 / avg_time_per_video / args.workers_per_gpu / args.num_gpus:.2f}x"
                        )
                        logger.info(
                            f"Total elapsed time: {checkpoint_time - start_time:.2f}s"
                        )
                        logger.info(f"{'=' * 60}\n")

                        temp_df = pd.DataFrame(results_list)
                        save_checkpoint(temp_df, checkpoint_csv, checkpoint_pkl)

                        last_checkpoint_time = checkpoint_time
                        videos_since_last_checkpoint = 0

                        if args.speed_run:
                            logger.info(
                                "Speed run enabled — exiting after first checkpoint."
                            )
                            # Important: shutdown properly before returning
                            worker_pool.shutdown()
                            return pd.DataFrame(results_list)

                except queue.Empty:
                    # No more results available
                    break
                except Exception as e:
                    logger.error(f"Error collecting result: {e}")
                    break

        # Shutdown worker pool
        worker_pool.shutdown()

    else:
        logger.info(f"Using CPU with {args.num_workers} workers")

        ## Precheck filter loading
        try:
            fm = get_filter_module(filter_name)
            args_copy = argparse.Namespace(**vars(args))
            args_copy.filter_prefix = f"{ROOT_DIR}/video_filters/{filter_name}"
            # Prefer CPU for preflight so we don't pin GPU mem here
            args_copy.device = "cpu"
            _ = fm.load_artifacts(args_copy)
            del _
        except Exception as e:
            logger.error(
                f"Artifact load failed for filter '{filter_name}' (CPU preflight): {e}"
            )
            if getattr(args, "abort_on_artifact_error", False):
                logger.error("Exiting entire run due to --abort_on_artifact_error.")
                sys.exit(2)
            logger.error(f"Skipping filter '{filter_name}' due to artifact load error.")
            return pd.DataFrame()

        # Use ThreadPoolExecutor or ProcessPoolExecutor for CPU processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Create partial function with fixed arguments
            worker_fn = partial(cpu_worker, filter_name=filter_name, args=args)

            # For speed_run, only process up to checkpoint
            if args.speed_run:
                scenes_to_submit = scenes_to_process[: args.checkpoint_freq + 50]
            else:
                scenes_to_submit = scenes_to_process

            # Submit all tasks
            future_to_scene = {
                executor.submit(worker_fn, work_item): work_item[0]
                for work_item in scenes_to_submit
            }

            # Collect results with progress bar
            with tqdm(
                total=len(scenes_to_process), desc=f"Processing {filter_name}"
            ) as pbar:
                for idx, future in enumerate(as_completed(future_to_scene)):
                    video_start_time = time.time()
                    try:
                        scene_video_id, filter_results, error = future.result(
                            timeout=args.timeout
                        )
                        video_process_time = time.time() - video_start_time
                        processing_times.append(video_process_time)

                        if error:
                            errors_list.append((scene_video_id, error))
                            logger.debug(
                                f"Error processing {scene_video_id}: {error[:100]}..."
                            )
                        else:
                            result_dict = {"video_id": scene_video_id}
                            result_dict.update(filter_results)
                            results_list.append(result_dict)
                    except Exception as e:
                        scene_id = future_to_scene[future]
                        errors_list.append((scene_id, str(e)))
                        logger.error(f"Error processing {scene_id}: {e}")

                    pbar.update(1)
                    videos_since_last_checkpoint += 1

                    # Save checkpoint and log performance metrics
                    if len(results_list) % args.checkpoint_freq == 0 and results_list:
                        checkpoint_time = time.time()
                        time_for_checkpoint = checkpoint_time - last_checkpoint_time
                        avg_time_per_video = (
                            time_for_checkpoint / videos_since_last_checkpoint
                        )
                        throughput = videos_since_last_checkpoint / time_for_checkpoint

                        logger.info(f"\n{'=' * 60}")
                        logger.info(
                            f"CHECKPOINT REACHED - {len(results_list)} videos processed"
                        )
                        logger.info(
                            f"Time for last {videos_since_last_checkpoint} videos: {time_for_checkpoint:.2f}s"
                        )
                        logger.info(
                            f"Average time per video: {avg_time_per_video:.2f}s"
                        )
                        logger.info(f"Throughput: {throughput:.2f} videos/second")
                        logger.info(
                            f"Estimated speedup vs sequential: {1.0 / avg_time_per_video / args.num_workers:.2f}x"
                        )
                        logger.info(
                            f"Total elapsed time: {checkpoint_time - start_time:.2f}s"
                        )
                        logger.info(f"{'=' * 60}\n")

                        temp_df = pd.DataFrame(results_list)
                        save_checkpoint(temp_df, checkpoint_csv, checkpoint_pkl)

                        last_checkpoint_time = checkpoint_time
                        videos_since_last_checkpoint = 0

                        if args.speed_run:
                            logger.info(
                                "Speed run enabled — exiting after first checkpoint."
                            )
                            # Cancel remaining futures
                            for f in future_to_scene:
                                f.cancel()
                            return pd.DataFrame(results_list)

    # Final performance summary
    total_time = time.time() - start_time
    if results_list:
        avg_process_time = np.mean(processing_times) if processing_times else 0
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FINAL PERFORMANCE SUMMARY for {filter_name}")
        logger.info(f"Total videos processed: {len(results_list)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(
            f"Overall throughput: {len(results_list) / total_time:.2f} videos/second"
        )
        logger.info(
            f"Average processing time per video: {total_time / len(results_list):.2f}s"
        )
        if use_gpu:
            theoretical_speedup = args.num_gpus * args.workers_per_gpu
            actual_speedup = (
                len(results_list) / (total_time / avg_process_time)
                if avg_process_time > 0
                else 1
            )
            efficiency = (
                (actual_speedup / theoretical_speedup) * 100
                if theoretical_speedup > 0
                else 0
            )
            logger.info(f"Theoretical max speedup: {theoretical_speedup}x")
            logger.info(f"Estimated actual speedup: {actual_speedup:.2f}x")
            logger.info(f"Parallel efficiency: {efficiency:.1f}%")
        logger.info(f"{'=' * 60}\n")

    # Print error summary
    if errors_list:
        logger.warning(f"\n{len(errors_list)} errors occurred during processing:")
        for scene_id, error in errors_list[:5]:  # Show first 5 errors
            logger.warning(f"  - {scene_id}: {error[:100]}...")

    # Return results as DataFrame
    if results_list:
        return pd.DataFrame(results_list)
    else:
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openvid1m")
    parser.add_argument("--filters", nargs="+", default=["laion_aesthetics"])
    parser.add_argument("--output_dir", type=str, default=f"{ROOT_DIR}/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aesthetic_model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--discard_last", type=int, default=None)
    parser.add_argument("--watermark_model_path", type=str, default=None)
    parser.add_argument(
        "--shot_categorizer_repo_id", type=str, default="diffusers/shot-categorizer-v0"
    )
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)

    # Multiprocessing arguments
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU only)"
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="Number of worker processes per GPU",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of CPU workers (used when num_gpus=0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (if supported by filter)",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=50, help="Save checkpoint every N videos"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for processing each video",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--speed_run",
        action="store_true",
        help="Exit after the first checkpoint for quick debug/profiling",
    )
    parser.add_argument(
        "--abort_on_artifact_error",
        action="store_true",
        help="Exit the entire run if artifact loading fails for any filter",
    )
    args = parser.parse_args()

    # Set up multiprocessing
    if args.num_gpus > 0:
        mp.set_start_method("spawn", force=True)
        # Check available GPUs
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if args.num_gpus > available_gpus:
                print(
                    f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available"
                )
                args.num_gpus = available_gpus
        else:
            print("Warning: No GPUs available, falling back to CPU")
            args.num_gpus = 0

    for f in args.filters:
        if f not in FILTER_MODULES:
            raise ValueError(f"Invalid filter: {f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("=" * 80)
    logger.info("VIDEO PROCESSING PIPELINE STARTED")
    logger.info("=" * 80)

    # Get video paths for the dataset
    video_items = get_openvid_paths(args.dataset)
    if args.start_idx is None and args.end_idx is None:
        random.shuffle(video_items)
    total_videos = len(video_items)

    # Handle start_idx and end_idx
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = total_videos

    if args.start_idx < 0 or args.end_idx > total_videos:
        args.start_idx = max(0, args.start_idx)
        args.end_idx = min(total_videos, args.end_idx)

    video_items_subset = video_items[args.start_idx : args.end_idx]

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Video Source: {args.dataset}")
    logger.info(
        f"  Processing Range: {args.start_idx} to {args.end_idx - 1} ({len(video_items_subset)} videos)"
    )
    logger.info(f"  Filters: {args.filters}")
    logger.info("Multiprocessing Configuration:")
    logger.info(f"  GPUs: {args.num_gpus}")
    if args.num_gpus > 0:
        logger.info(f"  Workers per GPU: {args.workers_per_gpu}")
        logger.info(f"  Total GPU workers: {args.num_gpus * args.workers_per_gpu}")
    else:
        logger.info(f"  CPU workers: {args.num_workers}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Checkpoint frequency: {args.checkpoint_freq}")
    logger.info(f"  Timeout per video: {args.timeout}s")
    logger.info(f"  Resume from checkpoint: {args.resume}")
    logger.info("=" * 80)

    # Collect all scenes upfront
    logger.info("Collecting all scene paths...")
    logger.info(f"Found {len(video_items_subset)} total scenes to process")

    # Track overall pipeline performance
    pipeline_start_time = time.time()

    # Process each filter
    for filter_idx, filter_name in enumerate(args.filters, 1):
        filter_start_time = time.time()

        logger.info(f"\n{'=' * 80}")
        logger.info(f"FILTER {filter_idx}/{len(args.filters)}: {filter_name}")
        logger.info(f"{'=' * 80}")

        # Define output paths
        base_output_path = os.path.join(args.output_dir, f"{filter_name}_results")
        range_suffix = f"_{args.start_idx}_{args.end_idx}"
        output_path = base_output_path + range_suffix
        checkpoint_csv = f"{output_path}_checkpoint.csv"
        checkpoint_pkl = f"{output_path}_checkpoint.pkl"

        # Load checkpoint if exists and resume is enabled
        filter_results_df = pd.DataFrame()
        processed_video_ids = set()

        if args.resume and os.path.exists(checkpoint_csv):
            logger.info(f"Found existing checkpoint: {checkpoint_csv}")
            try:
                filter_results_df = pd.read_csv(checkpoint_csv)
                processed_video_ids = set(filter_results_df["video_id"].unique())
                logger.info(
                    f"Resuming from {len(processed_video_ids)} already processed videos"
                )
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.info("Starting fresh...")

        # Process filter with multiprocessing
        new_results_df = process_filter_with_multiprocessing(
            filter_name,
            video_items_subset,
            args,
            processed_video_ids,
            checkpoint_csv,
            checkpoint_pkl,
            logger,
        )

        # Combine with existing results
        if not new_results_df.empty:
            if filter_results_df.empty:
                filter_results_df = new_results_df
            else:
                filter_results_df = pd.concat(
                    [filter_results_df, new_results_df], ignore_index=True
                )

        # Final save
        if not filter_results_df.empty:
            # Ensure "video_id" is the first column
            final_cols = list(filter_results_df.columns)
            if "video_id" in final_cols:
                final_cols.remove("video_id")
                final_cols = ["video_id"] + final_cols
            filter_results_df = filter_results_df.reindex(columns=final_cols)

            # Save final results
            filter_results_df.to_csv(f"{output_path}.csv", index=False)
            filter_results_df.to_pickle(f"{output_path}.pkl")

            filter_time = time.time() - filter_start_time
            logger.info(f"\nFilter '{filter_name}' completed in {filter_time:.2f}s")
            logger.info(f"Saved final results:")
            logger.info(f"  - CSV: {output_path}.csv")
            logger.info(f"  - Pickle: {output_path}.pkl")
            logger.info(f"  - Total processed: {len(filter_results_df)} scenes")
        else:
            logger.warning(f"No results to save for '{filter_name}'")

        if args.speed_run:
            logger.info("Speed run enabled — exiting after first filter.")
            sys.exit(0)

    # Final summary
    total_pipeline_time = time.time() - pipeline_start_time
    logger.info(f"\n{'=' * 80}")
    logger.info("PIPELINE COMPLETED")
    logger.info(f"Total pipeline execution time: {total_pipeline_time:.2f}s")
    logger.info(
        f"Average time per filter: {total_pipeline_time / len(args.filters):.2f}s"
    )
    logger.info("=" * 80)

if __name__ == "__main__":
    main()