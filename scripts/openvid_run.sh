python run_metrics_openvid1m.py \
    --filters laion_aesthetics \
    --num_gpus 2 \
    --workers_per_gpu 2 \
    --checkpoint_freq 50 \
    --batch_size 25

## increase: batch_size, workers_per_gpu, num_gpus; as per your GPU memory
## for more parallelization set start_idx and end_idx in run_metrics_openvid1m.py and launch parallel jobs

# python run_metrics_openvid1m.py \
#     --filters laion_aesthetics \
#     --num_gpus 2 \
#     --workers_per_gpu 2 \
#     --checkpoint_freq 25 \
#     --batch_size 5 \
#     --start_idx 0 \
#     --end_idx 1000

# python run_metrics_openvid1m.py \
#     --filters laion_aesthetics \
#     --num_gpus 2 \
#     --workers_per_gpu 2 \
#     --checkpoint_freq 25 \
#     --batch_size 5 \
#     --start_idx 1000 \
#     --end_idx 2000

# python run_metrics_openvid1m.py \
#     --filters laion_aesthetics \
#     --num_gpus 2 \
#     --workers_per_gpu 2 \
#     --checkpoint_freq 25 \
#     --batch_size 5 \
#     --start_idx 2000 \
#     --end_idx 3000
