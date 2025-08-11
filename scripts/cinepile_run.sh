python run_metrics_cinepile.py \
    --filters shot_categorizer \
    --num_gpus 2 \
    --workers_per_gpu 2 \
    --checkpoint_freq 120 \
    --batch_size 20 \
    --speed_run