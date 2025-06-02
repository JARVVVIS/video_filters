1. Change `ROOT_DIR` variable in download_cinepile_yt.py
2. Run `pip install .` to install `video_filters` as a package
3. `pip install -r requirements.txt` to install other dependencies

Example Commands:


To download a set of videos from CinePile dataset from YouTube (100th to 150th vidoe):
```python
python code/download_cinepile_yt.py --dataset cinepile --start_idx 100 --end_idx 150
```

Bas Script:
```bash
./run_parallel.sh 8 cinepile --enable_watermark_cropping
```

Denotes: 8 parallel processes, cinepile dataset, and enable watermark cropping. It automatically handles the start_idx and end_idx for each process.