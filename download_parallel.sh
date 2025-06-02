#!/bin/bash

# Parallel Video Processing Runner Script
# Usage: ./run_parallel.sh [num_jobs] [dataset_name] [additional_args...]

# Default values
NUM_JOBS=${1:-4}  # Default to 4 parallel jobs
DATASET=${2:-"cinepile"}  # Default dataset
shift 2  # Remove first two arguments so we can pass the rest as additional args

# Your Python script name
SCRIPT_NAME="code/download_cinpile_yt.py"  # Change this to your actual script name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Parallel Video Processing Setup ===${NC}"
echo "Number of parallel jobs: $NUM_JOBS"
echo "Dataset: $DATASET"
echo "Additional arguments: $@"

# First, get dataset info to determine total number of videos
echo -e "\n${YELLOW}Getting dataset information...${NC}"
python $SCRIPT_NAME --dataset $DATASET --info_only

# Read total number of videos (you might need to adjust this parsing)
TOTAL_VIDEOS=$(python $SCRIPT_NAME --dataset $DATASET --info_only 2>/dev/null | grep "Total videos:" | grep -o '[0-9]\+')

if [ -z "$TOTAL_VIDEOS" ]; then
    echo -e "${RED}Error: Could not determine total number of videos${NC}"
    exit 1
fi

echo -e "${GREEN}Total videos found: $TOTAL_VIDEOS${NC}"

# Calculate chunk size
CHUNK_SIZE=$((($TOTAL_VIDEOS + $NUM_JOBS - 1) / $NUM_JOBS))  # Ceiling division
echo "Chunk size per job: $CHUNK_SIZE"

# Create logs directory
LOGS_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOGS_DIR
echo "Logs will be saved to: $LOGS_DIR"

# Array to store job PIDs
pids=()

echo -e "\n${BLUE}Starting parallel jobs...${NC}"

# Launch parallel jobs
for ((i=0; i<$NUM_JOBS; i++)); do
    START_IDX=$((i * CHUNK_SIZE))
    END_IDX=$(((i + 1) * CHUNK_SIZE))
    
    # Don't exceed total videos
    if [ $END_IDX -gt $TOTAL_VIDEOS ]; then
        END_IDX=$TOTAL_VIDEOS
    fi
    
    # Skip if start index exceeds total
    if [ $START_IDX -ge $TOTAL_VIDEOS ]; then
        break
    fi
    
    LOG_FILE="$LOGS_DIR/job_${i}_${START_IDX}_${END_IDX}.log"
    
    echo -e "${YELLOW}Starting job $((i+1)): videos $START_IDX to $((END_IDX-1)) -> $LOG_FILE${NC}"
    
    # Launch the job in background
    python $SCRIPT_NAME \
        --dataset $DATASET \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --enable_watermark_cropping \
        "$@" > $LOG_FILE 2>&1 &
    
    # Store the PID
    pids+=($!)
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo -e "\n${GREEN}All jobs launched. PIDs: ${pids[@]}${NC}"
echo -e "${BLUE}Monitoring progress...${NC}"

# Function to check if a process is still running
is_running() {
    kill -0 $1 2>/dev/null
}

# Monitor jobs
while true; do
    running_jobs=0
    completed_jobs=0
    
    echo -e "\n${BLUE}Job Status:${NC}"
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if is_running $pid; then
            echo -e "Job $((i+1)) (PID $pid): ${YELLOW}RUNNING${NC}"
            running_jobs=$((running_jobs + 1))
        else
            echo -e "Job $((i+1)) (PID $pid): ${GREEN}COMPLETED${NC}"
            completed_jobs=$((completed_jobs + 1))
        fi
    done
    
    echo "Running: $running_jobs, Completed: $completed_jobs"
    
    # Check if all jobs are done
    if [ $running_jobs -eq 0 ]; then
        break
    fi
    
    # Wait before next check
    sleep 30
done

echo -e "\n${GREEN}All jobs completed!${NC}"

# Show summary from log files
echo -e "\n${BLUE}=== Processing Summary ===${NC}"
total_successful=0
total_failed=0
total_processed=0

for log_file in $LOGS_DIR/*.log; do
    if [ -f "$log_file" ]; then
        echo -e "\n${YELLOW}$(basename $log_file):${NC}"
        
        # Extract summary info (adjust grep patterns as needed)
        successful=$(grep "Successful:" "$log_file" | tail -1 | grep -o '[0-9]\+' || echo "0")
        failed=$(grep "Failed:" "$log_file" | tail -1 | grep -o '[0-9]\+' || echo "0")
        processed=$(grep "Videos processed:" "$log_file" | tail -1 | grep -o '[0-9]\+' || echo "0")
        
        echo "  Processed: $processed, Successful: $successful, Failed: $failed"
        
        total_processed=$((total_processed + processed))
        total_successful=$((total_successful + successful))
        total_failed=$((total_failed + failed))
    fi
done

echo -e "\n${GREEN}=== FINAL SUMMARY ===${NC}"
echo "Total processed: $total_processed"
echo "Total successful: $total_successful"
echo "Total failed: $total_failed"
echo "Success rate: $(( total_processed > 0 ? (total_successful * 100) / total_processed : 0 ))%"

# Show any errors
echo -e "\n${BLUE}Checking for errors...${NC}"
error_count=$(grep -c "Error\|Exception\|Failed" $LOGS_DIR/*.log 2>/dev/null || echo "0")
if [ $error_count -gt 0 ]; then
    echo -e "${RED}Found $error_count error messages. Check log files for details.${NC}"
    echo "Recent errors:"
    grep -h "Error\|Exception\|Failed" $LOGS_DIR/*.log | tail -10
else
    echo -e "${GREEN}No errors found!${NC}"
fi

echo -e "\n${BLUE}Logs saved in: $LOGS_DIR${NC}"