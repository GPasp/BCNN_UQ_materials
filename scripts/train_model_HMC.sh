TOTAL_ITERS=20
ITERS_PER_SCRIPT=10
START_ITER=0  # Set the starting point
for (( i=$START_ITER; i < TOTAL_ITERS; i+=ITERS_PER_SCRIPT ))
do
    export start_iter=$i
    export end_iter=$((i + ITERS_PER_SCRIPT))
    python train_model.py
done
