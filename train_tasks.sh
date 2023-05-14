#!/bin/bash

# Two tasks / 5 layers / 100 units
python train/individual.py --num_runs 10 \
                           --initial_seed 6789 \
                           --max_seed "10e5" \
                           --num_epochs 50 \
                           --num_hidden 100 100 100 100 100 \
                           --batch_size 100 \
                           --num_train 50000 \
                           --num_test 10000 \
                           --tasks "parity" "value"


python train/parallel.py --num_runs 10 \
                         --initial_seed 6789 \
                         --max_seed "10e5" \
                         --num_epochs 50 \
                         --num_hidden 100 100 100 100 100 \
                         --batch_size 100 \
                         --num_train 50000 \
                         --num_test 10000 \
                         --tasks "parity" "value"

python train/task_switching.py --num_runs 10 \
                               --initial_seed 6789 \
                               --max_seed "10e5" \
                               --num_epochs 50 \
                               --num_hidden 100 100 100 100 100 \
                               --batch_size 100 \
                               --num_train 50000 \
                               --num_test 10000 \
                               --tasks "parity" "value" \
                               --idxs_contexts 0

python train/task_switching.py --num_runs 10 \
                               --initial_seed 6789 \
                               --max_seed "10e5" \
                               --num_epochs 50 \
                               --num_hidden 100 100 100 100 100 \
                               --batch_size 100 \
                               --num_train 50000 \
                               --num_test 10000 \
                               --tasks "parity" "value" \
                               --idxs_contexts 0 1 2 3 4


# Two tasks / 10 layers / 100 units
python train/individual.py --num_runs 10 \
                           --initial_seed 6789 \
                           --max_seed "10e5" \
                           --num_epochs 50 \
                           --num_hidden 100 100 100 100 100 100 100 100 100 100 \
                           --batch_size 100 \
                           --num_train 50000 \
                           --num_test 10000 \
                           --tasks "parity" "value"

python train/parallel.py --num_runs 10 \
                         --initial_seed 6789 \
                         --max_seed "10e5" \
                         --num_epochs 50 \
                         --num_hidden 100 100 100 100 100 100 100 100 100 100 \
                         --batch_size 100 \
                         --num_train 50000 \
                         --num_test 10000 \
                         --tasks "parity" "value"

python train/task_switching.py --num_runs 10 \
                               --initial_seed 6789 \
                               --max_seed "10e5" \
                               --num_epochs 50 \
                               --num_hidden 100 100 100 100 100 100 100 100 100 100 \
                               --batch_size 100 \
                               --num_train 50000 \
                               --num_test 10000 \
                               --tasks "parity" "value" \
                               --idxs_contexts 0

python train/task_switching.py --num_runs 10 \
                               --initial_seed 6789 \
                               --max_seed "10e5" \
                               --num_epochs 50 \
                               --num_hidden 100 100 100 100 100 100 100 100 100 100 \
                               --batch_size 100 \
                               --num_train 50000 \
                               --num_test 10000 \
                               --tasks "parity" "value" \
                               --idxs_contexts 0 1 2 3 4 5 6 7 8 9