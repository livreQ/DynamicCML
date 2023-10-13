#!/bin/bash
# for m in $(seq 20 20 200)
#     do
#         for method in {1,2,3,4}
#            do
#             python src/main/run_gaussian.py --sample_num $m --K 25 --hazard 0.05 --method $method > log_"$m"_25_"$hazard"_"$method"
#     done
# done

# for K in $(seq 10 10 50)
#     do
#         for method in {1,2,3,4}
#             do
#             python src/main/run_gaussian.py --sample_num 100 --K $K --hazard 0.05 --method $method > log_100_"$K"_0.05_"$method"
#         done
# done

for seed in $(seq 100 4 120)
    do
        for hazard in $(seq -2.0 0.4 0.2)
            do
                for method in {1,2,3,4}
                    do
                    python src/main/run_gaussian.py --sample_num 100 --K 2 --hazard $hazard --method $method --seed $seed > log_100_2_"$hazard"_"$method"
                done
        done        
done
