# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_hiddens 256 256 --satt_acts tanh tanh
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt --no_iatt --satt_hiddens 256 256 --satt_acts tanh tanh
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 2022 --no_satt_bn
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.1
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.2
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_hiddens 256 256 --satt_acts tanh tanh --satt_dp 0.2
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.2 --w_kl_satt 0.01
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.2 --w_kl_satt 0.001
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.3
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.2
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.2 --learning_rate 0.01 --lr_schedual
#

# for lr in 0.0005 0.0001 0.001
# do
#   for n_satt_hiddens in 0 1 2
#   do
#     if [ ${n_satt_hiddens} -eq 0 ]
#     then
#       satt_hiddens=""
#       satt_acts=""
#     elif [ ${n_satt_hiddens} -eq 1 ]
#     then
#       satt_hiddens="256"
#       satt_acts="tanh"
#     elif [ ${n_satt_hiddens} -eq 2 ]
#     then
#       satt_hiddens="256 256"
#       satt_acts="tanh tanh"
#     fi
#     for satt_dp in 0.1 0.2 0.3
#     do
#       for iatt in 0 1
#       do
#         if [ $iatt -eq 0 ]
#         then
#           iatt_arg="--no_iatt"
#         else
#           iatt_arg=""
#         fi
#         python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 \
#           --learning_rate $lr --satt_hiddens $satt_hiddens --satt_acts $satt_acts --satt_dp $satt_dp $iatt_arg \
#
#       done
#     done
#   done
# done

# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.2
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt --learning_rate 0.01 --lr_schedual
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.2 --learning_rate 0.01 --lr_schedual
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt --lr_schedual
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.2 --lr_schedual
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.1
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.3
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.1 --slice_index 2 10
#

# ablation study
# 1. 无satt
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt
# 2. 无iatt（这里satt的dropout设为None）
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_iatt
# 4. 无iatt（这里satt的dropout设为0.1，算是一个补充）
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --satt_dp 0.1
# 5. 无satt kl regularization
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt None
# 6. 无focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --loss_func ce
# 7. satt kl reg的权重变化
# for w_kl_satt in 0.001 0.01 0.05 0.1 0.2
# do
#   python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt $w_kl_satt
# done
# for w_kl_satt in 0.3 0.4 0.5 0.7 1.0
# do
#   python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt $w_kl_satt
# done
# 8. focal loss的参数变化 TODO
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --..

# comparisons
# 1. cnn3d + focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model cnn3d
# 2. cnn3d TODO
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model cnn3d --loss_func ce
# 3. 无satt和iatt (cnn2d) + focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --no_iatt
# 4. 无satt和iatt (cnn2d) TODO
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --no_iatt --loss_func ce
# 5. svc
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model sklearn_svc
# 6. rf
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model sklearn_rf


# 实现新的结果
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt_bn
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt_bn
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt 0.01 ....
