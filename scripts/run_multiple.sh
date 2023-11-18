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
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.1
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_dp 0.3
