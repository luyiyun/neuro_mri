# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --satt_hiddens 256 256 --satt_acts tanh tanh
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt --no_iatt --satt_hiddens 256 256 --satt_acts tanh tanh
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 2022 --no_satt_bn
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.1
python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --no_satt_bn --satt_dp 0.2
