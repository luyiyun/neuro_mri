# comparisons
# 1. cnn3d + focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model cnn3d
# 2. cnn3d
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model cnn3d --loss_func ce
# 3. 无satt和iatt (cnn2d) + focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --no_iatt
# 4. 无satt和iatt (cnn2d)
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --no_iatt --loss_func ce
# 5. svc
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model sklearn_svc
# 6. rf
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --model sklearn_rf
# 7. proposed, 无focal loss
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --loss_func ce


# ablation study
# 1. 无satt
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt
# 2. 无iatt（这里satt的dropout设为None）
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_iatt
# 4. 无iatt（这里satt的dropout设为0.1，算是一个补充）
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --no_satt --satt_dp 0.1
# 5. 无satt kl regularization
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt None
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt None --loss_func ce
# 6. satt kl reg的权重变化
# for w_kl_satt in 0.001 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7 1.0
# do
#   python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt $w_kl_satt
# done
# for w_kl_satt in 0.001 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.7 1.0
# do
#   python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --w_kl_satt $w_kl_satt --loss_func ce
# done
# 7. focal loss的参数变化
# for alpha in 0.25 0.5 0.75
# do
#   for gamma in 0.5 1.0 2.0 5.0
#   do
#     python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 \
#       --focal_alpha $alpha --focal_gamma $gamma
#   done
# done
# 出现NaN，换个seed试试
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 1 --slice_index 2 10 --focal_alpha 0.5 --focal_gamma 0.5
# 8. backbone: 从头训练 & freezing
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --satt_dp 0.1 --no_backbone_pretrained
# python ./scripts/train.py --device "cuda:1" --cv 5 --seed 0 --slice_index 2 10 --satt_dp 0.1 --backbone_freeze


# plotting
for pyname in ablation comparison heatmap hist imgsize
do
  echo "Plot ${pyname}"
  python ./scripts/plot_${pyname}.py
done
