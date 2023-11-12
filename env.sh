set -v

# 必须分开实现，不然会默认从conda-forge中安装torch，而torchaudio等包只能从
# pytorch channel中安装，这会导致两者存在不兼容，无法使用
torch_pkgs="python=3.8 pytorch torchvision torchaudio pytorch-cuda=12.1"
mamba install $torch_pkgs -c pytorch -c nvidia -y
unset torch_pkgs

pkgs="tqdm numpy scipy pandas matplotlib scikit-learn seaborn nibabel"
export pkgs="$pkgs ipython ipdb flake8 jupyterlab tensorboard"
export pkgs="$pkgs monai torchmetrics"
echo $pkgs
mamba install $pkgs -c conda-forge -y
unset pkgs

pkgs_pip="lmdb nipype[all] timm scikit-image openpyxl"
echo $pkgs_pip
pip install $pkgs_pip
unset pkgs_pip
