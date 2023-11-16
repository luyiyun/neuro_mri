import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("/mnt/data1/tiantan/fn_rm_skull_mv.csv", index_col=0)
df = df.loc[df.v1_filter2, :]
imgsize = df["shape"].str.split("x", expand=True).astype(int)
imgsize.columns = ["height", "width", "depth"]
imgsize["label"] = df["label"].values

fig, axs = plt.subplots(nrows=3, figsize=(8, 8), sharex=False)
for i, name in enumerate(["height", "width", "depth"]):
    sns.histplot(
        data=imgsize, x=name, hue="label", ax=axs[i], kde=True, bins=30
    )
fig.tight_layout()
fig.savefig("/mnt/data1/tiantan/results/plot_imgsize.png")
