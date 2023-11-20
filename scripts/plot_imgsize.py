import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sci_palettes

df = pd.read_csv("/mnt/data1/tiantan/fn_rm_skull_mv.csv", index_col=0)
df = df.loc[df.v1_filter2, :]
imgsize = df["shape"].str.split("x", expand=True).astype(int)
imgsize.columns = ["Height", "Width", "Depth"]
imgsize["label"] = df["label"].values
imgsize_melt = imgsize.melt(id_vars=["label"], var_name="var")
imgsize_melt_cnt = (
    imgsize_melt.groupby(["var"])["value"].value_counts().reset_index()
)

imgsize["Volume"] = imgsize["Height"] * imgsize["Width"] * imgsize["Depth"]
imgsize_vol_cnt = (
    imgsize.groupby("label")["Volume"].value_counts().reset_index()
)
# print(sci_palettes.PALETTES.keys())

# plot
sci_palettes.register_cmap()
use_palette = "npg_nrc"
plt.rcParams["font.family"] = "Times New Roman"
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
sns.barplot(
    data=imgsize_melt_cnt,
    x="value",
    y="count",
    hue="var",
    ax=axs[0],
    palette=use_palette,
)
sns.barplot(
    data=imgsize_vol_cnt,
    x="Volume",
    y="count",
    hue="label",
    ax=axs[1],
    palette=use_palette,
)

# axs[1].xaxis.set_tick_params(labelrotation=15)
axs[0].set_xlabel("Length (pixels)")
axs[1].set_xlabel("Volume ($pixels^3$)")
for ax in axs:
    ax.get_legend().set(title="", frame_on=False)
    ax.set_ylabel("Number of Images")
    ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()
fig.savefig("/mnt/data1/tiantan/results/plot_imgsize.png")
