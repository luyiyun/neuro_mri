import matplotlib.pyplot as plt
import pandas as pd
import sci_palettes
import seaborn as sns

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
palette_colors = sns.color_palette(use_palette)
plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(constrained_layout=True, figsize=(10, 5))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1])
fig_hwd, fig_vol = subfigs[0], subfigs[1]
fig_hwd.text(
    0.03,
    0.98,
    "A",
    fontweight="bold",
    fontsize=12,
    va="center",
    ha="center",
)
fig_vol.text(
    0.03,
    0.98,
    "B",
    fontweight="bold",
    fontsize=12,
    va="center",
    ha="center",
)

axs = fig_hwd.subplots(nrows=3)
for i, (name, ax) in enumerate(zip(["Height", "Width", "Depth"], axs)):
    sns.barplot(
        data=imgsize_melt_cnt.query("var == '%s'" % name),
        x="value",
        y="count",
        ax=ax,
        # palette=use_palette,
        color=palette_colors[i + 2],
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.spines[["right", "top"]].set_visible(False)
    ax.bar_label(ax.containers[0], fmt="%d")
fig_hwd.supylabel("Number of Images")
fig_hwd.supxlabel("Length (pixels)")

ax = fig_vol.subplots()
sns.barplot(
    data=imgsize_vol_cnt,
    x="Volume",
    y="count",
    hue="label",
    ax=ax,
    palette=use_palette,
)
ax.set_xlabel("Volume ($pixels^3$)")
ax.set_ylabel("Number of Images")
ax.spines[["right", "top"]].set_visible(False)
ax.bar_label(ax.containers[0], fmt="%d")
ax.bar_label(ax.containers[1], fmt="%d")

# fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
# sns.barplot(
#     data=imgsize_melt_cnt,
#     x="value",
#     y="count",
#     hue="var",
#     ax=axs[0],
#     palette=use_palette,
# )
# sns.barplot(
#     data=imgsize_vol_cnt,
#     x="Volume",
#     y="count",
#     hue="label",
#     ax=axs[1],
#     palette=use_palette,
# )
#
# # axs[1].xaxis.set_tick_params(labelrotation=15)
# axs[0].set_xlabel("Length (pixels)")
# axs[1].set_xlabel("Volume ($pixels^3$)")
# for ax in axs:
#     ax.get_legend().set(title="", frame_on=False)
#     ax.set_ylabel("Number of Images")
#     ax.spines[['right', 'top']].set_visible(False)
# fig.tight_layout()

fig.savefig("/mnt/data1/tiantan/results/plot_imgsize.png", dpi=300)
fig.savefig("/mnt/data1/tiantan/results/plot_imgsize.pdf")
