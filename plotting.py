import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import statsmodels.formula.api as smf
from post_processing import run_post_processing
from scipy.stats import linregress, ttest_ind

sns.set_theme()
palette_tab10 = sns.color_palette("tab10", 10)

# df = run_post_processing()
df = pd.read_csv("reversed_speech.csv")
df_single_source = df[df["task_type"] == "single_source"]
df_single_source = df_single_source[df_single_source["task_phase"] == "experiment"]
df_multi_source = df[df["task_type"] == "multi_source"]

for subject_id in df_single_source.subject_id.unique():
    subject_id = "sub_20"
    df_sub = df_single_source[df_single_source["subject_id"] == subject_id]
    df_sub = df_sub[df_sub["task_phase"] == "experiment"]
    plt.figure()
    fig = sns.pointplot(
        data=df_sub,
        x="target_segment_length",
        y="score"
    )
    plt.ylim(0, 1.1)
    title = "Single source intelligibility (" + subject_id + ")"
    plt.title(title)
    plt.savefig(title, dpi=400)
    # plt.show()

g = sns.FacetGrid(df_multi_source, col="subject_id", col_wrap=4, hue="task_plane", sharex=False)
g.map(sns.pointplot, "masker_segment_length", "score_masker")
g.add_legend()
plt.savefig("Masker score (subject grid)", dpi=400, overwrite=True)

g = sns.pointplot(
    data=df_single_source,
    x="target_segment_length",
    y="score",
    capsize=0.05
    # alpha=0
)
title = f"Intelligibility vs segment duration (n={len(df_single_source.subject_id.unique())})"
g.set_title(title, fontsize=16)
g.set(xlabel="Reversed segment duration (ms)", ylabel="Score (%)", ylim=(0, 1.1))
xticks = np.asarray(sorted(df_single_source.target_segment_length.unique()))
g.set_xticks(range(len(xticks)), labels=(xticks * 1000).astype(int))
plt.savefig(title, dpi=400)

# Confusion matrices
cm_colour = metrics.confusion_matrix(df_multi_source["response_colour"], df_multi_source["target_colour"])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_colour, display_labels=[True, True])
cm_display.plot()
plt.show()

colour_labels = ['One', 'Two','Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
cm_target_number = metrics.confusion_matrix(df_multi_source["response_number"], df_multi_source["target_number"],
                                     labels=colour_labels, normalize="true")
cm_masker_number = metrics.confusion_matrix(df_multi_source["response_number"], df_multi_source["masker_number"],
                                     labels=colour_labels, normalize="true")
cm_target_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_target_number, display_labels=colour_labels)
cm_masker_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_masker_number, display_labels=colour_labels)
cm_target_display.plot()
cm_masker_display.plot()
plt.show()

fig, ax = plt.subplots(1, 4, sharey=True)
masker_segment_lengths = df.masker_segment_length.dropna().unique()
masker_segment_lengths.sort()
task_planes = ["collocated", "elevation", "front-back", "azimuth"]
N = len(masker_segment_lengths)
for task_idx, task_plane in enumerate(task_planes):
    ax_curr = ax[task_idx]
    df_curr = df[df["task_plane"] == task_plane]
    target_means = df_curr.groupby("masker_segment_length")["score"].mean()
    masker_means = df_curr.groupby("masker_segment_length")["score_masker"].mean()
    target_std = df_curr.groupby("masker_segment_length")["score"].std()
    masker_std = df_curr.groupby("masker_segment_length")["score_masker"].std()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence
    p1 = ax_curr.bar(ind + width/2, target_means, width, label="target")
    p2 = ax_curr.bar(ind + width/2, masker_means, width, bottom=target_means, label="masker")
    ax_curr.set_title(task_plane)
    ax_curr.set_xticks(ind, (masker_segment_lengths * 1000).astype(int))
    # ax_curr.set_box_aspect(1/0.6)
lines_labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines_labels[0], lines_labels[1], loc='center right')
fig.supxlabel("Masker reversed segment duration (ms)")
fig.supylabel('Intelligibility (%)')
plt.ylim(0, 1.1)
plt.suptitle('Target - Masker scores')
plt.show()


# TARGET AND MASKER SCORES =====================================================================

for subject_id in df.subject_id.unique():
    subject_id = "sub_20"
    df_sub = df_multi_source[(df.subject_id == subject_id)]
    g = sns.FacetGrid(df_multi_source, col="task_plane", sharey=True, sharex=True,
                      col_order=["collocated", "elevation", "front-back", "azimuth"], height=6, aspect=0.6)
    g.map(sns.pointplot, "masker_segment_length", "score_masker", color=palette_tab10[1], capsize=0.05)
    g.map(sns.pointplot, "masker_segment_length", "score", color=palette_tab10[0], capsize=0.05)
    g.set_titles(template="{col_name}")
    title = f'Target and Masker scores in different spatial setups (n={len(df.subject_id.unique())})'
    # title = f'Target and Masker scores in different spatial setups ({subject_id})'
    g.add_legend(labels=["target", "masker"])
    [h.set_color(palette_tab10[i]) for i, h in enumerate(g.legend.legendHandles)]
    g.set_xlabels(label="", clear_inner=True)
    g.set_ylabels(label='Correct (%)')
    plt.ylim(0, 1)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(title)
    g.fig.supxlabel("Masker segment duration (ms)", fontsize=13)
    xticks = np.asarray(sorted(df.masker_segment_length.dropna().unique()))
    g.set_xticklabels((xticks * 1000).astype(int))
    yticks = g.axes[0][0].get_yticks()
    g.axes[0][0].set_yticklabels((yticks * 100).astype(int))
    plt.savefig(title, dpi=400)

# =================================================================================================

g = sns.FacetGrid(df[df.block_index == 1], col="task_plane", sharey=True, sharex=True,
                  col_order=["collocated", "elevation", "front-back", "azimuth"], height=6, aspect=0.6)
g.map(sns.regplot, "trial_index", "score", scatter=False, color=palette_tab10[1])
g.set_titles(template="{col_name}")
plt.ylim(0, 1)


g = sns.FacetGrid(df_multi_source, col="task_plane", sharey=True, sharex=True,
                  col_order=["collocated", "elevation", "front-back", "azimuth"], height=6, aspect=0.6)
g.map(sns.pointplot, "masker_segment_length", "score_TM_diff", color=palette_tab10[5])
g.set_titles(template="{col_name}")


plt.figure(figsize=(6, 10))
plt.axhline(y=0, linestyle="--", color="red", alpha=.2)
plt.axvline(x=0.06, linestyle="--", color="lightgrey")
ax = sns.lineplot(df_multi_source,
             x="masker_segment_length",
             y="score_TM_diff",
             hue="task_plane",
             hue_order=["azimuth", "front-back", "elevation", "collocated"],
             palette=sns.color_palette("viridis", len(df_multi_source.task_plane.unique())),
             style="task_plane",
             markers=True,
             dashes=False,
             markersize=10,
             )
ax.grid(False, axis="x")
plt.legend(title='spatial setup')
title = f"Target - Masker Difference Scores (n={len(df_multi_source.subject_id.unique())})"
plt.title(title, fontsize=20)
ax.set(xlabel='Masker segment duration (ms)', ylabel='Target - Masker Difference Score (%)')
ax.set_xticks(np.asarray(sorted(df.masker_segment_length.dropna().unique())))
xticks = ax.get_xticks()
ax.set_xticklabels((xticks * 1000).astype(int))
yticks = ax.get_yticks()
ax.set_yticklabels(np.rint(yticks * 100).astype(int))
# plt.savefig(title, dpi=400)

# REACTION TIME =======================================================================================
g = sns.FacetGrid(df, col="task_plane", sharey=True, sharex=True,
                  col_order=["collocated", "elevation", "front-back", "azimuth"], height=6, aspect=0.6)
g.map(sns.pointplot, "masker_segment_length", "reaction_time")
g.set_titles(template="{col_name}")
title = f'Reaction time (n={len(df.subject_id.unique())})'
g.set_xlabels(label="", clear_inner=True)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle(title)
g.fig.supxlabel("Masker segment duration (ms)", fontsize=13)
xticks = np.asarray(sorted(df.masker_segment_length.dropna().unique()))
g.set_xticklabels((xticks * 1000).astype(int))
# plt.savefig(title, dpi=400)


g = sns.FacetGrid(df, col="task_plane", sharey=True, sharex=True,
                  col_order=["collocated", "elevation", "front-back", "azimuth"], height=6, aspect=0.6)
g.map(sns.regplot, "reaction_time", "score_TM_diff", scatter=False)
# g.map(sns.regplot, "score_masker", "reaction_time", scatter=False, color=palette_tab10[1])
g.set_titles(template="{col_name}")

sns.histplot(df, x="reaction_time")

sns.lmplot(df_multi_source, x="masker_segment_length", y="score", hue="task_plane", scatter=False)
md = smf.mixedlm("score ~ masker_segment_length + C(task_plane)",
                 data=df_multi_source,
                 groups="subject_id")
mdf = md.fit()
print(mdf.summary())


df_performance = df_multi_source.groupby(["subject_id", "task_plane"], as_index=False)
df_performance = df_performance[["score", "score_masker", "score_TM_diff", "score_TM_diff_square", "elevation_RMSE"]].mean()
df_performance.loc[:, "target_slope"] = None
df_performance.loc[:, "masker_slope"] = None
df_performance.loc[:, "TM_slope_diff"] = None
df_performance.loc[:, "target_intercept"] = None
df_performance.loc[:, "masker_intercept"] = None
df_performance.loc[:, "TM_intercept_diff"] = None
df_performance.loc[:, "single_source_score"] = None
df_performance.loc[:, "single_source_slope"] = None

for subject_id in df.subject_id.unique():
    df_sub = df[df.subject_id == subject_id]
    for task_plane in df_sub.task_plane.dropna().unique():
        df_slope = df_sub[df_sub.task_plane == task_plane]
        reg_line_target = linregress(df_slope["masker_segment_length"], df_slope["score"])
        reg_line_masker = linregress(df_slope["masker_segment_length"], df_slope["score_masker"])
        q = (df_performance.subject_id == subject_id) & (df_performance.task_plane == task_plane)
        df_performance.loc[q, "target_slope"] = reg_line_target.slope
        df_performance.loc[q, "masker_slope"] = reg_line_masker.slope
        df_performance["TM_slope_diff"] = df_performance["target_slope"] - df_performance["masker_slope"]
        df_performance.loc[q, "target_intercept"] = reg_line_target.intercept
        df_performance.loc[q, "masker_intercept"] = reg_line_masker.intercept
        df_performance["TM_intercept_diff"] = df_performance["target_intercept"] - df_performance["masker_intercept"]
    df_sub_single_source = df_sub[df_sub.task_type == "single_source"]
    df_sub_single_source = df_sub_single_source[df_sub_single_source.task_phase != "intro"]
    reg_line_single_source = linregress(df_sub_single_source["target_segment_length"], df_sub_single_source["score"])
    q = (df_performance.subject_id == subject_id)
    df_performance.loc[q, "single_source_score"] = df_sub_single_source["score"].mean()
    df_performance.loc[q, "single_source_slope"] = reg_line_single_source.slope
