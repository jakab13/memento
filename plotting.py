import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import statsmodels.formula.api as smf

sns.set_theme()
palette_tab10 = sns.color_palette("tab10", 10)

df = pd.read_csv("memento_pilot.csv")
final_subjects = ["pilot_jakab", "pilot_zofia", "pilot_lukas", "pilot_oskar"]
df_single_source = df[df["task_type"] == "single_source"]
df_single_source_final = df_single_source[df_single_source.subject_id.isin(final_subjects)]
df_multi_source = df[df["task_type"] == "multi_source"]
df_multi_source_final = df_multi_source[df_multi_source.subject_id.isin(final_subjects)]

for subject_id in df_single_source.subject_id.unique():
    df_sub = df_single_source[df_single_source["subject_id"] == subject_id]
    plt.figure()
    fig = sns.pointplot(
        data=df_sub,
        x="target_segment_length",
        y="score"
    )
    plt.ylim(0, 1.1)
    title = "Single source intelligibility (" + subject_id + ")"
    plt.title(title)
    plt.savefig(title, dpi=400, overwrite=True)
    # plt.show()

g = sns.FacetGrid(df_multi_source, col="subject_id", col_wrap=4, hue="task_plane", sharex=False)
g.map(sns.pointplot, "masker_segment_length", "score_masker")
g.add_legend()
plt.savefig("Masker score (subject grid)", dpi=400, overwrite=True)

g = sns.pointplot(
    data=df_multi_source_final,
    x="masker_segment_length",
    y="score",
    hue="task_plane",
    hue_order=["azimuth", "elevation", "collocated"]
)
title = f"Release from masking (n={len(df_multi_source_final.subject_id.unique())})"
g.set_title(title, fontsize=16)
g.set(xlabel="Masker reversed segment duration (ms)", ylabel="Target intelligibility (%)", ylim=(0, 1.1))
xticks = np.asarray(sorted(df_multi_source_final.masker_segment_length.unique()))
g.set_xticks(range(len(xticks)), labels=(xticks * 1000).astype(int))
plt.legend(title='spatial config', loc='upper left')
plt.savefig(title, dpi=400, overwrite=True)

g = sns.pointplot(
    data=df_single_source_final,
    x="target_segment_length",
    y="score",
    alpha=0
)
title = f"Intelligibility vs segment duration (n={len(df_multi_source_final.subject_id.unique())})"
g.set_title(title, fontsize=16)
g.set(xlabel="Reversed segment duration (ms)", ylabel="Intelligibility (%)", ylim=(0, 1.1))
xticks = np.asarray(sorted(df_single_source_final.target_segment_length.unique()))
g.set_xticks(range(len(xticks)), labels=(xticks * 1000).astype(int))
plt.savefig(title, dpi=400)

# Mixed linear model
md = smf.mixedlm("score ~ masker_segment_length + C(task_plane)", df_multi_source_final, groups="subject_id")
mdf = md.fit()
print(mdf.summary())

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

fig, ax = plt.subplots(1, 3, sharey=True)
masker_segment_lengths = df_multi_source_final.masker_segment_length.unique()
masker_segment_lengths.sort()
task_planes = ["collocated", "elevation", "azimuth"]
N = len(masker_segment_lengths)
for task_idx, task_plane in enumerate(task_planes):
    ax_curr = ax[task_idx]
    df_curr = df_multi_source_final[df_multi_source_final["task_plane"] == task_plane]
    target_means = df_curr.groupby("masker_segment_length")["score"].mean()
    masker_means = df_curr.groupby("masker_segment_length")["score_masker"].mean()
    target_std = df_curr.groupby("masker_segment_length")["score"].std()
    masker_std = df_curr.groupby("masker_segment_length")["score_masker"].std()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars: can also be len(x) sequence
    p1 = ax_curr.bar(ind - width/2, target_means, width, label="target")
    p2 = ax_curr.bar(ind + width/2, masker_means, width, label="masker")
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


g = sns.FacetGrid(df_multi_source_final, col="task_plane", sharey=True, sharex=True, col_order=["collocated", "elevation", "azimuth"], height=6, aspect=0.6)
score_target = g.map(sns.pointplot, "masker_segment_length", "score", color=palette_tab10[0])
# score_masker = g.map(sns.pointplot, "masker_segment_length", "score_masker", color=palette_tab10[1])
g.set_titles(template="{col_name}")
# title = f'Target and Masker intelligibility (n={len(df_multi_source_final.subject_id.unique())})'
title = f'Target intelligibility in the presence of a masker (n={len(df_multi_source_final.subject_id.unique())})'
# g.fig.suptitle(title)
# g.add_legend(labels=["target", "masker"])
g.add_legend(labels=["target"])
# g.legend.legendHandles[1].set_color(palette_tab10[1])
g.set_xlabels(label="Masker reversed segment duration (ms)", clear_inner=True)
g.set_ylabels(label='Intelligibility (%)')
xticks = np.asarray(sorted(df_multi_source_final.masker_segment_length.unique()))
g.set_xticklabels((xticks * 1000).astype(int))
plt.ylim(0, 1)
plt.savefig(title, dpi=400)