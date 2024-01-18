import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
from sklearn import metrics
import statsmodels.formula.api as smf

# sns.set_theme()

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

so.Plot(df_multi_source_final, x="masker_segment_length", color="sex").add(so.Bar(), so.Count(), so.Stack())