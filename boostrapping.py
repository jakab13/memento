import numpy as np
import pandas as pd
import seaborn as sns
import os
from post_processing import run_post_processing
from scipy.stats import linregress

sns.set_theme()
palette_crest = sns.color_palette("crest", 3)

df = run_post_processing()
df_single_source = df[df["task_type"] == "single_source"]
df_single_source = df_single_source[df_single_source["task_phase"] == "experiment"]
df_multi_source = df[df["task_type"] == "multi_source"]


# MINIMUM NUMBER OF PARTICIPANTS AND TRIALS PER CONDITION FOR SINGLE SOURCE ===========================
output_path = "bs_single_source_linreg.csv"
df_bs_single_source_linreg = pd.DataFrame(columns=["n_subjects", "trials_per_condition", "slope", "pvalue", "rvalue"])
n_conditions = len(df_single_source.target_segment_length.unique())
bs_rows = []
subjects = df_single_source.subject_id.unique()
n_bs = 10000
for n_subjects in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for trials_per_condition in [2, 4, 8]:
        for _ in range(n_bs):
            subjects_resamp = np.random.choice(subjects, n_subjects)
            df_curr = pd.DataFrame(columns=df_single_source.columns)
            for subject_id in subjects_resamp:
                df_sub = df_single_source[df_single_source.subject_id == subject_id]
                df_resamp = df_sub.sample(frac=trials_per_condition * n_conditions, weights="target_segment_length")
                df_curr = pd.concat([df_curr, df_resamp], ignore_index=True)
            linreg = linregress(df_curr.target_segment_length, df_curr.score)
            row = {
                "n_subjects": n_subjects,
                "trials_per_condition": trials_per_condition,
                "slope": linreg.slope,
                "pvalue": linreg.pvalue,
                "rvalue": linreg.rvalue
               }
            df_row = pd.DataFrame.from_dict([row])
            df_row.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

# MINIMUM NUMBER OF PARTICIPANTS AND TRIALS PER CONDITION FOR MULTIPLE SOURCE ===========================
output_path = "bs_multi_source_linreg.csv"
df_bs_multi_source_linreg = pd.DataFrame(columns=["n_subjects", "trials_per_condition", "task_plane", "score_type",
                                                   "slope", "pvalue", "rvalue"])
n_conditions = len(df_single_source.masker_segment_length.unique())
bs_rows = []
subjects = df_multi_source.subject_id.unique()
n_bs = 10000
for n_subjects in [1, 2, 3, 4, 5, 6, 7, 8]:
    for trials_per_condition in [2, 4, 8]:
        for task_plane in df_multi_source.task_plane.unique():
            df_task = df_multi_source[df_multi_source.task_plane == task_plane]
            for _ in range(n_bs):
                subjects_resamp = np.random.choice(subjects, n_subjects)
                df_curr = pd.DataFrame(columns=df_multi_source.columns)
                for subject_id in subjects_resamp:
                    df_sub = df_task[df_task.subject_id == subject_id]
                    df_resamp = df_sub.sample(n=trials_per_condition * 5, weights="masker_segment_length", replace=True)
                    df_curr = pd.concat([df_curr, df_resamp], ignore_index=True)
                linreg = linregress(df_curr.masker_segment_length, df_curr.score_TM_diff)
                row = {
                    "n_subjects": n_subjects,
                    "trials_per_condition": trials_per_condition,
                    "task_plane": task_plane,
                    "score_type": "TM_diff",
                    "slope": linreg.slope,
                    "pvalue": linreg.pvalue,
                    "rvalue": linreg.rvalue
                   }
                df_row = pd.DataFrame.from_dict([row])
                df_row.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
            print("Done with", "n_subjects:", n_subjects,
                  "trials_per_condition:", trials_per_condition,
                  "task_plane:", task_plane)

df_bs_multi_source_linreg = pd.read_csv(output_path)

g = sns.FacetGrid(data=df_bs_multi_source_linreg, col="task_plane", hue="trials_per_condition", palette=palette_crest)
g.map(sns.pointplot, "n_subjects", "pvalue")
g.add_legend(labels=["2", "4", "8"])
[h.set_color(palette_crest[i]) for i, h in enumerate(g.legend.legendHandles)]