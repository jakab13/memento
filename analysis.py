import pandas as pd
import slab
import pathlib
import os
from config import COLOURS, NUMBERS
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

results_folder = pathlib.Path(os.getcwd()) / "Results"

def load_df():
    subjects_excl = ["test"]

    subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
    subjects = sorted([s for s in subjects if not any(s in excl for excl in subjects_excl)])

    results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

    columns = ["subject_id", "task_type", "task_phase", "task_plane", "task_level",
               "trial_timestamp", "trial_index",
               "target_talker_id", "target_call_sign", "target_colour", "target_number", "target_filename",
               "target_segment_length", "target_reverse_seed", "target_speaker_id",  "target_speaker_azi",  "target_speaker_ele",
               "masker_talker_id", "masker_call_sign", "masker_colour", "masker_number", "masker_filename",
               "masker_segment_length", "masker_reverse_seed", "masker_speaker_id",  "masker_speaker_azi",  "masker_speaker_ele",
               "response_colour", "response_number", "response_timestamp", "score"]

    df = pd.DataFrame(columns=columns)

    for subject, results_file_list in results_files.items():
        for results_file_name in results_file_list:
            path = results_folder / subject / results_file_name
            block_length = slab.ResultsFile.read_file(path, tag="trial_seq")["n_trials"]
            exp_params = slab.ResultsFile.read_file(path, tag="exp_params")
            task_params = slab.ResultsFile.read_file(path, tag="task_params")
            trial_params = slab.ResultsFile.read_file(path, tag="trial_params")
            target_params = slab.ResultsFile.read_file(path, tag="target_params")
            masker_params = slab.ResultsFile.read_file(path, tag="masker_params")
            response_params = slab.ResultsFile.read_file(path, tag="response_params")

            df_curr = pd.DataFrame()
            df_curr["subject_id"] = [exp_params["subject_id"]] * block_length
            df_curr["task_type"] = [task_params["type"]] * block_length
            df_curr["task_phase"] = [task_params["phase"]] * block_length
            df_curr["task_plane"] = [task_params["plane"]] * block_length
            df_curr["task_level"] = [task_params["level"]] * block_length
            df_curr["trial_timestamp"] = [trial["timestamp"] for trial in trial_params]
            df_curr["trial_index"] = [trial["index"] for trial in trial_params]
            df_curr["target_talker_id"] = [target["talker_id"] for target in target_params]
            df_curr["target_call_sign"] = [target["call_sign"] for target in target_params]
            df_curr["target_colour"] = [target["colour"] for target in target_params]
            df_curr["target_number"] = [target["number"] for target in target_params]
            df_curr["target_filename"] = [target["filename"] for target in target_params]
            df_curr["target_segment_length"] = [target["segment_length"] for target in target_params]
            df_curr["target_reverse_seed"] = [target["reverse_seed"] for target in target_params]
            df_curr["target_speaker_id"] = [target["speaker_id"] for target in target_params]
            df_curr["masker_talker_id"] = [masker["talker_id"] for masker in masker_params]
            df_curr["masker_call_sign"] = [masker["call_sign"] for masker in masker_params]
            df_curr["masker_colour"] = [masker["colour"] for masker in masker_params]
            df_curr["masker_number"] = [masker["number"] for masker in masker_params]
            df_curr["masker_filename"] = [masker["filename"] for masker in masker_params]
            df_curr["masker_segment_length"] = [masker["segment_length"] for masker in masker_params]
            df_curr["masker_reverse_seed"] = [masker["reverse_seed"] for masker in masker_params]
            df_curr["masker_speaker_id"] = [masker["speaker_id"] for masker in masker_params]
            df_curr["response_colour"] = [response["colour"] for response in response_params]
            df_curr["response_number"] = [response["number"] for response in response_params]
            df_curr["response_timestamp"] = [response["timestamp"] for response in response_params]
            df = pd.concat([df, df_curr], ignore_index=True)

    df["score_colour_correct"] = df["target_colour"] == df["response_colour"]
    df["score_number_correct"] = df["target_number"] == df["response_number"]
    df["score"] = (df["score_colour_correct"] * len(COLOURS) + df["score_number_correct"] * len(NUMBERS)) / (len(COLOURS) + len(NUMBERS))
    df["score"] = df["score"].astype('float64')
    return df


def plot_results(subject_id=None, task_type="single_source"):
    df = load_df()
    df = df[df["subject_id"] == subject_id] if subject_id is not None else df
    df = df[df["task_phase"] == "experiment"]
    df = df[df["task_type"] == task_type]
    if task_type == "single_source":
        ax = sns.pointplot(
            data=df,
            x="target_segment_length",
            y="score",
            scatter=False
        )
    elif task_type == "multi_source":
        ax = sns.pointplot(
            data=df,
            x="masker_segment_length",
            y="score",
            hue="task_plane",
            errorbar="sd",
            scatter=False,
            facet_kws={'legend_out': True}
        )
        title = subject_id or "all subjects"
        ax.set_title(f"Temporal unmasking ({title})")
        ax.set(xlabel='Masker segment length (ms)', ylabel='Target intelligibility (%)')
    ax.set_ylim(0, 1.1)
    plt.show()