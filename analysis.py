import pandas as pd
import slab
import pathlib
import os
from config import COLOURS, NUMBERS
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

pd.options.mode.chained_assignment = None
sns.set_theme()
results_folder = pathlib.Path(os.getcwd()) / "Results"


def load_df():
    subjects_excl = ["holubowska", "jakab", "gina", "pilot_paul", "pilot_varvara", "pilot_carsten", "pilot_sasha", "redundant"]

    subjects = [s for s in os.listdir(results_folder) if not s.startswith('.')]
    subjects = sorted([s for s in subjects if not any(s in excl for excl in subjects_excl)])

    results_files = {s: [f for f in sorted(os.listdir(results_folder / s)) if not f.startswith('.')] for s in subjects}

    columns = ["subject_id", "task_type", "task_phase", "task_plane", "task_level", "block_index",
               "trial_timestamp", "trial_index",
               "target_talker_id", "target_call_sign", "target_colour", "target_number", "target_filename",
               "target_segment_length", "target_reverse_seed", "target_speaker_chan", "target_speaker_proc",  "target_speaker_azi",  "target_speaker_ele",
               "masker_talker_id", "masker_call_sign", "masker_colour", "masker_number", "masker_filename",
               "masker_segment_length", "masker_reverse_seed", "masker_speaker_chan", "masker_speaker_proc",  "masker_speaker_azi",  "masker_speaker_ele",
               "response_colour", "response_number", "response_timestamp", "score"]

    block_counter_to_block_index = {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1, 6: 2, 7: 1, 8: 2, 9: 1, 10: 2}

    df = pd.DataFrame(columns=columns)

    for subject, results_file_list in results_files.items():
        block_counter = 0
        for results_file_name in results_file_list:
            path = results_folder / subject / results_file_name
            exp_params = slab.ResultsFile.read_file(path, tag="exp_params")
            task_params = slab.ResultsFile.read_file(path, tag="task_params")
            if task_params["type"] == "loc_test" or task_params["type"] == "questionnaire":
                continue
            trial_params = slab.ResultsFile.read_file(path, tag="trial_params")
            target_params = slab.ResultsFile.read_file(path, tag="target_params")
            masker_params = slab.ResultsFile.read_file(path, tag="masker_params")
            response_params = slab.ResultsFile.read_file(path, tag="response_params")
            block_length = slab.ResultsFile.read_file(path, tag="trial_seq")["n_trials"]

            df_curr = pd.DataFrame(index=range(0, block_length))
            df_curr["subject_id"] = exp_params["subject_id"]
            df_curr["task_type"] = task_params["type"]
            df_curr["task_phase"] = task_params["phase"]
            df_curr["task_plane"] = task_params["plane"]
            df_curr["task_level"] = task_params["level"]
            df_curr["block_index"] = block_counter_to_block_index[block_counter]
            df_curr["trial_timestamp"] = [trial["timestamp"] for trial in trial_params]
            df_curr["trial_index"] = [trial["index"] for trial in trial_params]
            df_curr["target_talker_id"] = [target["talker_id"] for target in target_params]
            df_curr["target_call_sign"] = [target["call_sign"] for target in target_params]
            df_curr["target_colour"] = [target["colour"] for target in target_params]
            df_curr["target_number"] = [target["number"] for target in target_params]
            df_curr["target_filename"] = [target["filename"] for target in target_params]
            df_curr["target_segment_length"] = [target["segment_length"] for target in target_params]
            df_curr["target_reverse_seed"] = [target["reverse_seed"] for target in target_params]
            df_curr["target_speaker_chan"] = [target.get("speaker_chan") for target in target_params]
            df_curr["target_speaker_proc"] = [target.get("speaker_proc") for target in target_params]
            df_curr["masker_talker_id"] = [masker["talker_id"] for masker in masker_params]
            df_curr["masker_call_sign"] = [masker["call_sign"] for masker in masker_params]
            df_curr["masker_colour"] = [masker["colour"] for masker in masker_params]
            df_curr["masker_number"] = [masker["number"] for masker in masker_params]
            df_curr["masker_filename"] = [masker["filename"] for masker in masker_params]
            df_curr["masker_segment_length"] = [masker["segment_length"] for masker in masker_params]
            df_curr["masker_reverse_seed"] = [masker["reverse_seed"] for masker in masker_params]
            df_curr["masker_speaker_chan"] = [masker.get("speaker_chan") for masker in masker_params]
            df_curr["masker_speaker_proc"] = [masker.get("speaker_proc") for masker in masker_params]
            df_curr["response_colour"] = [response["colour"] for response in response_params]
            df_curr["response_number"] = [response["number"] for response in response_params]
            df_curr["response_timestamp"] = [response["timestamp"] for response in response_params]
            df = pd.concat([df, df_curr], ignore_index=True)
            block_counter += 1
    df["masker_segment_length"] = df["masker_segment_length"].astype(float)
    df["target_segment_length"] = df["target_segment_length"].astype(float)
    df["trial_index"] = df["trial_index"].astype(float)
    df["colour_correct"] = df["target_colour"] == df["response_colour"]
    df["number_correct"] = df["target_number"] == df["response_number"]
    df["score"] = (df["colour_correct"] * len(COLOURS) + df["number_correct"] * len(NUMBERS)) / (len(COLOURS) + len(NUMBERS))
    df["score"] = df["score"].astype(float)
    df["masker_colour_correct"] = df["masker_colour"] == df["response_colour"]
    df["masker_number_correct"] = df["masker_number"] == df["response_number"]
    df["score_masker"] = (df["masker_colour_correct"] * len(COLOURS) + df["masker_number_correct"] * len(NUMBERS)) / (
                len(COLOURS) + len(NUMBERS))
    df["score_masker"] = df["score_masker"].astype(float)
    df["score_TM_diff"] = df["score"] - df["score_masker"]
    df["response_time_diff"] = df["response_timestamp"] - df["trial_timestamp"]
    df["task_plane"].replace("colocated", "collocated", inplace=True)
    return df


def print_current_score(subject_id=None):
    df = load_df()
    df = df[df["subject_id"] == subject_id] if subject_id is not None else df
    df = df[df["task_phase"] == "experiment"]
    mean = df["score"].mean()
    if not pd.isna(mean):
        print(subject_id, "current average is:", f"{round(mean, 2) * 100}%")


def plot_results(subject_id=None, task_type="multi_source"):
    df = load_df()
    df = df[df["subject_id"] == subject_id] if subject_id is not None else df
    df = df[df["task_phase"] == "experiment"]
    df = df[df["task_type"] == task_type]
    if task_type == "single_source":
        ax = sns.pointplot(
            data=df,
            x="target_segment_length",
            y="score"
        )
        title = subject_id or "all subjects"
        ax.set_title(f"Temporal unmasking ({title})")
        ax.set(xlabel='Segment length (ms)', ylabel='Intelligibility (%)')
        linreg = linregress(x=df["target_segment_length"], y=df["score"])
        print(f"Linreg p-value: {subject_id}:  \t{round(linreg.pvalue, 4)}")
    elif task_type == "multi_source":
        for plane in df["task_plane"].unique():
            df_plane = df[df["task_plane"] == plane]
            linreg = linregress(x=df_plane["masker_segment_length"], y=df_plane["score"])
            print(f"Linreg p-value: {subject_id}/{plane}:  \t{round(linreg.pvalue, 4)}")
        ax = sns.pointplot(
            data=df,
            x="masker_segment_length",
            y="score",
            hue="task_plane",
            errorbar="sd"
        )
        title = subject_id or "all subjects"
        ax.set_title(f"Temporal unmasking ({title})")
        ax.set(xlabel='Masker segment length (ms)', ylabel='Target intelligibility (%)')
    ax.set_ylim(0, 1.1)
    plt.show()


def print_multi_source_linear_model(subject_id=None):
    df = load_df()
    df = df[df["task_phase"] == "experiment"]
    df = df[df["task_type"] == "multi_source"]
    df = df[df["subject_id"] == subject_id] if subject_id is not None else df
    for plane in df["task_plane"].unique():
        df_plane = df[df["task_plane"] == plane]
        linreg = linregress(x=df_plane["masker_segment_length"], y=df_plane["score"])
        print(f"Linreg p-value: {subject_id}/{plane}:  \t{round(linreg.pvalue, 4)}")
