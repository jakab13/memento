import pandas as pd
from analysis import load_df


def run_post_processing(save=True):
    df = load_df()
    # speaker_table = pd.read_csv("speakertable_dome.csv")
    stim_features = pd.read_csv("data/tables/stim_features.csv")

    # Insert speaker locations
    df.loc[df["target_speaker_chan"] == 1, ['target_speaker_azi', 'target_speaker_ele']] = 0, 0

    speaker_chan_locs = df["masker_speaker_chan"] == 1
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 0

    speaker_chan_locs = (df["masker_speaker_chan"] == 18) & (df["task_plane"] == "azimuth")
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 17.5, 0

    speaker_chan_locs = (df["masker_speaker_chan"] == 23) & (df["task_plane"] == "azimuth")
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 35, 0

    speaker_chan_locs = (df["masker_speaker_chan"] == 4) & (df["task_plane"] == "elevation")
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 37.5

    speaker_chan_locs = (df["masker_speaker_chan"] == 5) & (df["task_plane"] == "elevation")
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 0, 50

    speaker_chan_locs = (df["masker_speaker_chan"] == 18) & (df["task_plane"] == "front-back")
    df.loc[speaker_chan_locs, ['masker_speaker_azi', 'masker_speaker_ele']] = 180, 180

    df["task_plane"].replace("colocated", "collocated", inplace=True)


    def find_feature(filename, feature):
        feature_val = None
        if filename is not None:
            filename_orig = filename[:-17] + filename[-4:]
            features = stim_features.loc[stim_features["filename"] == filename_orig]
            feature_val = features[feature].values[0]
        return feature_val


    df["target_duration"] = [find_feature(f, "duration") for f in df["target_filename"]]
    df["target_centroid"] = [find_feature(f, "centroid") for f in df["target_filename"]]
    df["target_flatness"] = [find_feature(f, "flatness") for f in df["target_filename"]]
    df["masker_duration"] = [find_feature(f, "duration") for f in df["masker_filename"]]
    df["masker_centroid"] = [find_feature(f, "centroid") for f in df["masker_filename"]]
    df["masker_flatness"] = [find_feature(f, "flatness") for f in df["masker_filename"]]
    df["target_masker_duration_diff"] = df["target_duration"] - df["masker_duration"]

    df["reaction_time"] = df["response_time_diff"] - df[["target_duration", "masker_duration"]].max(axis=1)

    # df = df[df["reaction_time"].between(0, 20)]

    if save:
        df.to_csv("reversed_speech.csv")

    return df
