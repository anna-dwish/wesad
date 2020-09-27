import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from itertools import chain

subjects = [i for i in range(2, 18)]
subjects.remove(12)
subjects = [10, 11, 13]

SUBJECT = "subject"
SIGNAL = "signal"
SENSOR = "sensor"
CHEST = "chest"
WRIST = "wrist"
LABEL = "label"
CHEST_METRICS = ["EMG", "ECG", "EDA", "Temp", "Resp"]
WRIST_METRICS = ["EDA", "HR", "IBI", "TEMP"]#["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]
MERGED_METRICS = CHEST_METRICS + [m + "_EMP4" for m in WRIST_METRICS] + ["HRV"]
MERGED_METRICS.remove("IBI_EMP4")
PEAKS = ["ECG, Resp"]


UP_SAMPLE = {"EDA": [4, 24]}
PATH = "WESAD/"

STRESS = 2
AMUSEMENT = 3
WINDOW = 25
DOWN_SAMPLING = 14


"""
# This function basically un-nests the nested arrays within the pickle format
"""


def flatten(list_of_lists):
    "Flatten one level of nesting"
    return chain.from_iterable(list_of_lists)


"""
This function up-samples the low resolution data from its current_Hz to its goal_Hz. It accomplishes this with linear
spacing between each pair of values (for the final value, it simply repeats the value). For example, if I was given an
input of [2, 6, 12, 24] where the current_Hz = 2 (so this input covers 2 seconds) and I wanted to linearly up-sample
to 4 Hz, this function would return [2.0, 3.33, 4.66, 6.0, 12.0, 16.0, 20.0, 24.0]. Note that it assumes goal_Hz > 
current_Hz
"""


def linear_up_sample(low_res_data, current_Hz, goal_Hz):
    assert (goal_Hz > current_Hz)

    idx = 0

    while idx < len(low_res_data):
        # For the last value, we can't do a linear spacing between it and the next value, so just repeat
        if idx + current_Hz > len(low_res_data):
            remaining = len(low_res_data) - idx
            low_res_data[idx:] = np.linspace(low_res_data[idx], low_res_data[-1], remaining)
            break
        else:
            low_res_data[idx:(idx + current_Hz)] = np.linspace(low_res_data[idx],
                                                               low_res_data[idx + current_Hz - 1],
                                                               goal_Hz).tolist()
        idx = idx + goal_Hz
    return low_res_data


"""
This function takes in a higher res data object (ie something that's been up-sampled or the chest data) and 
apply the func, by default np.mean, of every *window* with a shift of *shift*. Default is shift every 0.25 seconds  
and apply the func over every 2 seconds worth of data, where the data input is at 700 hz 

Example: high_res_data = [0,1,2,3,4,5,6,7], func = np.mean, window = 4, shift = 2
Result: [1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 6.5, 6.5]
"""


def roll_apply(high_res_data, func=np.mean, window=1400, shift=175):
    assert (window < len(high_res_data))

    idx = 0
    lower_res_data = []

    while idx + window <= len(high_res_data):
        lower_res_data.extend([func(high_res_data[idx:idx + window])] * shift)
        idx = idx + shift

    if idx < len(high_res_data):
        lower_res_data.extend([func(high_res_data[idx:])] * (len(high_res_data) - len(lower_res_data)))

    return lower_res_data


"""
This function will replicate measures, AFTER any needed up-sampling, in order to align with the 700Hz
chest data.
"""


def repeat_observations(low_res_df, goal_length):
    rep_factor = int(goal_length / len(low_res_df))

    if rep_factor > 0:
        low_res_df = low_res_df.loc[low_res_df.index.repeat(rep_factor)]

    if len(low_res_df) != goal_length:
        last_row = low_res_df.iloc[[-1]]
        last_row_rep = last_row.loc[last_row.index.repeat(goal_length - len(low_res_df))]

        low_res_df = pd.concat([low_res_df, last_row_rep], ignore_index=True)

    return low_res_df


"""
This function will handle transforming the IBI csv into heart rate variability information using a 30s sweep
"""


def generate_hrv(ibi, sweep=30):
    prev_length = len(ibi)
    ibi = [float(x) for x in ibi]
    hrv = []
    idx = 0

    while idx < prev_length:
        curr_sum = 0
        start_of_run = idx
        while idx < prev_length and curr_sum < sweep:
            curr_sum += float(ibi[idx])
            idx = idx + 1
        hrv.extend([np.std(ibi[start_of_run:idx])] * (idx - start_of_run))

    return hrv


"""
This function loads in the pickle file, segments it to the desired sensor, and converts it to a
data frame, along with an additional column that will help distinguish it by subject when it is
merged with the rest of the sensor
"""


def generate_subject_df(file_path, s_id):
    file_name = file_path + s_id + ".pkl"
    data = pd.read_pickle(file_name)
    df = pd.DataFrame()

    for m in CHEST_METRICS:
        if s_id == "S3":
            data[SIGNAL][CHEST][m][1771709:1771793] = np.linspace(data[SIGNAL][CHEST][m][1771709],
                                                                  data[SIGNAL][CHEST][m][1771793],
                                                                  1771793 - 1771709)
        df[m] = list(flatten(data[SIGNAL][CHEST][m]))

    for m in WRIST_METRICS:

        df_wrist = pd.read_csv(file_path + m + ".csv", header=None)

        if m in UP_SAMPLE:
            high_res_data = linear_up_sample(list(df_wrist.iloc[:, 0]), UP_SAMPLE[m][0], UP_SAMPLE[m][1])
            df_wrist = pd.DataFrame()
            df_wrist[m] = high_res_data

        if m == "IBI":
            ibi = list(df_wrist.iloc[:, 1])[1:]
            df_wrist = pd.DataFrame()
            df_wrist["HRV"] = generate_hrv(ibi)

            df_wrist = repeat_observations(df_wrist, len(df))
            df["HRV"] = df_wrist["HRV"]
        else:
            df_wrist = repeat_observations(df_wrist, len(df))
            df[m + "_EMP4"] = list(df_wrist.iloc[:, 0])

    for m in MERGED_METRICS:
        if m == "HRV":
            continue
        df[m + "_MEAN"] = roll_apply(df[m])
        df[m + "_STDDEV"] = roll_apply(df[m], func=np.std)
        if m in PEAKS:
            df[m + "_PEAKS"] = roll_apply(df[m], func=find_peaks)

    df[LABEL] = data[LABEL]
    df = df.loc[df[LABEL].isin([STRESS, AMUSEMENT])]
    df[LABEL].replace({STRESS: "Stressed", AMUSEMENT: "Amused"}, inplace=True)

    df[SUBJECT] = [s_id for i in range(len(df))]

    df = df.iloc[::DOWN_SAMPLING, :]

    return df


merged_df = generate_subject_df(PATH + "S2/", "S2")

for s in subjects[1:]:
    sid = "S" + str(s)
    current_subject_df = generate_subject_df(PATH + sid + "/", "S" + str(s))
    merged_df = merged_df.append(current_subject_df)


merged_df.to_csv("merged_down_wesad_10_11_13.csv")
