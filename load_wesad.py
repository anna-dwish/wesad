import pandas as pd
import numpy as np
from itertools import chain

subjects = [i for i in range(2, 18)]
subjects.remove(12)

SUBJECT = "subject"
SIGNAL = "signal"
SENSOR = "sensor"
CHEST = "chest"
WRIST = "wrist"
LABEL = "label"
METRICS = ["EMG", "ECG", "EDA", "Temp", "Resp"]
WRIST_METRICS = ["EDA", "TEMP", "HR"]

UP_SAMPLE = {"EDA": [4, 24]}
PATH = "/hpc/group/sta440-f20/WESAD/WESAD/"
EMP_DATA = "_E4_Data"

STRESS = 2
AMUSEMENT = 3

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
to 4 Hz, this function would return [2, 4, 6, 9, 12, 18, 24, 24]. Note that it assumes goal_Hz > current_Hz and 
goal_Hz % current_Hz = 0
"""


def linear_up_sample(low_res_data, current_Hz, goal_Hz):
    assert(goal_Hz > current_Hz)
    assert(goal_Hz % current_Hz == 0)

    idx = 0
    reps = int(goal_Hz/current_Hz)

    while idx < len(low_res_data):
        # For the last value, we can't do a linear spacing between it and the next value, so just repeat
        if idx + 1 >= len(low_res_data):
            low_res_data[idx:] = [low_res_data[-1] for j in range(reps - 1)]
            break
        else:
            low_res_data[idx:(idx + 2)] = np.linspace(low_res_data[idx], low_res_data[idx + 1], reps).tolist()
        idx = idx + reps - 1
    return low_res_data


"""
This function loads in the pickle file, segments it to the desired sensor, and converts it to a 
dataframe, along with an additional column that will help distinguish it by subject when it is
merged with the rest of the sensor
"""


def generate_subject_df(file_path, s_id):
    file_name = file_path + s_id + ".pkl"
    data = pd.read_pickle(file_name)
    df = pd.DataFrame()
    df_wrist = pd.DataFrame()

    for m in METRICS:
        df[m] = list(flatten(data[SIGNAL][CHEST][m]))

    for m in WRIST_METRICS:
        df_wrist = pd.read_csv(file_path + s_id + EMP_DATA + "/" + m + ".csv", header=None)

        if m in UP_SAMPLE:
            high_res_data = linear_up_sample(list(df_wrist.iloc[:, 0]), UP_SAMPLE[m][0], UP_SAMPLE[m][1])
            df_wrist = pd.DataFrame()
            df_wrist[m] = high_res_data

        rep_factor = int(len(df) / len(df_wrist))
        df_wrist = df_wrist.loc[df_wrist.index.repeat(rep_factor)]

        if len(df_wrist) != len(df):
            # extend the last row for the remaining ones if uneven division
            df_wrist_last = df_wrist.iloc[[-1]]
            df_wrist_last = df_wrist_last.loc[df_wrist_last.index.repeat(len(df) - len(df_wrist))]

            df_wrist = pd.concat([df_wrist, df_wrist_last], ignore_index=True)

        df[m + "_EMP4"] = list(df_wrist.iloc[:, 0])

    df[LABEL] = data[LABEL]
    df = df.loc[df[LABEL].isin([STRESS, AMUSEMENT])]
    df[SUBJECT] = [s_id for i in range(len(df))]

    df = pd.DataFrame(df)

    return df


merged_df = generate_subject_df(PATH + "S2/", "S2")

for s in subjects[1:]:
    sid = "S" + str(s)
    current_subject_df = generate_subject_df(PATH + sid + "/", "S" + str(s))
    merged_df = merged_df.append(current_subject_df)


merged_df.to_csv("merged_wesad.csv")
