import pandas as pd
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
WATCH_METRICS = ["EDA", "TEMP", "HR"]
PATH = "/hpc/group/sta440-f20/WESAD/WESAD/"

STRESS = 2
AMUSEMENT = 3

"""
# This function basically un-nests the nested arrays within the pickle format
"""


def flatten(list_of_lists):
    "Flatten one level of nesting"
    return chain.from_iterable(list_of_lists)

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

        up_sampling_factor = int(len(df) / len(df_wrist))
        df_wrist = df_wrist.loc[df_wrist.index.repeat(up_sampling_factor)]

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

