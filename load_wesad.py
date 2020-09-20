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


def load_pkl(file_name, s_id, type=CHEST):
    data = pd.read_pickle(file_name)
    df = pd.DataFrame()

    for m in METRICS:
        df[m] = list(flatten(data[SIGNAL][CHEST][m]))

    df[LABEL] = data[LABEL]
    df = df.loc[df[LABEL].isin([STRESS, AMUSEMENT])]
    df[SUBJECT] = [s_id for i in range(len(df))]

    df = pd.DataFrame(df)

    return df


chest_df = load_pkl(PATH + "S2/S2.pkl", "S2")

for s in subjects[1:]:
    sid = "S" + str(s)
    curr_df = load_pkl(PATH + sid + "/" + sid + ".pkl", "S" + str(s))
    chest_df = chest_df.append(curr_df)


chest_df.to_csv("merged_chest.csv")


