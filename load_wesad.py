import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

subjects = [i for i in range(2, 18)]
subjects.remove(12)

SUBJECT = "subject"
SIGNAL = "signal"
SENSOR = "sensor"
CHEST = "chest"
WRIST = "wrist"
LABEL = "label"

STRESS = 2
AMUSEMENT = 3

CHEST_METRICS = ["EMG", "ECG", "EDA", "Temp", "Resp"]
WRIST_METRICS = ["EDA", "BVP", "TEMP"]

UP_SAMPLE = {"EDA": [4, 24], "HR": [1, 4]}

PATH = "/hpc/group/sta440-f20/WESAD/WESAD/"

"""
For all of the CHEST metrics, SR was 700 Hz, so there are 700 * 60 samples/min and 175 samples represent 0.25 seconds
For EDA, with a final SR of 24Hz, there are 24 * 60 = 1440 samples/min and 6 samples represent 0.25 seconds
For TEMP, with a final SR of 4Hz, there is 4 * 60 = 240 samples/min, and 1 sample represents 0.25 seconds
For BVP, with a final SR of 64 Hz, there is 64 * 60 = 3840 samples/min, and 16 samples represents 0.25 seconds
For HR, with a final SR of 4Hz, there is 4 * 60 = 240 samples/min, and 1 sample represents 0.25 seconds

For the final csv, we removed some of the many transformation we attempted for runtime purposes, as our initial 
explorations & modeling steps did not show any evidence that they provided any additional useful information. If you
wish to examine any additional transformations, simply add an entry to the inner dictionary of your chosen
metric where the key is a label for the transformation, and value is a function that takes in a numeric vector and 
returns a single value
"""

ROLL_APPLY = {"EMG_CHEST": [42000, 175, {"MEAN": np.mean}],

              "ECG_CHEST": [42000, 175, {"MEAN": np.mean}],

              "EDA_CHEST": [42000, 175, {"MEAN": np.mean}],

              "Temp_CHEST": [42000, 175, {"MEAN": np.mean}],

              "Resp_CHEST": [42000, 175, {"MEAN": np.mean}],

              "EDA_EMP4": [1440, 6, {"MEAN": np.mean}],

              "TEMP_EMP4": [240, 1, {"MEAN": np.mean}],

              "BVP_EMP4": [3840, 16, {"MEAN": np.mean}],

              "HR_EMP4": [240, 1, {"MEAN": np.mean, "STDDEV": np.std}]}


"""
This function basically un-nests the nested arrays within the pickle format
"""


def flatten(list_of_lists):
    "Flatten one level of nesting"
    return chain.from_iterable(list_of_lists)


"""
This function up-samples the low resolution data from its current_Hz to its goal_Hz. It accomplishes this with linear
spacing between each pair of values (for the final value, it simply repeats the value). For example, if I was given an
input of [2, 6, 12, 24] where the current_Hz = 2 (so this input covers 2 seconds) and I wanted to linearly up-sample
to 4 Hz, this function would return [2.0, 3.33, 4.66, 6.0, 12.0, 16.0, 20.0, 24.0]. Note that it assumes goal_Hz > 
current_Hz, and will fail otherwise.
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


def roll_apply(high_res_data, func=np.mean, window=42000, shift=175, goal_length=None):
    assert (window < len(high_res_data))

    idx = 0
    lower_res_data = []

    while idx + window <= len(high_res_data):
        lower_res_data.append(func(high_res_data[idx:idx + window]))
        idx = idx + shift
        if goal_length is not None and len(lower_res_data) == goal_length - 1:
            lower_res_data.append(func(high_res_data[idx:]))
            return lower_res_data

    if idx < len(high_res_data):
        lower_res_data.append(func(high_res_data[idx:]))

    return lower_res_data


"""
This function applies the mutations required by the input metric and updates the given data-frame. metric_title
provides a title for the column
"""


def apply_mutations(df, x, metric):
    m_window = ROLL_APPLY[metric][0]
    m_shift = ROLL_APPLY[metric][1]
    mutations_dict = ROLL_APPLY[metric][2]
    print(metric)
    for sub_title in mutations_dict:

        mutation_func = mutations_dict[sub_title]
        goal = None
        if len(df) > 0:
            goal = len(df)

        mutated_metric = roll_apply(x, window=m_window, shift=m_shift, func=mutation_func, goal_length=goal)

        if len(df) != 0 and len(mutated_metric) < len(df):
            mutated_metric.extend(mutated_metric[-1] * (len(df) - len(mutated_metric)))

        print(len(df))
        print(len(mutated_metric))
        df[metric + "_" + sub_title] = mutated_metric


"""
This function loads in the pickle file, up-samples if necessary, applies the necessary mutations, 
and converts it to a data frame, along with additional columns for the current subject and label
"""


def generate_subject_df(file_path, s_id):
    file_name = file_path + s_id + ".pkl"
    data = pd.read_pickle(file_name)
    df = pd.DataFrame()

    for m in CHEST_METRICS:
        if s_id == "S3":
            # Data issue - these values were all impossible i.e. temperatures of absolute 0, etc
            data[SIGNAL][CHEST][m][1771709:1771793] = np.linspace(data[SIGNAL][CHEST][m][1771709],
                                                                  data[SIGNAL][CHEST][m][1771793],
                                                                  1771793 - 1771709)
        m_chest = list(flatten(data[SIGNAL][CHEST][m]))
        apply_mutations(df, m_chest, m + "_CHEST")

    for m in WRIST_METRICS:

        m_wrist = list(flatten(data[SIGNAL][WRIST][m]))

        if m in UP_SAMPLE:
            m_wrist = linear_up_sample(m_wrist, UP_SAMPLE[m][0], UP_SAMPLE[m][1])

        apply_mutations(df, m_wrist, m + "_EMP4")

    hr = pd.read_csv(file_path + "HR.csv")
    hr = [0] * 10 + list(hr.iloc[1:, 0])
    hr = linear_up_sample(hr, UP_SAMPLE["HR"][0], UP_SAMPLE["HR"][1])

    apply_mutations(df, hr, "HR_EMP4")

    df[LABEL] = roll_apply(data[LABEL], func=lambda x: (Counter(x)).most_common(1)[0][0])
    df = df.loc[df[LABEL].isin([STRESS, AMUSEMENT])]
    df[LABEL].replace({STRESS: "Stressed", AMUSEMENT: "Amused"}, inplace=True)
    df[SUBJECT] = [s_id for i in range(len(df))]

    return df


status_file = open(r"status_csv.txt", "a")
status_file.write("Subject 2 has begun!\n")
status_file.close()

sid = "S" + str(subjects[0])
merged_df = generate_subject_df(PATH + sid + "/", sid)

status_file = open(r"status_csv.txt", "a")
status_file.write("Subject 2 has finished!\n")
status_file.close()

for s in subjects[1:]:
    status_file = open(r"status_csv.txt", "a")
    status_file.write("Subject " + str(s) + " has begun!\n")
    status_file.close()

    sid = "S" + str(s)
    current_subject_df = generate_subject_df(PATH + sid + "/", "S" + str(s))
    merged_df = merged_df.append(current_subject_df)

    status_file = open(r"status_csv.txt", "a")
    status_file.write("Subject " + str(s) + " has finished!\n")
    status_file.close()

merged_df_name = "merged_wesad_minimal.csv"
merged_df.to_csv(merged_df_name)
status_file = open(r"status_csv.txt", "a")
status_file.write("Finished csv: " + merged_df_name)
status_file.close()
