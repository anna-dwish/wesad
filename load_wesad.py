import pandas as pd

subjects = [i for i in range(2, 18)]
subjects.remove(12)

SUBJECT = "subject"
SIGNAL = "signal"
SENSOR = "sensor"
CHEST = "chest"
WRIST = "wrist"

"""
This function loads in the pickle file, segments it to the desired sensor, and converts it to a 
dataframe, along with an additional column that will help distinguish it by subject when it is
merged with the rest of the sensor
"""


def load_pkl(file_name, s_id, type=CHEST):
    with open(file_name, "rb") as file:
        pkl = pd.read_pickle(file)
        data = pkl[SIGNAL]
        data_sensor = data[type]

        for metric, value in data_sensor.items():
            data_sensor[metric] = value.tolist()

        df = pd.DataFrame(data_sensor)
        df[SUBJECT] = [s_id for i in range(len(df))]
        #df[sensor] = [type for i in range(len(df))]
    return df


chest_df = load_pkl("S2/S2.pkl", "S2")

for s in subjects[1:]:
    sid = "S" + str(s)
    curr_df = load_pkl(sid + "/" + sid + ".pkl", "S" + str(s))
    chest_df = chest_df.append(curr_df)


chest_df.to_csv("merged_chest.csv")


