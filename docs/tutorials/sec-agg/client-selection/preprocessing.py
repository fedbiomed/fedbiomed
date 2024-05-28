import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import os

FEDBIOMED_DIR = os.environ.get("FEDBIOMED_DIR")
FOLDER_PATH_DATASET = f"{FEDBIOMED_DIR}/data/replace-bg"  # insert you dataset path
REPLACE_DIR = f"{FOLDER_PATH_DATASET}/REPLACE-BG\\ Dataset-79f6bdc8-3c51-4736-a39f-c4c0f71d45e5/Data\\ Tables"
FOLDER_PATH_ORIGINAL = f"{FOLDER_PATH_DATASET}/original"
FOLDER_PATH_RAW = f"{FOLDER_PATH_DATASET}/raw"
FOLDER_PATH_PATIENTS = f"{FOLDER_PATH_RAW}/patients"

# write os commands to create original, raw and patients folders
os.makedirs(FOLDER_PATH_ORIGINAL, exist_ok=True)
os.makedirs(FOLDER_PATH_RAW, exist_ok=True)
os.makedirs(FOLDER_PATH_PATIENTS, exist_ok=True)

# copy from REPLACE_DIR HDeviceBolus.txt, HDeviceCGM.txt, HDeviceWizard.txt to FOLDER_PATH_ORIGINAL
# write os commands to copy files
os.system(f"cp {REPLACE_DIR}/HDeviceBolus.txt {FOLDER_PATH_ORIGINAL}")
os.system(f"cp {REPLACE_DIR}/HDeviceCGM.txt {FOLDER_PATH_ORIGINAL}")
os.system(f"cp {REPLACE_DIR}/HDeviceWizard.txt {FOLDER_PATH_ORIGINAL}")

#delete the REPLACE_DIR
os.system(f"rm -rf {FOLDER_PATH_DATASET}/REPLACE-BG\\ Dataset-79f6bdc8-3c51-4736-a39f-c4c0f71d45e5")

# Read the text file
bolus = pd.read_csv(f"{FOLDER_PATH_ORIGINAL}/HDeviceBolus.txt", sep="|")
cgm = pd.read_csv(f"{FOLDER_PATH_ORIGINAL}/HDeviceCGM.txt", sep="|")
wizard = pd.read_csv(f"{FOLDER_PATH_ORIGINAL}/HDeviceWizard.txt", sep="|")

bolus_columns = ["PtID", "SiteID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "Normal"]
bolus = bolus[bolus_columns]
bolus.to_csv(f"{FOLDER_PATH_RAW}/bolus.csv", index=False)

cgm_columns = ["PtID", "SiteID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "GlucoseValue"]
cgm = cgm[cgm_columns]
cgm.to_csv(f"{FOLDER_PATH_RAW}/cgm.csv", index=False)

wizard.rename(columns={"PtId": "PtID"}, inplace=True)
wizard_columns = ["PtID", "SiteID", "DeviceDtTmDaysFromEnroll", "DeviceTm", "CarbInput"]
# wizard = wizard[wizard_columns]
wizard.to_csv(f"{FOLDER_PATH_RAW}/wizard.csv", index=False)

cgm = pd.read_csv(f"{FOLDER_PATH_RAW}/cgm.csv")
cgm["time"] = (
    pd.to_datetime("2000-01-01")
    + pd.to_timedelta(cgm["DeviceDtTmDaysFromEnroll"], unit="d")
    + pd.to_timedelta(cgm["DeviceTm"])
)
cgm = cgm.drop(columns=["DeviceDtTmDaysFromEnroll", "DeviceTm", "SiteID"])

# compute the difference between two consecutive time for each patient
cgm["day"] = cgm["time"].dt.date
cgm.sort_values(by=["PtID", "time"], inplace=True)
cgm["time_diff"] = cgm.groupby(["PtID", "day"])["time"].diff()
cgm["time_diff"] = cgm["time_diff"].dt.total_seconds() / 60
cgm["time_diff"].fillna(0, inplace=True)

max_time_diff_per_patient_per_day = cgm.groupby(["PtID", "day"])["time_diff"].max()
cgm = cgm.merge(max_time_diff_per_patient_per_day, on=["PtID", "day"], suffixes=("", "_max"))
cgm = cgm[cgm["time_diff_max"] < 60]

# count the number of cgm per day
cgm_count = cgm.groupby(["PtID", "day"]).size().reset_index(name="count")
cgm = cgm.merge(cgm_count, on=["PtID", "day"], suffixes=("", "_count"))
cgm = cgm[cgm["count"] > 1]

cgm.drop(columns=["time_diff", "day", "time_diff_max", "count"], inplace=True)

wizard = pd.read_csv(f"{FOLDER_PATH_RAW}/wizard.csv")
wizard["time"] = (
    pd.to_datetime("2000-01-01")
    + pd.to_timedelta(wizard["DeviceDtTmDaysFromEnroll"], unit="d")
    + pd.to_timedelta(wizard["DeviceTm"])
)
wizard = wizard.drop(columns=["DeviceDtTmDaysFromEnroll", "DeviceTm", "SiteID"])

bolus = pd.read_csv(f"{FOLDER_PATH_RAW}/bolus.csv")
bolus["time"] = (
    pd.to_datetime("2000-01-01")
    + pd.to_timedelta(bolus["DeviceDtTmDaysFromEnroll"], unit="d")
    + pd.to_timedelta(bolus["DeviceTm"])
)
bolus = bolus.drop(columns=["DeviceDtTmDaysFromEnroll", "DeviceTm", "SiteID"])

df = pd.concat([cgm, wizard, bolus], axis=0)
start_time = cgm["time"].min()
end_time = cgm["time"].max()
df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
df.set_index("time", inplace=True)

df_resampled = df.groupby("PtID").resample("15min").mean()
df_resampled.drop(columns=["PtID"], inplace=True)
len(df_resampled)

nan_glucose_pre = df_resampled["GlucoseValue"].isna().sum()
print(f"Number of NaN in GlucoseValue before filling: {nan_glucose_pre}")

df_resampled["GlucoseValue"] = df_resampled["GlucoseValue"].interpolate(
        method="linear", limit_direction="both", limit_area="inside", limit=4
    )

nan_glucose_post = df_resampled["GlucoseValue"].isna().sum()
print(f"Number of NaN in GlucoseValue after filling: {nan_glucose_post}")
# drop all rows with GlucoseValue missing
df_resampled.dropna(subset=["GlucoseValue"], inplace=True)
df_resampled.reset_index(inplace=True)
df_resampled.fillna(0, inplace=True)
print("Total number of samples after resampling: ", len(df_resampled))
df_resampled["day"] = df_resampled["time"].dt.date
df_resampled_count = df_resampled.groupby(["PtID", "day"]).size().reset_index(name="count_day")
df_resampled = df_resampled.merge(df_resampled_count, on=["PtID", "day"], suffixes=("", "_count_day"))
df_resampled = df_resampled[df_resampled["count_day"] >= 18]
df_resampled_count_normal = df_resampled.groupby(["PtID", "day"])["Normal"].sum().reset_index(name="count_normal")
df_resampled = df_resampled.merge(df_resampled_count_normal, on=["PtID", "day"], suffixes=("", "_count_normal"))
df_resampled = df_resampled[df_resampled["count_normal"] > 0]
df_resampled_count_carb = df_resampled.groupby(["PtID", "day"])["CarbInput"].sum().reset_index(name="count_carb")
df_resampled = df_resampled.merge(df_resampled_count_carb, on=["PtID", "day"], suffixes=("", "_count_carb"))
df_resampled = df_resampled[df_resampled["count_carb"] > 0]
print("Total number of samples after removing the day inconplete: ", len(df_resampled))
df_resampled.to_csv(f"{FOLDER_PATH_RAW}/all.csv", index=False)
for patient in df_resampled["PtID"].unique():
    patient_df = df_resampled[df_resampled["PtID"] == patient]
    patient_df.loc[
        :, ["time", "GlucoseValue", "Normal", "CarbInput", "count_day", "count_normal", "count_carb"]
    ].to_csv(f"{FOLDER_PATH_PATIENTS}/{patient}.csv", index=False)