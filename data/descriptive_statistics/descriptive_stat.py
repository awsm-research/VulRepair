import pandas as pd
import numpy as np

df = pd.read_csv("../fine_tune_data/cve_fixes_whole.csv")
source_length = []
target_length = []
for i in range(len(df)):
    source_length.append(len(str(df["source"][i]).split(" ")))
    target_length.append(len(str(df["target"][i]).split(" ")))

assert len(source_length) == len(target_length) == len(df) 

source_min = min(source_length)
source_25 = np.quantile(source_length, 0.25)
source_median = np.quantile(source_length, 0.5)
source_75 = np.quantile(source_length, 0.75)
source_max = max(source_length)
source_avg = round(sum(source_length)/len(source_length), 2)

target_min = min(target_length)
target_25 = np.quantile(target_length, 0.25)
target_median = np.quantile(target_length, 0.5)
target_75 = np.quantile(target_length, 0.75)
target_max = max(target_length)
target_avg = round(sum(target_length)/len(target_length), 2)

print(df.columns)
print("Total data points: ", len(df))

print("Source")
print("Min: ", source_min)
print("25th Quantile: ", source_25)
print("Median Quantile: ", source_median)
print("75th Quantile: ", source_75)
print("Max: ", source_max)
print("Avg: ", source_avg)

print("Target")
print("Min: ", target_min)
print("25th Quantile: ", target_25)
print("Median Quantile: ", target_median)
print("75th Quantile: ", target_75)
print("Max: ", target_max)
print("Avg: ", target_avg)