import pandas as pd

df = pd.read_csv("./vulrepair_raw_preds_wt_cc.csv")
cc = df["cc"].dropna()
avg_cc = round(sum(cc)/len(cc), 4)
print("Average Cyclomatic Complexity: ", avg_cc)

cc_0_10 = df[df["cc"]<=10]
cc_11_20 = df[(df["cc"]>10) & (df["cc"]<=20)]
cc_21_30 = df[(df["cc"]>20) & (df["cc"]<=30)]
cc_31_40 = df[(df["cc"]>30) & (df["cc"]<=40)]
cc_41 = df[df["cc"]>40]
cc_10_up = df[df["cc"]>10]
cc_20_up = df[df["cc"]>20]
cc_30_up = df[df["cc"]>30]

ppp_0_10 = round(sum(cc_0_10["correctly_predicted"]), 4) /len(cc_0_10)
ppp_11_20 = round(sum(cc_11_20["correctly_predicted"]), 4) /len(cc_11_20)
ppp_21_30 = round(sum(cc_21_30["correctly_predicted"]), 4) /len(cc_21_30)
ppp_31_40 = round(sum(cc_31_40["correctly_predicted"]), 4) /len(cc_31_40)

ppp_41 = round(sum(cc_41["correctly_predicted"]), 4) /len(cc_41)
ppp_10_up = round(sum(cc_10_up["correctly_predicted"]), 4) /len(cc_10_up)
ppp_20_up = round(sum(cc_20_up["correctly_predicted"]), 4) /len(cc_20_up)
ppp_30_up = round(sum(cc_30_up["correctly_predicted"]), 4) /len(cc_30_up)

print("% Perfect Prediction")
print("CC 0-10: ", ppp_0_10)
print("CC 11-20: ", ppp_11_20)
print("CC 21-30: ", ppp_21_30)
print("CC 31-40: ", ppp_31_40)
print("CC >40: ", ppp_41)

"""
print("CC >10: ", ppp_10_up)
print("CC >20: ", ppp_20_up)
print("CC >30: ", ppp_30_up)
"""