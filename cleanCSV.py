import csv
import pandas as pd

FILE = "hand_landmarks.csv"

# oneHand = []
twoHand = ["A", "B", "D", "E", "F", "F", "G", "H", "J", "K", "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"]

data = []

with open(FILE, "r") as csvFile:
    dataReader = csv.reader(csvFile)
    for i in dataReader:
        label = i[0]
        landmarks = i[1:]
        if label in twoHand:
            if landmarks.count("0.0") > 1:
                # print(label)
                # print(landmarks)
                continue
        
        if landmarks.count("0.0") == 84:
            continue
        
        data.append(i)

df = pd.DataFrame(data[1:], columns=data[0])
df.to_csv("clean_landmarks.csv", index=False)
print(df)

print(df["label"].value_counts())