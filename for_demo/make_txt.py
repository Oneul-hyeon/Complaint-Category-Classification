import os
import pandas as pd
from ast import literal_eval

os.chdir("..")
TEST_DATA_DIR = os.getcwd() + "/data/df/test_df.csv"
TXT_DIR = os.getcwd() + "/files_for_test_demo/txt"
test_df = pd.read_csv(TEST_DATA_DIR)
test_df["dialogue"] = test_df.dialogue.apply(lambda x : literal_eval(x))

for _, row in test_df.iterrows() :
    dialogue_number = row["dialogue_number"]
    dialogue = row["dialogue"]
    f = open(f"{TXT_DIR}/{dialogue_number}.txt", "w")
    if not dialogue[-1] : dialogue = dialogue[:-1]
    f.write('\n'.join(dialogue))
    f.close()