import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_encoding(df) :
    label_encoder = LabelEncoder()
    return_series = label_encoder.fit_transform(df["category"])

    np.save(f"{os.getcwd()}/data/encoderclass/ec", label_encoder.classes_)

    return return_series
def split_dialogue(dialogue) :
    except_colon_dialogue = []
    for utterance in dialogue :
        colon_idx = utterance.index(":")
        utter = utterance[colon_idx+1:].strip()
        except_colon_dialogue.append(utter)
    return ' '.join(except_colon_dialogue)