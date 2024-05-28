"""
다산 콜센터 데이터, 금융/보험 데이터 추출 후 데이터 프레임으로 변환
Data :
train용 데이터 -> Train, Validation 용 데이터
validation용 데이터 -> Test 용 데이터
"""

import json
import os
import pandas as pd

def collect_data(DIRS) :
    json_data = []
    for DIR in DIRS :
        for json_file in os.listdir(DIR) : 
            data = json.load(open(DIR + json_file))
            json_data.extend(data)
    return json_data

def utterance_transformation(data) :
    utterance = ''
    if data["고객질문(요청)"] or data["고객답변"] :
        utterance = f'고객 : {data["고객질문(요청)"]}' if data["고객질문(요청)"] else f'고객 : {data["고객답변"]}'
    elif data["상담사질문(요청)"] or data["상담사답변"] :
        utterance = f'상담사 : {data["상담사질문(요청)"]}' if data["상담사질문(요청)"] else f'상담사 : {data["상담사답변"]}'
    return utterance

def drop_duplicates(df) :
    all_dialogue = {}
    mapping = {"코로나19 관련 상담" : 0,"일반행정 문의" : 1,"생활하수도 관련 문의" : 2,"대중교통 안내" : 3, "사고 및 보상 문의" : 4, "상품 가입 및 해지" : 5, "이체, 출금, 대출서비스" : 6, "잔고 및 거래내역" : 7}
    for _, row in df.iterrows() :
        domain_, category_, dialogue_number_, dialogue_ = row
        key = str(dialogue_)
        try :
            all_dialogue[key]["cnt"][mapping[category_]] += 1
        except :
            all_dialogue[key] = {"cnt" : [0 for _ in range(len(mapping.keys()))], "domain" : domain_, "dialogue" : dialogue_, "dialogue_number" : dialogue_number_}
            all_dialogue[key]["cnt"][mapping[category_]] += 1
    
    reverse_mapping = list(mapping.keys())
    domain, category, dialogue, dialogue_number = [], [], [], []
    for row in all_dialogue.values() :
        cnt = row["cnt"]
        domain_ = row["domain"]
        dialogue_ = row["dialogue"]
        dialogue_number_ = row["dialogue_number"]

        domain.append(domain_)
        category.append(reverse_mapping[cnt.index(max(cnt))])
        dialogue.append(dialogue_)
        dialogue_number.append(dialogue_number_)
    df = pd.DataFrame({"domain" : domain, "category" : category, "dialogue_number" : dialogue_number, "dialogue" : dialogue})
    return df

def make_df(json_data) :
    domain, category, dialogue_number, dialogue = [], [], [], []
    dialogue_per_dn = []
    prev_dialogue_number = -1

    for data in json_data :
        domain_, category_, dialogue_number_, dialogue_ = data["도메인"], data["카테고리"], data["대화셋일련번호"], utterance_transformation(data)
        if not dialogue_ : continue
        if prev_dialogue_number != dialogue_number_:
            domain.append(domain_)
            category.append(category_)
            dialogue_number.append(dialogue_number_)
            if prev_dialogue_number != -1 :
                dialogue.append(dialogue_per_dn)
            dialogue_per_dn = [dialogue_]
            prev_dialogue_number = dialogue_number_
        else :
            dialogue_per_dn.append(dialogue_)
    else : dialogue.append(dialogue_per_dn)
    df = pd.DataFrame({"domain" : domain, "category" : category, "dialogue_number" : dialogue_number, "dialogue" : dialogue})
    df = drop_duplicates(df)
    return df

os.chdir("..")
DATA_DIR = "data/"
RAW_DIR = DATA_DIR + "raw/"
TRAIN_DIRS = [f"{RAW_DIR}1.Training/라벨링데이터_220121_add/다산콜센터/", f"{RAW_DIR}1.Training/라벨링데이터_220121_add/금융보험/"]
VALID_DIRS = [f"{RAW_DIR}2.Validation/라벨링데이터_220121_add/다산콜센터/", f"{RAW_DIR}2.Validation/라벨링데이터_220121_add/금융보험/"]
DF_DIR = DATA_DIR + "df"

train_json_data = collect_data(TRAIN_DIRS)
valid_json_data = collect_data(VALID_DIRS) 

train_df = make_df(train_json_data)
test_df = make_df(valid_json_data)

train_df.to_csv(f'{DF_DIR}/train_df.csv', index=False)
test_df.to_csv(f'{DF_DIR}/test_df.csv', index=False)