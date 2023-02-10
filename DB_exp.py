import gzip
import json
import html
import pandas as pd


dataset = []

with gzip.open("dataset/qa_Video_Games.json.gz", "rb") as fl:
    for line in fl:
        try:
            processed_line = html.unescape(line.strip().decode()).replace('\"', '').replace('\'', '\"')
            dataset.append(json.loads(processed_line))
        except Exception as e:
            print(e)

dataframe = pd.DataFrame.from_records(dataset)
dataframe.to_parquet("dataset/qa_VG.parquet")
