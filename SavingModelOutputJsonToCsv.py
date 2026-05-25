import json
import pandas as pd
from collections import defaultdict


def process_dict(results):
    rows = []
    for sampleid, jsondata in results:
        row = {"sample_id": sampleid}
        jsondata = jsondata.dict()
        for key, value in jsondata.items():
            if isinstance(value, list):
                if len(value)>=3:
                    a,b,c = val[:3]
                elif len(value)==2:
                    a,b,c = value[0],value[1],None
                elif len(value)==1:
                    a,b,c = value[0],None,None
                else:
                    a,b,c = None,None,None
                row[f"{key}_0"] = a
                row[f"{key}_1"] = b
                row[f"{key}_2"] = c
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)
    


def solve(ranges, name, df):
    for start, end in ranges:
        batch_df = df.iloc[start:end].copy()
        results = []
        for _, row in batch_df.iterrows():
            sample_id = row["sample_id"]
            catalog_content = row["catalog_content"] #Images/train/
            image_path = f"./Images/{name}/{sample_id}.jpg"
            result = extract_and_validate(image_path,catalog_content)
            print(result)
            results.append((sample_id, result))
        result_df = process_dict(results)
        merged_df = batch_df.merge(result_df, on="sample_id", how="left")
        merged_df.to_csv(f"./archive/{name}/{start}-{end}.csv", index=False)


# df = pd.read_csv("./archive/train.csv")
# lst = np.linspace(start=0, stop=75000, num=7500, endpoint=True)
# step = 1
# ranges = [(i, min(i + step, int(lst[-1]))) for i in range(0, int(lst[-1]), step)]
# ranges[0]
# solve(ranges[:5],"train",df)