from PIL import Image
import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()

def load_single(sample_id, text, name):
    img_path = f"Images/{name}/{sample_id}.jpg"
    try:
        image = Image.open(img_path).convert("RGB")
        return sample_id, text, image
    except Exception:
        return None


def solve(name, df, batch_size=128, num_workers=32):
    df = df.copy()

    sample_ids = df["sample_id"].tolist()
    texts = df["catalog_content"].astype(str).tolist()

    combined_rows = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_ids = sample_ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        # 🔥 Load only this batch (parallel)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda x: load_single(x[0], x[1], name),
                zip(batch_ids, batch_texts)
            ))

        # filter valid
        valid = [r for r in results if r is not None]
        if len(valid) == 0:
            continue

        ids = [x[0] for x in valid]
        texts_batch = [x[1] for x in valid]
        images = [x[2] for x in valid]

        inputs = processor(
            text=texts_batch,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            text_emb = outputs.text_embeds
            image_emb = outputs.image_embeds
            combined = torch.cat([image_emb, text_emb], dim=1)

        combined = combined.cpu().numpy()

        for j, sid in enumerate(ids):
            row = {"sample_id": sid}
            for k, val in enumerate(combined[j]):
                row[f"emb_{k}"] = val
            combined_rows.append(row)

        # 🔥 Explicit cleanup (important)
        del inputs, outputs, combined
        torch.cuda.empty_cache()

    emb_df = pd.DataFrame(combined_rows)
    out = df.merge(emb_df, on="sample_id", how="left")

    os.makedirs(f"./archive/embedding/{name}", exist_ok=True)
    out.to_csv(f"./archive/embedding/{name}/embedded.csv", index=False)

    return out


train = pd.read_csv("./archive/train.csv")
# test = pd.read_csv("./archive/test.csv")
# sample_test = pd.read_csv("./archive/sample_test.csv")
solve("train",train)