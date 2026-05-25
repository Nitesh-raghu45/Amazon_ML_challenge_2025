import os
import pandas as pd
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(link, imgPath):
    try:
        res = requests.get(link, timeout=10)
        res.raise_for_status()
        with open(imgPath, 'wb') as f:
            f.write(res.content)
        return True
    except Exception:
        return False

def DounloadImages(name, csv_path, max_threads=256):
    folder = f'Images/{name}/'
    os.makedirs(folder, exist_ok=True)
    df = pd.read_csv(csv_path)

    tasks = []
    for sample_id, link in zip(df['sample_id'], df['image_link']):
        imgPath = f"{folder}{sample_id}.jpg"
        if not os.path.exists(imgPath):
            tasks.append((link, imgPath))

    print(f"Total images to download: {len(tasks)}")

    success = 0
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [executor.submit(download_image, link, path) for link, path in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            if f.result():
                success += 1

    print("Successfully downloaded:", success)
DounloadImages("train", "./archive/train.csv")
DounloadImages("test", "./archive/test.csv")
DounloadImages("sample_test", "./archive/sample_test.csv")
