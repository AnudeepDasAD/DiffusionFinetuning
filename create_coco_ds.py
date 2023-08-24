from clip_benchmark.datasets.builder import build_dataset
import pandas as pd
import os

root_path = "coco-dataset/" # set this to smth meaningful


def load_and_save(built_ds, split):
    coco = built_ds.coco
    imgs = coco.loadImgs(coco.getImgIds())
    future_df = {"filepath":[], "title":[]}
    for img in imgs:
        caps = coco.imgToAnns[img["id"]]
        for cap in caps:
            future_df["filepath"].append(img["file_name"])
            future_df["title"].append(cap["caption"])
    pd.DataFrame.from_dict(future_df).to_csv(
    os.path.join(root_path, f"{split}2014.csv"), index=False, sep="\t"
)

train_ds = build_dataset("mscoco_captions", root=root_path, split="train") # this downloads the dataset if it is not there already
test_ds = build_dataset("mscoco_captions", root=root_path, split="test")

load_and_save(train_ds, split='train')
load_and_save(test_ds, split='test')