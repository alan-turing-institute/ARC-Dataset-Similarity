import json

from dataset_similarity.data.domainnet import DomainNetDataset

with open("data/DomainNet/label_names.json") as f:
    label_names = json.load(f)

data = DomainNetDataset(
    data_root="data/DomainNet",
    domain="real",
    split="train",
)

item = next(iter(data))

print(item)
