# -*- coding: utf-8 -*-

import json
import os
import csv


file_path = "./cluener_public"
mode_class = [
    "train",
    "dev"
]
output_path = "t5_ner_data"
if not os.path.exists(output_path):
    os.makedirs(output_path)

SEPARATOR = " | "

def parse(data):
    text = data["text"]
    entity_txt = []
    for type, entity_list in data["label"].items():
        entity_list = list(map(lambda x: x.strip().strip("《").strip("》"), entity_list.keys()))
        type_entity = f"{type}:{','.join(entity_list)}"
        entity_txt.append(type_entity)
    return text, SEPARATOR.join(entity_txt)


for mode in mode_class:
    data = []
    file = os.path.join(file_path, mode + ".json")
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            data.append(line)
    output_data = os.path.join(output_path, mode + ".tsv")
    with open(output_data, "w", encoding="utf-8", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        for line in data:
            text, entity_txt = parse(line)
            writer.writerow([text, entity_txt])