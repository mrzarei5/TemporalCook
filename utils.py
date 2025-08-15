import json

from PIL import Image
import requests
import os
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        image = Image.open(requests.get(image_file, stream=True).raw)
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def write_output(id, data, save_path):
    export_dic = {}
    path = os.path.join(save_path, 'results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            export_dic = json.load(f)
    
    export_dic[id] = data
    

    with open(path, "w") as outfile:
        json.dump(export_dic, outfile, indent = 2) 