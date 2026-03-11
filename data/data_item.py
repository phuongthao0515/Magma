import json
import yaml
import torch
import random
import os
import glob
import pickle
from tqdm import tqdm

class DataItem:
    """
    Curate data items from all data sources
    """
    def __init__(self, training_size=-1, local_run=False):
        self.training_size = training_size
        self.local_run = local_run

    def _get_dataset_tag(self, data_path):
        if 'aitw' in data_path.lower():
            return "aitw"
        elif 'seeclick' in data_path.lower() and 'ocr' in data_path.lower():
            return "seeclick_ocr"
        elif 'seeclick' in data_path.lower():
            return "seeclick"
        elif 'mind2web' in data_path.lower():
            return "mind2web"
        elif 'llava' in data_path.lower():
            return "llava"
        elif 'magma' in data_path.lower():
            return "magma"
        elif 'sharegpt4v' in data_path.lower():
            return "sharegpt4v"
        else:
            raise ValueError(f"Dataset tag not found for {data_path}")

    def _get_items(self, data_path, image_folder=None, processor=None, conversation_lib=None):
        if data_path.endswith(".json"):
            list_data_dict = json.load(open(data_path, "r"))
        elif data_path.endswith(".jsonl"):
            list_data_dict = [json.loads(line) for line in open(data_path, "r")]
        elif data_path.endswith(".pth"):
            list_data_dict = torch.load(data_path, map_location="cpu")
        else:
            data_folder = os.path.dirname(data_path)
            # get file name from data_path
            data_files = data_path.split('/')[-1].split('+')
            list_data_dict = []
            for file in data_files:
                json_path = os.path.join(data_folder, file + '.json')
                list_data_dict.extend(json.load(open(json_path, "r")))
        return list_data_dict

    def __call__(self, data_path, processor=None, conversation_lib=None, is_eval=False):
        assert data_path is not None, "Data path is not provided"
        if data_path.endswith(".yaml"):
            data_dict = yaml.load(open(data_path, "r"), Loader=yaml.FullLoader)
            data_path_key = 'DATA_PATH' if not is_eval else 'DATA_PATH_VAL'
            image_folder_key = 'IMAGE_FOLDER' if not is_eval else 'IMAGE_FOLDER_VAL'
            assert len(data_dict[data_path_key]) == len(data_dict[image_folder_key]), "Data path and image folder mismatch"
            items = {}
            dataset_names = []
            dataset_folders = []
            for i, (data_path, image_folder) in enumerate(zip(data_dict[data_path_key], data_dict[image_folder_key])):
                items_temp = self._get_items(data_path, image_folder, processor, conversation_lib)
                dataset_tag = self._get_dataset_tag(data_path)
                # add image_folder and dataset tag to each item
                for item in items_temp:
                    item['image_folder'] = image_folder
                    item['dataset_tag'] = dataset_tag
                if dataset_tag in items:
                    items[dataset_tag].extend(items_temp)
                else:
                    items[dataset_tag] = items_temp
                    dataset_names.append(dataset_tag)
                    dataset_folders.append(image_folder)
        else:
            items = self._get_items(data_path)
            dataset_names = None
            dataset_folders = None
        return items, dataset_names, dataset_folders
