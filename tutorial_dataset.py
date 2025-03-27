import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset_3(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/' + source_filename, cv2.IMREAD_COLOR)
        target = cv2.imread('./training/' + target_filename, cv2.IMREAD_COLOR)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class MyDataset_1(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source2']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/' + source_filename, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread('./training/' + target_filename, cv2.IMREAD_COLOR)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source = np.expand_dims(source, axis=-1)  # Add a channel dimension at the end

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class MyDataset_4(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        source2_filename = item['source2']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/' + source_filename, cv2.IMREAD_COLOR)
        source2 =cv2.imread('./training/' + source2_filename,cv2.IMREAD_GRAYSCALE)
        target = cv2.imread('./training/' + target_filename, cv2.IMREAD_COLOR)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source2= source2.astype(np.float32)/ 255.0
        source2 = np.expand_dims(source2, axis=-1)

        concat_source = np.concatenate((source, source2), axis=-1)
        
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=concat_source)

