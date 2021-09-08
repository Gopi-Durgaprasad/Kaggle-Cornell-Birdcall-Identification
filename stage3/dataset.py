
import random, glob
import numpy as np
import pandas as pd
import soundfile as sf

from torch.utils.data import Dataset
from typing import Optional, List

import torch
from albumentations.pytorch.functional import img_to_tensor

from bird_codes import BIRD_CODE, INV_BIRD_CODE

class PANNsDataset(Dataset):
    def __init__(
        self,
        file_list: List[List[str]],
        period=1,
        transforms=None,
        train=True,
        waveform_transforms=None):

        self.file_list = file_list
        self.period = period
        self.transforms = transforms
        self.train = train
        self.waveform_transforms = waveform_transforms

        self.noise_files = glob.glob("/content/train_stage1_1sec_sudo_noise/tmp/birdsongWav/train_stage1_1sec_sudo_noise/*.wav")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int):
        
        wav_path, ebird_code, pred_prob = self.file_list[idx]

        labels = np.zeros(len(BIRD_CODE), dtype="f")

        frand = np.random.random()
        if self.train:
            if frand > 0.4:
                y, sr = sf.read(wav_path)
                labels[BIRD_CODE[ebird_code]] = 1
            else:
                y, sr = sf.read(random.choice(self.noise_files))
                labels[BIRD_CODE[ebird_code]] = 0
        else:
            y, sr = sf.read(wav_path)
            try:
                ebird_code = ebird_code.split(' ')
                #print(ebird_code)
                for eb in ebird_code:
                    labels[BIRD_CODE[eb]] = 1
            except:
                pass

        

        if not self.train:
            self.period = 5#self.period

        if self.waveform_transforms:
            y = self.waveform_transforms(y, sample_rate=sr)
        else:
            len_y = len(y)
            effective_length = sr * self.period
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)
        
        #if self.transforms:
        #    pass
            #spec, sr = self.transforms(data=(y, sr))['data']
            #spec = np.expand_dims(spec, axis=0) #np.transpose(spec, (2, 0, 1)).astype(np.float32)

        """
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        if self.train:
            #if np.random.random() > 0.5:
            #    labels[BIRD_CODE[ebird_code]] = pred_prob
            #else:
            labels[BIRD_CODE[ebird_code]] = 1
        #labels = BIRD_CODE[ebird_code]
        else:
            ebird_code = ebird_code.split(' ')
            #print(ebird_code)
            for eb in ebird_code:
                labels[BIRD_CODE[ebird_code]] = 1
        """

        return {
            "waveform" : y, #torch.tensor(spec, dtype=torch.float),
            "targets" : labels, #torch.tensor(labels, dtype=torch.long)
        }

    def __get_labels__(self):
        return np.array(self.file_list)[:, 1]
