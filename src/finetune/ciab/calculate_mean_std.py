
# -*- coding: utf-8 -*-
# @Author  : Harry Coppock
# @Email   : harry.coppock@imperial.ac.uk
# @File    : calculate_mean_std.py
import sys
sys.path.append('../../')
from dataloader import AudioDataset
import torch
import numpy as np
from tqdm import tqdm

def calc_mean_std(dataset):
    psum = 0.0
    psum_sq = 0.0
    count = 0
    for i in tqdm(range(len(dataset))):
        fbank = dataset[i][0]
        psum += fbank.sum()
        psum_sq += (fbank**2).sum()
        count += fbank.numel()
    dataset_mean = psum/count
    dataset_var = (psum_sq / count) - (dataset_mean ** 2)
    dataset_std = np.sqrt(dataset_var)
    return dataset_mean, dataset_std



if __name__ == '__main__':

    audio_conf = {'num_mel_bins': 128, 
            'target_length': 512,
            'freqm': 24,
            'timem': 96,
            'mixup': 0,
            'dataset': 'ciab',
            'mode': 'train',
            'noise': False,
            'skip_norm': True,
            'calc_mean_std': True,
            'mode': 'train'}


    POSSIBLE_MODALITIES = ['audio_sentence_url',
                           'audio_ha_sound_url',
                           'audio_cough_url',
                           'audio_three_cough_url']
    for modality in POSSIBLE_MODALITIES: 
        dataset = AudioDataset(
                f'./data/datafiles/{modality}/ciab_train_data_1.json',
                label_csv='./data/ciab_class_labels_indices.csv',
                audio_conf=audio_conf
                )
        dataset_mean, dataset_std = calc_mean_std(dataset)
        print('modality: ', modality)
        print('mean: ', dataset_mean)
        print('std: ', dataset_std)
