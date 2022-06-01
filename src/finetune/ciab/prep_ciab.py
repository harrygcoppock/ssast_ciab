# -*- coding: utf-8 -*-
# @Author  : Harry Coppock
# @Email   : harry.coppock@imperial.ac.uk
# @File    : prep_ciab.py

import numpy as np
import json
import os
import zipfile
import wget
from sklearn.model_selection import KFold, train_test_split

import pandas as pd
import subprocess, glob, csv
from tqdm import tqdm
import pickle
from botocore import UNSIGNED
from botocore.config import Config
import io
import boto3
import librosa, librosa.display
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import soundfile as sf

class PrepCIAB():
    POSSIBLE_MODALITIES = ['audio_sentence_url',
                           'audio_ha_sound_url',
                           'audio_cough_url',
                           'audio_three_cough_url']

    PATHS = {
            'meta': 'train-test-split/final_objects/meta_data_v5.csv',
            'meta_bucket': 'ciab-879281191186-prod-s3-pii-ciab-wip',
            'audio_bucket': 'ciab-879281191186-prod-s3-pii-ciab-approved',
            }
    RANDOM_SEED = 42

    def __init__(self, modality='audio_three_cough_url'):
        self.modality = self.check_modality(modality)
        self.bucket_meta = self.get_bucket(self.PATHS['meta_bucket'])
        self.bucket_audio = self.get_bucket(self.PATHS['audio_bucket'])
        self.meta = pd.read_csv(self.get_file(
                                            self.PATHS['meta'],
                                            self.bucket_meta))
        self.load_splits()
        # base directory for audio files
        self.output_base= f'./data/ciab/final/{self.modality}'
        self.error_list = []

    def main(self):
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
            
        print('Beginining ciab train prepocessing')
        self.iterate_through_files(self.train, 'train')
        print('Beginining ciab validation prepocessing')
        self.iterate_through_files(self.val, 'val')
        print('Beginining ciab test prepocessing')
        self.iterate_through_files(self.test, 'test')
        print('Beginining ciab long test prepocessing')
        self.iterate_through_files(self.long, 'long_test') 
        print('Beginining ciab matched test prepocessing')
        self.iterate_through_files(self.matched_test, 'matched_test') 
        print('Beginining ciab matched_train prepocessing')
        self.iterate_through_files(self.matched_train, 'matched_train')
        print('Beginining ciab matched_validation prepocessing')
        self.iterate_through_files(self.matched_validation, 'matched_validation')
        print('Beginining ciab naive train prepocessing')
        self.iterate_through_files(self.naive_train, 'naive_train')
        print('Beginining ciab naive validation prepocessing')
        self.iterate_through_files(self.naive_validation, 'naive_validation')
        print('Beginining ciab naive test prepocessing')
        self.iterate_through_files(self.naive_test, 'naive_test')
        print('Beginining ciab original train prepocessing')
        self.iterate_through_files(self.train_original, 'train_original')
        print('Beginining ciab original test prepocessing')
        self.iterate_through_files(self.test_original, 'test_original')
        print('Beginining ciab matched test prepocessing')
        self.iterate_through_files(self.matched_test_original, 'matched_test_original') 
        print('Beginining ciab matched_train prepocessing')
        self.iterate_through_files(self.matched_train_original, 'matched_train_original')
        print('Beginining ciab matched_validation prepocessing')
        self.iterate_through_files(self.matched_validation_original, 'matched_validation_original')
        with open(f'{self.output_base}/audio_16k/errorlist.txt', "w") as output:
            output.write(str(self.error_list))
        print('creating json')
        self.create_json()

    def check_modality(self, modality):
        if modality not in self.POSSIBLE_MODALITIES:
            raise Exception(f"{modaliity} is not one of the recorded functionalities,\
                                 please choose from {self.POSSIBLE_MODALITIES}")
        else:
            return modality


    def get_bucket(self, bucket_name):
        s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED), region_name='eu-west-2')
        return s3_resource.Bucket(bucket_name)


    def get_file(self, path, bucket):
        return io.BytesIO(bucket.Object(path).get()['Body'].read())

    def load_splits(self):
        '''
        Loads the train and test barcode splits and the corresponding meta_data
        '''
        self.train = self.meta[self.meta['splits'] == 'train'].audio_sentence.tolist()
        self.val = self.meta[self.meta['splits'] == 'val'].audio_sentence.tolist()
        self.test = self.meta[self.meta['splits'] == 'test'].audio_sentence.tolist()
        self.long = self.meta[self.meta['splits'] == 'long'].audio_sentence.tolist()

        self.matched_train = self.meta[self.meta['matched_train_splits'] == 'matched_train'].audio_sentence.tolist()
        self.matched_validation = self.meta[self.meta['matched_train_splits'] == 'matched_validation'].audio_sentence.tolist()
        self.matched_test = self.meta[self.meta['in_matched_rebalanced_test'] == True].audio_sentence.tolist()

        self.naive_train = self.meta[self.meta['naive_splits'] == 'train'].audio_sentence.tolist()
        self.naive_validation = self.meta[self.meta['naive_splits'] == 'validation'].audio_sentence.tolist()
        self.naive_test = self.meta[self.meta['naive_splits'] == 'test'].audio_sentence.tolist()

        self.train_original = self.meta[self.meta['original_splits'] == 'train'].audio_sentence.tolist()
        self.val_original = self.meta[self.meta['original_splits'] == 'val'].audio_sentence.tolist()
        self.test_original = self.meta[self.meta['original_splits'] == 'test'].audio_sentence.tolist()

        self.matched_train_original = self.meta[self.meta['matched_original_train_splits'] == 'matched_train'].audio_sentence.tolist()
        self.matched_validation_original = self.meta[self.meta['matched_original_train_splits'] == 'matched_validation'].audio_sentence.tolist()
        self.matched_test_original = self.meta[self.meta['in_matched_original_test'] == True].audio_sentence.tolist()

    def iterate_through_files(self, dataset, split='train'):
        if not os.path.exists(f'{self.output_base}/audio_16k/{split}'):
            os.makedirs(f'{self.output_base}/audio_16k/{split}')
        self.tot_removed = [] 
        bootstrap_results = Parallel(n_jobs=-1, verbose=10, prefer='threads')(delayed(self.process_file)(barcode_id, split) for barcode_id in dataset)
        
        print(f'Average fraction removed: {np.mean(self.tot_removed)}')
        

    def process_file(self, barcode_id, split):

        df_match = self.meta[self.meta['audio_sentence'] == barcode_id]
        assert len(df_match) != 0, 'This unique code does not exist in the meta data file currently loaded - investigate!'

        if df_match['audio_sentence_size'].iloc[0] <= 44 or df_match['audio_ha_sound_size'].iloc[0] <= 44 or df_match['audio_cough_size'].iloc[0] <= 44 or df_match['audio_three_cough_size'].iloc[0] <= 44:
            self.error_list.append(df_match[self.modality].iloc[0])
            return 1

        if df_match['test_result'].iloc[0] == 'Unknown/Void':
            self.error_list.append(df_match[self.modality].iloc[0])
            return 1
        try:
            filename = self.get_file(df_match[self.modality].iloc[0], self.bucket_audio)
        except:
            print(f"{df_match[self.modality].iloc[0]} not possible to load. From {df_match['processed_date']} Total so far: {len(self.error_list)}")
            self.error_list.append(df_match[self.modality].iloc[0])
            return 1
        label = df_match['test_result'].iloc[0]
        try:
            signal, sr = librosa.load(filename, sr=16000)
        except:
            print(f"{filename} not possible to load. From {df_match['processed_date']} Total so far: {len(self.error_list)}")
            self.error_list.append(df_match[self.modality].iloc[0])
            return 1
        clipped_signal, frac_removed = self.remove_silence(signal, barcode_id)
        self.tot_removed.append(frac_removed)
        sf.write(f'{self.output_base}/audio_16k/{split}/{barcode_id}', clipped_signal, 16000)
        return 1

    def print_stats(self):
        print(f'Sample numbers: Train: {len(self.train)}, \
                Validation: {len(self.val)} \
                Test: {len(self.test)}, \
                Long_test: {len(self.long)} \
                matched_test: {len(self.matched_test)} \
                matched_train: {len(self.matched_train)} \
                matched_val: {len(self.matched_validation)} \
                naive train: {len(self.naive_train)} \
                naive validation: {len(self.naive_validation)} \
                naive test: {len(self.naive_test)}')
    
    def create_json(self):
        fold = 1
        dic_train_list = self.list_to_dict(self.train, 'train')
        dic_validation_list = self.list_to_dict(self.val, 'validation')
        dic_test_list = self.list_to_dict(self.test, 'test')
        dic_matched_test_list = self.list_to_dict(self.matched_test, 'matched_test')
        dic_matched_train_list = self.list_to_dict(self.matched_train, 'matched_train')
        dic_matched_validation_list = self.list_to_dict(self.matched_validation, 'matched_validation')
        dic_long_test_list = self.list_to_dict(self.long, 'long_test')
        dic_naive_train_list = self.list_to_dict(self.naive_train, 'naive_train')
        dic_naive_validation_list = self.list_to_dict(self.naive_validation, 'naive_validation')
        dic_naive_test_list = self.list_to_dict(self.naive_test, 'naive_test')

        dic_train_original_list = self.list_to_dict(self.train_original, 'train_original')
        dic_validation_original_list = self.list_to_dict(self.val_original, 'train_original')
        dic_test_original_list = self.list_to_dict(self.test_original, 'test_original')
        dic_matched_test_original_list = self.list_to_dict(self.matched_test_original, 'matched_test_original')
        dic_matched_train_original_list = self.list_to_dict(self.matched_train_original, 'matched_train_original')
        dic_matched_validation_original_list = self.list_to_dict(self.matched_validation_original, 'matched_validation_original')

        with open(f'./data/datafiles/{self.modality}/ciab_train_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_train_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_validation_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_validation_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_long_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_long_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_matched_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/dic_matched_validation_list'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_validation_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/dic_matched_train_list'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_train_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/naive_train_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_naive_train_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/naive_validation_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_naive_validation_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/naive_test_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_naive_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_train_original_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_train_original_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_validation_original_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_validation_original_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_test_original_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_test_original_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_matched_test_original_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_test_original_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/dic_matched_validation_original_list'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_validation_original_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/dic_matched_train_original_list'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_train_original_list}, f, indent=1)
    
    
    
    def list_to_dict(self, data, split):
        '''
        THe ssast library requires a json file in the following format
         {
            "data": [
                {
                    "wav": "/data/sls/audioset/data/audio/eval/_/_/--4gqARaEJE_0.000.flac",
                    "labels": "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
                },
                {
                    "wav": "/data/sls/audioset/data/audio/eval/_/_/--BfvyPmVMo_20.000.flac",
                    "labels": "/m/03l9g"
                },
              // ... many audio files
                {
                    "wav": "/data/sls/audioset/data/audio/eval/_/0/-0BIyqJj9ZU_30.000.flac",
                    "labels": "/m/07rgt08,/m/07sq110,/t/dd00001"
                }
            ]
        }
        '''
        #note the additional check for checking if the audio file is faulty. These samples are included in the exp
        # as participants submitted 4 recordings and we do not disregard them even if one recording is missing.
        if len(self.error_list) == 0:
            formatted_list = [{"wav": f'{self.output_base}/audio_16k/{split}/{instance}', "labels": self.meta[self.meta['audio_sentence'] == instance].test_result.iloc[0]} for instance in data]
            return formatted_list
        error_list = [self.meta[self.meta[self.modality] == instance].audio_sentence.iloc[0] for instance in self.error_list]
        formatted_list = [{"wav": f'{self.output_base}/audio_16k/{split}/{instance}', "labels": self.meta[self.meta['audio_sentence'] == instance].test_result.iloc[0]} for instance in data if instance not in error_list]
        return formatted_list

    def remove_silence(self, signal, filename):
        '''
        Removes the silent proportions of the signal, concatenating the remaining clips
        '''
        length_prior = len(signal)
        clips = librosa.effects.split(signal, top_db=60)

        clipped_signal = []
        for clip in clips:
            data = signal[clip[0]:clip[1]]
            clipped_signal.extend(data)
        length_post = len(clipped_signal)
        
        random_number = np.random.uniform(0,1,1)
        #hacky way to avoid different plots being assigned to the same fig instance when in parrallel
        #if random_number[0] < 0.1:
            #self.plot_b_a(signal, np.array(clipped_signal), filename)

        return np.array(clipped_signal), (length_prior - length_post)/(length_prior + 0.0000000000001)

    def plot_b_a(self, before, after, filename):
        '''
        plot the waveform before and after the silence is removed
        '''
        fig, ax = plt.subplots(nrows=2)
        librosa.display.waveshow(before, sr=16000, ax=ax[0])
        librosa.display.waveshow(after, sr=16000, ax=ax[1])
        ax[0].set(title='HOw much we remove')
        plt.savefig(f'figs/{filename}.png')
        plt.close()

    def create_naive_splits(self, data, just_val=False):
        '''
        given a list of ids of all the available data randomly create train/val/splits
        '''
        train_X, dev_test_X = train_test_split(
                    data,
                    test_size=0.3 if not just_val else 0.2,
                    random_state=self.RANDOM_SEED)
        if just_val:
            return train_X, dev_test_X
        devel_X, test_X = train_test_split(
                    dev_test_X,
                    test_size=0.5,
                    random_state=self.RANDOM_SEED)

        return train_X, devel_X, test_X

    def back_to_list(self, list_dic):
        '''
        for some tests it required to convert back from a list of dicts to simly as list
        '''
        return [x['wav'] for x in list_dic]

if __name__ == '__main__':
    for modality in ['audio_sentence_url']:#,'audio_ha_sound_url', 'audio_cough_url', 'audio_three_cough_url']:


        ciab = PrepCIAB(modality)
        ciab.main()
        ciab.print_stats()
