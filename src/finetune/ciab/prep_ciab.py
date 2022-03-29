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
            'pre_june_meta': 'combined-results/submissions/study_data_v6_pre_jun_11012022.pkl',
            'post_june_meta': 'combined-results/submissions/study_data_v6_post_jun_11012022.pkl',
            'meta_bucket': 'ciab-879281191186-prod-s3-pii-ciab-wip',
            'audio_bucket': 'ciab-879281191186-prod-s3-pii-ciab-approved',
            'splits': 'train-test-split/train_test_split_final_audiosentence_v6_final.pkl',
            'matched': 'audio_sentences_for_matching/test_set_matched_audio_sentences_v6_final.csv',
            'matched_train': 'audio_sentences_for_matching/train_set_matched_audio_sentences_v6_final.csv',
            'longitudinal': 'train-test-split/LongitudinalTestSet.csv'
            }
    RANDOM_SEED = 42

    def __init__(self, modality='audio_three_cough_url'):
        self.modality = self.check_modality(modality)
        self.bucket_meta = self.get_bucket(self.PATHS['meta_bucket'])
        self.bucket_audio = self.get_bucket(self.PATHS['audio_bucket'])
        self.meta_data, self.train, self.test, self.long_test, self.matched_test, self.matched_train = self.load_train_test_splits()
        # base directory for audio files
        self.output_base= f'./data/ciab/{self.modality}'
        self.create_folds()

    def main(self):
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
            
        print('creating json')
        self.create_json()
        print('Beginining ciab train prepocessing')
        self.iterate_through_files(self.train, 'train')
        print('Beginining ciab test prepocessing')
        self.iterate_through_files(self.test, 'test')
        print('Beginining ciab long test prepocessing')
        self.iterate_through_files(self.long_test, 'long_test') 
        print('Beginining ciab matched test prepocessing')
        self.iterate_through_files(self.matched_test, 'matched_test') 
        print('Beginining ciab matched_train prepocessing')
        self.iterate_through_files(self.matched_train, 'matched_train')

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

    def load_train_test_splits(self):
        '''
        Loads the train and test barcode splits and the corresponding meta_data
        '''

        train_test = pd.read_pickle(self.get_file(
                                        self.PATHS['splits'],
                                         self.bucket_meta))
        pre_june_meta_data = pd.read_pickle(self.get_file(
                                        self.PATHS['pre_june_meta'],
                                        self.bucket_meta))
        post_june_meta_data = pd.read_pickle(self.get_file(
                                        self.PATHS['post_june_meta'],
                                        self.bucket_meta))
        meta_data = pd.concat([
                            pre_june_meta_data,
                            post_june_meta_data
                            ])
        
        long_test = pd.read_csv(self.get_file(
                            self.PATHS['longitudinal'],
                            self.bucket_meta))

        matched_test = pd.read_csv(self.get_file(
                                    self.PATHS['matched'],
                                    self.bucket_meta),
                                    names=['id'])
        matched_train = pd.read_csv(self.get_file(
                                    self.PATHS['matched_train'],
                                    self.bucket_meta),
                                    names=['id'])

        return meta_data, train_test['train'], train_test['test'], long_test['audio_sentence'].tolist(), matched_test['id'].tolist(), matched_train['id'].tolist()


    def iterate_through_files(self, dataset, split='train'):
        if not os.path.exists(f'{self.output_base}/audio_16k/{split}'):
            os.makedirs(f'{self.output_base}/audio_16k/{split}')
        self.error_list = []
        self.tot_removed = [] 
        bootstrap_results = Parallel(n_jobs=-1, verbose=10, prefer='threads')(delayed(self.process_file)(barcode_id, split) for barcode_id in dataset)
        
        print(f'Average fraction removed: {np.mean(self.tot_removed)}')
        
        with open(f'{self.output_base}/audio_16k/{split}/errorlist.txt', "w") as output:
            output.write(str(self.error_list))

    def process_file(self, barcode_id, split):

        df_match = self.meta_data[self.meta_data['audio_sentence'] == barcode_id]
        assert len(df_match) != 0, 'This unique code does not exist in the meta data file currently loaded - investigate!'
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
            self.error_list.append(filename)
            return 1
        clipped_signal, frac_removed = self.remove_silence(signal, barcode_id)
        self.tot_removed.append(frac_removed)
        sf.write(f'{self.output_base}/audio_16k/{split}/{barcode_id}', clipped_signal, 16000)
        return 1

    def print_stats(self):
        print(f'Sample numbers: Train: {len(self.train)}, \
                Test: {len(self.test)}, \
                Long_test: {len(self.long_test)} \
                matched_test: {len(self.matched_test)} \
                naive train: {len(self.naive_train)} \
                naive validation: {len(self.naive_val)} \
                naive test: {len(self.naive_test)}')
    
    def create_folds(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        self.folds = [[self.train[idx] for idx in test]
                 for (train, test) in kfold.split(self.train)]

        self.matched_train_folds = [[self.matched_train[idx] for idx in test]
                 for (train, test) in kfold.split(self.matched_train)]
    def create_json(self):
        #for fold in tqdm([1,2,3,4,5]):# for compute reasons have a fixed validation set when evaluating on test sets
        fold = 1
        train_list = [instance for instance in self.train if instance not in self.folds[fold-1]]
        validation_list = [instance for instance in self.train if instance in self.folds[fold-1]]
        assert not any(x in validation_list for x in train_list), 'there is cross over between train and validation'
        dic_train_list = self.list_to_dict(train_list, 'train')
        dic_validation_list = self.list_to_dict(validation_list, 'train')
        dic_test_list = self.list_to_dict(self.test, 'test')
        dic_matched_test_list = self.list_to_dict(self.matched_test, 'matched_test')
        dic_matched_train_list = self.list_to_dict(self.matched_train, 'matched_train')
        dic_long_test_list = self.list_to_dict(self.long_test, 'long_test')
        with open(f'./data/datafiles/{self.modality}/ciab_train_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_train_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_validation_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_validation_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_standard_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_matched_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_matched_test_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_long_test_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': dic_long_test_list}, f, indent=1)
        # create naive training evaluation sets. Note: this is for an ablation study to show the inflated performance
        all_data = dic_train_list + dic_validation_list + dic_test_list + dic_long_test_list # matched test and matched train are subsets of test and train respectively
        self.naive_train, self.naive_val, self.naive_test = self.create_naive_splits(all_data)

        assert not any(x in self.back_to_list(self.naive_val) for x in self.naive_train), 'there is cross over between naive train and naive validation'
        assert not any(x in self.back_to_list(self.naive_test) for x in self.naive_train), 'there is cross over between naive train and naive test'
        assert not any(x in self.back_to_list(self.naive_test) for x in self.naive_val), 'there is cross over between naive test and naive validation'
        with open(f'./data/datafiles/{self.modality}/naive_train_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': self.naive_train}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/naive_validation_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': self.naive_val}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/naive_test_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': self.naive_test}, f, indent=1)
        # Here we create an additional training set where we add the longitudinal data to the train
        big_train = dic_train_list + dic_validation_list + dic_long_test_list 
        self.big_train, self.big_val = self.create_naive_splits(big_train, just_val=True) 
        with open(f'./data/datafiles/{self.modality}/big_train_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': self.big_train}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/big_validation_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': self.big_val}, f, indent=1)
        # matched train and validation
        matched_train_list = [instance for instance in self.matched_train if instance not in self.matched_train_folds[fold-1]]
        matched_validation_list = [instance for instance in self.matched_train if instance in self.matched_train_folds[fold-1]]
        assert not any(x in matched_validation_list for x in matched_train_list), 'there is cross over between matched train and validation'
        matched_dic_train_list = self.list_to_dict(matched_train_list, 'train')
        matched_dic_validation_list = self.list_to_dict(matched_validation_list, 'train')
        with open(f'./data/datafiles/{self.modality}/ciab_matched_train_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': matched_dic_train_list}, f, indent=1)
        with open(f'./data/datafiles/{self.modality}/ciab_matched_validation_data_'+ str(fold) +'.json', 'w') as f:
            json.dump({'data': matched_dic_validation_list}, f, indent=1)
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
        formatted_list = [{"wav": f'{self.output_base}/audio_16k/{split}/{instance}', "labels": self.meta_data[self.meta_data['audio_sentence'] == instance].test_result.iloc[0]} for instance in data]
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
    ciab = PrepCIAB('audio_ha_sound_url')
    ciab.main()
    ciab.print_stats()
