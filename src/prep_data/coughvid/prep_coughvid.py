import wget
import zipfile
import numpy as np
import json
import os
from sklearn.model_selection import KFold

import pandas as pd
import subprocess, glob, csv
from tqdm import tqdm
import pickle
import librosa, librosa.display
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import soundfile as sf
# download
# resample to 16k
# mean and std
# create csv
# create json


class PrepCoughVid():

    RANDOM_SEED = 42

    def __init__(self, modality='audio_three_cough_url'):
        self.output_base= f'./data/audio_16k'
        self.input_base = f'.data/public_dataset'
        self.create_folds()
        if not os.path.exists(self.input_base):
            self.download_data('https://zenodo.org/record/4498364/files/public_dataset.zip?download=1')

    def download_data(self, url):
        wget.download(url, out='./data/')
        with zipfile.ZipFile('./data/public_dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('./data/')
        os.system(f'rm -rf .data/public_dataset.zip')

    def main(self):
        if not os.path.exists(self.output_base):
            os.makedirs(self.output_base)
        print('creating json')
        self.create_json()
        # now iterate through the files
        audio_files = [audio_file for audio_file in os.listdir(self.input_base) if audio_file.endswith(".webm") or audio_file.endswith(".ogg")]
        self.error_list = []
        self.tot_removed = 0
        Parallel(n_jobs=-1, verbose=10, prefer='threads')(delayed(self.process_file)(id) for id in audio_files)
        
        print(f'Average fraction removed: {np.mean(self.tot_removed)}')
        
        with open(f'{self.output_base}/audio_16k/{split}/errorlist.txt', "w") as output:
            output.write(str(self.error_list))

    def process_file(self, filename):

        try:
            signal, sr = librosa.load(filename, sr=16000)
        except RuntimeError:
            print(f"{filename} not possible to load. Total so far: {len(error_list)}")
            self.error_list.append(filename)
            return 
        clipped_signal, frac_removed = self.remove_silence(signal)
        self.tot_removed += frac_removed
        sf.write(f"{self.output_base}/audio_16k/{re.match('([^/]+$)', filename).group(0)}", clipped_signal, 16000)
        return 
        
    
    def create_folds(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.RANDOM_SEED)
        self.folds = [[self.train[idx] for idx in test]
                 for (train, test) in kfold.split(self.train)]

    def create_json(self):
        for fold in [1]:# we do not have the compute for 5 fold cross validation in pretraining just finetuning
            train_list = [instance for instance in self.train if instance not in self.folds[fold-1]]
            validation_list = [instance for instance in self.train if instance in self.folds[fold-1]]
            assert not any(x in validation_list for x in train_list), 'there is cross over between train and validation'
            with open('./data/datafiles/ciab_train_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': self.list_to_dict(train_list, 'train')}, f, indent=1)
            with open('./data/datafiles/ciab_validation_data_'+ str(fold) +'.json', 'w') as f:
                json.dump({'data': self.list_to_dict(validation_list, 'train')}, f, indent=1)

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
        formatted_list = [{"wav": f'{self.output_base}/{instance}', "labels": 'blank'} for instance in data]
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

        return np.array(clipped_signal), (length_prior - length_post)/length_prior

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
if __name__ == "__main__":
    e = PrepCoughVid()
    e.main()