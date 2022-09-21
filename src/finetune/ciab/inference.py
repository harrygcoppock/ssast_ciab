'''
file: inference.py
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''
import torch
import torch.nn as nn
import torchaudio
import pickle
import os
import sys
import json
import numpy as np

sys.path.append('../../')

from models.ast_models import ASTModel
import dataloader
from traintest import validate


def load_trained_model(model_path, device, pretrain_path='SSAST-Base-Frame-400.pth'
):
    args = load_args(model_path)
    args.wandb = False
    audio_model = ASTModel(
            label_dim=args.n_class, 
            fshape=args.fshape, 
            tshape=args.tshape, 
            fstride=args.fstride, 
            tstride=args.tstride,
            input_fdim=args.num_mel_bins,
            input_tdim=args.target_length,
            model_size=args.model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path=pretrain_path)
    sd = torch.load(os.path.join(model_path, 'models/best_audio_model.pth'), map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)
    return audio_model, args

def load_args(model_path):
    with open(os.path.join(model_path, 'args.pkl'), "rb") as file:
        args = pickle.load(file)
        print(args)
    return args

def get_dataset(args, path_data='./data/datafiles/audio_sentence_url/ciab_test_data_1.json', indices=False):
    val_audio_conf = {
            'num_mel_bins': args.num_mel_bins, 
            'target_length': args.target_length, 
            'freqm': 0, 'timem': 0, 
            'mixup': 0, 
            'dataset': args.dataset, 
            'mode': 'evaluation', 
            'mean': args.dataset_mean, 
            'std': args.dataset_std, 
            'noise': False
            }
    eval_dataset = dataloader.AudioDataset(
        path_data,
        label_csv='./data/ciab_class_labels_indices.csv',
        audio_conf=val_audio_conf,
        indices=indices)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=8,
        pin_memory=True)
    return eval_dataset, eval_loader

def main(model_path, data_path, test_type, output_dir, method='frame', pca_proj=False):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    audio_model, args = load_trained_model(model_path, device)
    args.exp_dir = output_dir
    if args.loss == 'BCE':
        args.loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = torch.nn.CrossEntropyLoss()
    args.batch_size = 50
    eval_dataset, eval_loader = get_dataset(args, data_path, indices=True) #to get indexes  
    stats, _ = validate(
            audio_model, 
            eval_loader, 
            args, 
            0,
            pca_proj=pca_proj,
            dataset=eval_dataset,
            test_type=test_type
            )
    if pca_proj:
        loss, pca_proj = _
    else:
        loss = _
   # converting the np arrays to lists
    for item in stats:
        
        item['precisions'] = item['precisions'].tolist()
        item['recalls'] = item['recalls'].tolist()
        item['fpr'] = item['fpr'].tolist()
        item['fnr'] = item['fnr'].tolist() 
    with open(os.path.join(output_dir, f'metrics_{test_type}_covid.json'), 'w') as f:
        json.dump(stats[1], f)
    with open(os.path.join(output_dir, f'metrics_{test_type}_covid_neg.json'), 'w') as f:
        json.dump(stats[0], f)
    if pca_proj:
        pca_proj.to_csv(os.path.join(output_dir, f'analysis_{test_type}_pca_projections.csv'))



if __name__ == '__main__':
    #main(
    #    './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
    #    './data/datafiles/audio_sentence_url/ciab_matched_test_data_1.json',
    #    'train_matched_test',
    #    './exp/inference/train_matched_test'
    #    )
    main(
        './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
        './data/datafiles/audio_sentence_url/ciab_long_test_data_1.json',
        'train_long_test',
        './exp/inference/train_long_test'
        )
    main(
        './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
        './data/datafiles/audio_sentence_url/ciab_test_data_1.json',
        'train_test',
        './exp/inference/train_test'
        )
    main(
        './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
        './data/datafiles/audio_sentence_url/ciab_validation_data_1.json',
        'train_validation',
        './exp/inference/train_validation'
        )
    main(
        './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
        './data/datafiles/audio_sentence_url/ciab_train_data_1.json',
        'train_train',
        './exp/inference/train_train'
        )
    #main(
    #    './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1',
    #    './data/datafiles/audio_sentence_url/ciab_long_matched_data_1.json',
    #    'train_long_matched_test',
    #    './exp/inference/train_long_matched_test'
    #    )
    #main(
    #    './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-matched-train/fold1',
    #    './data/datafiles/audio_sentence_url/ciab_matched_test_data_1.json',
    #    'matched_train_matched_test',
    #    './exp/inference/matched_train_matched_test'
    #    )
    #main(
    #    './exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-matched-train/fold1',
    #    './data/datafiles/audio_sentence_url/ciab_long_matched_data_1.json',
    #    'matched_train_long_matched_test',
    #    './exp/inference/matched_train_long_matched_test'
    #    )
