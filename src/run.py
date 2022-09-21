# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py


#TODO: add back in pca proj for evaluation step.
import argparse
import os
import ast
import pickle
import sys
import time
import torch
import torch.nn as nn
import json
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest import train, validate
from traintest_mask import trainmask

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-test", type=str, default=None, help="test data json")
parser.add_argument("--data-matched-test", type=str, default=None, help="matched test data json")
parser.add_argument("--data-long-test", type=str, default=None, help="longitudinal data json")
parser.add_argument("--data-long-matched", type=str, default=None, help="longitudinal matched data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])

parser.add_argument("--wandb", type=str, default="wandb", help="Do you want to log your changes to wandb?", choices=[None, "wandb"])
args = parser.parse_args()



if args.wandb == 'wandb':
    import wandb
    wandb.init(project='ciab-turing-ukhsa', entity='harrygcoppock', config=args)

audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.
if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_dataset = dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))

# in the pretraining stage
if 'pretrain' in args.task:
    cluster = (args.num_mel_bins != args.fshape)
    if cluster == True:
        print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
    else:
        print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
    # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
    audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)
# in the fine-tuning stage
else:
    audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path=args.pretrained_mdl_path)

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

if 'pretrain' not in args.task:
    print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_loader, val_loader, args, test_type='training', val_dataset=val_dataset)
else:
    print('Now starting self-supervised pretraining for {:d} epochs'.format(args.n_epochs))
    trainmask(audio_model, train_loader, val_loader, args)

# if the dataset has a seperate evaluation set (e.g., speechcommands), then select the model using the validation set and eval on the evaluation set.
# this is only for fine-tuning
if args.data_test != None:
    # saving covid specific results for turing
    metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best models on the validation set
    if args.loss == 'BCE':
        args.loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = nn.CrossEntropyLoss()
    stats, _ = validate(audio_model, val_loader, args, 'valid_set', dataset=val_dataset)
    # note it is NOT mean of class-wise accuracy
    metrics['val'] = stats[1]
    val_acc = stats[1]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the models on the evaluation set
    eval_dataset = dataloader.AudioDataset(args.data_test, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    stats, _= validate(audio_model,#, pca_proj 
            eval_loader, 
            args, 
            'eval_set', 
            pca_proj=True, 
            dataset=eval_dataset,
            test_type='test')
    loss, pca_proj = _
    metrics['test']= stats[1]
    eval_acc = stats[1]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    pca_proj.to_csv(args.exp_dir+'/test_pca_projections.csv') 
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
    if args.data_matched_test != None:
        matched_dataset = dataloader.AudioDataset(args.data_matched_test, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
        matched_test_loader = torch.utils.data.DataLoader(
                matched_dataset,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model,#, pca_proj 
                matched_test_loader, 
                args, 
                'matched_set', 
                pca_proj=True, 
                dataset=matched_dataset,
                test_type='matched_test')
        loss, pca_proj = _
        metrics['matched_test'] = stats[1]
        eval_acc = stats[1]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the matched test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/matched_test_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
        pca_proj.to_csv(args.exp_dir+'/matched_test_pca_projections.csv') 
    if args.data_long_test != None:
        long_dataset = dataloader.AudioDataset(args.data_long_test, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
        long_test_loader = torch.utils.data.DataLoader(
                long_dataset,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model,#, pca_proj 
                long_test_loader, 
                args, 
                'long_set', 
                pca_proj=False, 
                dataset=long_dataset,
                test_type='long_test')
        loss = _
        metrics['long_test'] = stats[1]
        eval_acc = stats[1]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the long test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/long_test_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
        #pca_proj.to_csv(args.exp_dir+'/long_test_pca_projections.csv') 
    
    if args.data_long_matched != None:
        long_dataset = dataloader.AudioDataset(args.data_long_matched, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
        long_test_loader = torch.utils.data.DataLoader(
                long_dataset,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, _ = validate(audio_model,#, pca_proj 
                long_test_loader, 
                args, 
                'matched_long_set', 
                pca_proj=True, 
                dataset=long_dataset,
                test_type='matched_long_test')
        loss, pca_proj = _
        metrics['matched_long_test'] = stats[1]
        eval_acc = stats[1]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print('---------------evaluate on the matched long test set---------------')
        print("Accuracy: {:.6f}".format(eval_acc))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/matched_long_test_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
        pca_proj.to_csv(args.exp_dir+'/matched_long_test_pca_projections.csv') 
    # repeat eval for the training set - this is useful so we can train 1NN model and analysis the learnt features

    #analysis_train_dataset = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=val_audio_conf, indices=True)
    #analysis_train_loader = torch.utils.data.DataLoader(
    #        analysis_train_dataset,
    #        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    #stats, _ = validate(audio_model, #, pca_proj
    #        analysis_train_loader, 
    #        args, 
    #        'analysis_train_set', 
    #        pca_proj=True, 
    #        dataset=analysis_train_dataset,
    #        test_type='analysis_train')
    #loss, pca_proj = _
    #metrics['analysis_train_dataset'] = stats[1]
    #eval_acc = stats[1]['acc']
    #eval_mAUC = np.mean([stat['auc'] for stat in stats])
    #print('---------------evaluate on the train set---------------')
    #print("Accuracy: {:.6f}".format(eval_acc))
    #print("AUC: {:.6f}".format(eval_mAUC))
    #np.savetxt(args.exp_dir + '/analysis_train_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
    #pca_proj.to_csv(args.exp_dir+'/analysis_train_pca_projections.csv') 
    # converting the np arrays to lists
    for key, item in metrics.items():
        item['precisions'] = item['precisions'].tolist()
        item['recalls'] = item['recalls'].tolist()
        item['fpr'] = item['fpr'].tolist()
        item['fnr'] = item['fnr'].tolist()
    with open(os.path.join(args.exp_dir, f'metrics.json'), 'w') as f:
        json.dump(metrics, f)
