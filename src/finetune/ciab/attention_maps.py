'''
file: attention_maps.py
author: Harry Coppock
Qs: harry.coppock@imperial.ac.uk
'''
import torch
import torch.nn as nn
import torchaudio
import pickle
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import colorConverter
sys.path.append('../../')
from models.ast_models import ASTModel
import dataloader
np.set_printoptions(threshold=sys.maxsize)

def load_trained_model(model_path, device, pretrain_path='/workspace/ssast_ciab/src/finetune/ciab/SSAST-Base-Patch-400.pth'
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

def get_dataset(args, path_data='/workspace/ssast_ciab/src/finetune/ciab/data/datafiles/audio_sentence_url/ciab_test_data_1.json'):
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
        label_csv='/workspace/ssast_ciab/src/finetune/ciab/data/ciab_class_labels_indices.csv',
        audio_conf=val_audio_conf,
        pca_proj=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    return eval_dataset, eval_loader

def get_attention(audio_model, loader, device, args):
    for i, (fbank, label_indices, index) in enumerate(loader):
        if i == 10:
            break

    print('label', label_indices)
    fbank = fbank.to(device)
    audio_model = audio_model.to(device)
    audio_model.eval()
    with torch.no_grad():
        output, tup_out = audio_model(
            fbank,
            args.task,
            pca_proj=True,
            return_attention=True
            )
    pca_proj, attention = tup_out
    return attention, fbank.cpu().transpose(1,2), pca_proj, index

def format_attention_map(attentions, audio_model, method, args, threshold_att_maps=True, batch_num=0):
    '''
    reshape attention so that it is the same size as orignal fbank
    '''
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attenion
    print(audio_model.module.f_dim)
    print(audio_model.module.t_dim)
    attentions = attentions[batch_num, :, 0, 2:].reshape(nh, -1)
    if method == 'frame':
        attentions = attentions.reshape(nh, audio_model.module.t_dim)
        attentions = attentions[0].cpu().numpy()
        #fig = plt.figure()
        #plt.plot(list(range(len(attentions))), attentions)
        #plt.savefig('attention.png')
        return attentions, nh
    else:
        if threshold_att_maps:
            attentions = threshold_att(attentions, nh, audio_model, args)
            #plt.imsave(fname='first3.png', arr=attentions[2], format='png')
            #return attentions, nh
        print('assuming patch based approach')
        print(attentions.size())
        attentions = attentions.reshape(nh, audio_model.module.f_dim, audio_model.module.t_dim)
        attentions = nn.functional.interpolate(
                attentions.unsqueeze(0), 
                scale_factor=(args.fshape, args.tshape), 
                mode="nearest")[0].cpu().numpy()
        #plt.imsave(fname='first2.png', arr=attentions[0], format='png')
        return attentions, nh

def threshold_att(attentions, nh, audio_model, args):
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - 0.1)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    #th_attn = th_attn.reshape(nh, audio_model.module.f_dim, audio_model.module.t_dim).float()
    # interpolate
    #th_attn = nn.functional.interpolate(
    #        th_attn.unsqueeze(0), 
    #        scale_factor=(args.fshape, args.tshape), 
    #        mode="nearest"
    #        )[0]
    th_attn = th_attn.float()
    print(th_attn)
    print(th_attn.size())
    return th_attn

def plot_attentions(attensions, fbank, nh, mean, std, batch_num=0):
    fig, axs = plt.subplots(nh+1,1, figsize=(8,20), sharex=True)
    axs[0].imshow(fbank[batch_num])
    for i in range(nh):
        #plot for each head
        axs[i+1].imshow(attensions[i])
    plt.savefig('files/attentions_0_pos.png', bbox_inches='tight')

def plot_attentions_overlay(attensions, fbank, nh, mean, std, batch_num=0, axs=None):
    out = []
    out.append(axs.imshow(fbank[batch_num]))
    color_list = ['cyan', 'magenta', 'darkviolet', 'olive', 'pink', 'blue', 'green', 'lime', 'red', 'orange', 'yellow', 'purple']
    for i in range(nh):
        #plot for each head
        # generate the colors for your colormap
        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba(color_list[i])

        # make the colormaps
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

        cmap1._init() # create the _lut array, with rgba values

        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0, 0.8, cmap1.N+3)
        cmap1._lut[:,-1] = alphas
        out.append(axs.imshow(attensions[i], cmap=cmap1))
    return out



def logits_per_patch(audio_model, device, pca_proj, ax, batch_num):
    audio_model = audio_model.to(device)
    audio_model.eval()
    with torch.no_grad():
        logits = torch.nn.functional.softmax(audio_model.module.mlp_head(pca_proj[batch_num,:,:]))
    #logits = logits[:,1] - logits[:,0]
    #logits = logits[2:].tolist()
    #out = threshold_plot(ax, range(0, len(logits[2:,1].tolist())), logits[2:,1].tolist(), 0.5, 'green', 'red')
    out = ax.plot(range(0, len(logits[2:,1].tolist())), logits[2:,1].tolist())
    ax.set_ylim(0,1)
    ax.fill_between(range(0, len(logits[2:,1].tolist())), 0.5, 1, facecolor='red', alpha=0.4, label='COVID Positive')
    ax.fill_between(range(0, len(logits[2:,1].tolist())), 0, 0.5, facecolor='green', alpha=0.4, label='COVID Negative')
  #out.append(ax.plot(range(0, len(logits[2:,1].tolist())), logits[2:,1].tolist(), label='Positive'))
    #out.append(ax.plot(range(0, len(logits[2:,1].tolist())), logits[2:,0].tolist(), label='Negative'))
    ax.legend()
    return out



def convert_attention_map(attention, spectrum):
    '''
    given the coords of the attention map for the fbank, project onto the spectrum
    '''
    width, height = spectrum.size()[1], spectrum.size()[0]
    print('width and heigh', width, height)
    print('attention size', np.shape(attention))
    new = np.zeros(shape=(width, height))
    new[:-1:2] = attention
    new[1::2] = attention
    return new

def sonfiy_attention(attention, index, eval_dataset, batch_num, args):
    '''
    given the attetion map sonficy the attention portions
    '''
    datum = eval_dataset.data[index[batch_num]]
    filename = datum['wav']
    waveform, sr = torchaudio.load(filename)
    spectrum, window_shift, window_size = spectrogram_rep(waveform, args)
    print(np.shape(attention))
    attention = np.sum(attention, axis=0)
    print(np.shape(attention))
    attention = convert_attention_map(attention, spectrum)

    # now only keep the sections of the spectrogram which the model
    # paid attention to
    spectrum_att = select_areas(spectrum, np.transpose(attention))
    t = torchaudio.transforms.InverseSpectrogram(n_fft=512,
                                         win_length=window_size,
                                         hop_length=window_shift)

    att_wav = t(spectrum_att.T)
    torchaudio.save(f'files/{batch_num}waveform.wav', waveform, 16000)
    torchaudio.save(f'files/{batch_num}atten_recon.wav', att_wav.view(1,-1), 16000)
    fig, axs = plt.subplots()
    axs.imshow(spectrum_att.abs().T)
    plt.savefig(f'files/{batch_num}spectrum_att.png', bbox_inches='tight')

def select_areas(spectrum, attention):
    attention = torch.tensor(attention)
    assert spectrum.size() == attention.size()

    for i in range(spectrum.size()[0]):
        for j in range(spectrum.size()[1]):
            spectrum[i,j] = spectrum[i,j] if attention[i,j] >= 1.0 else 0
    return spectrum

def spectrogram_rep(waveform, args):
    '''
    mimic the spectrogram step in torchaudio fbank
    '''
    waveform, window_shift, window_size, padded_window_size = torchaudio.compliance.kaldi._get_waveform_and_window_properties(
        waveform,
        channel=0,
        sample_frequency=16000,
        frame_shift=10.0,
        frame_length=25.0,
        round_to_power_of_two=True,
        preemphasis_coefficient=0.97)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = torchaudio.compliance.kaldi._get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type='hanning',
        blackman_coeff=0.42,
        snip_edges=True,
        raw_energy=True,
        energy_floor=1.0,
        dither=0.0,
        remove_dc_offset=True,
        preemphasis_coefficient=0.97
    )

    # size (m, padded_window_size // 2 + 1)
    spectrum = torch.fft.rfft(strided_input)
    print('spectrum size:   ', spectrum.size())
    target_length = args.target_length
    n_frames = spectrum.shape[0]
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        spectrum = m(spectrum)
    elif p < 0:
        spectrum = spectrum[0:target_length, :]
    print('spectrum size:   ', spectrum.size())
    return spectrum, window_shift, window_size


def main(model_path, method='patch', batch_num=85):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model, args = load_trained_model(model_path, device)
    if args.loss == 'BCE':
        args.loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = torch.nn.CrossEntropyLoss()
    args.batch_size = 50
    eval_dataset, eval_loader = get_dataset(args)   
    attention, fbank, pca_proj, index = get_attention(audio_model, eval_loader, device, args)
    attentions, nh = format_attention_map(attention, audio_model, method, args, threshold_att_maps=False, batch_num=batch_num)
    plot_attentions(attentions, fbank, nh, eval_dataset.norm_mean, eval_dataset.norm_std, batch_num)
    plt.close()

    fig, axs = plt.subplots(2,1, sharex=True)
    attentions, nh = format_attention_map(attention, audio_model, method, args, threshold_att_maps=True, batch_num=batch_num)
    plot_attentions_overlay(attentions, fbank, nh, eval_dataset.norm_mean, eval_dataset.norm_std, batch_num=batch_num, axs=axs[0])

    #get logits per time step
    audio_model, args = load_trained_model('/workspace/ssast_ciab/src/finetune/ciab/exp/final/ciab_sentence-f128-128-t1-2-b20-lr1e-4-ft_avgtok-base-unknown-SSAST-Base-Frame-400-1x-noiseTrue-standard-train/fold1', device, pretrain_path='/workspace/ssast_ciab/src/finetune/ciab/SSAST-Base-Frame-400.pth'
)
    if args.loss == 'BCE':
        args.loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = torch.nn.CrossEntropyLoss()
    args.batch_size = 50
    eval_dataset_1, eval_loader_1 = get_dataset(args)
    attention_1, fbank_1, pca_proj_1, index_1 = get_attention(audio_model, eval_loader_1, device, args)
    logits_per_patch(audio_model, device, pca_proj_1, axs[1], batch_num=batch_num)

    plt.savefig(f'files/attention_maps_logits_batchnum_{batch_num}.png', bbox_inches='tight')
    plt.close()

    sonfiy_attention(attentions, index, eval_dataset, batch_num, args)

if __name__ == '__main__':
    main('/workspace/ssast_ciab/src/finetune/ciab/exp/test01-ciab_sentence-f16-16-t16-16-b18-lr1e-4-ft_cls-base-unknown-SSAST-Base-Patch-400-1x-noiseTrue-standard-train-2/fold1')
