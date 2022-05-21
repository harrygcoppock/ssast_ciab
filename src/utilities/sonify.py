import torch
import torchaudio
import matplotlib.pyplot as plt
from typing import Callable, Optional
from torch import Tensor

def sonify(fbank, sr):
    '''
    Applys the inverse mel and stft to a mel filter bank
    fbank: torch.Tensor (m, n_mels) where m = 1 + (num_samples - window_size) // window_shift
    sr: int sample rate 
    '''
    inverse_mel_pred = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=257)(fbank.T)
    pred_audio = torchaudio.transforms.GriffinLim(
        n_fft=512,
        n_iter=1000,
        hop_length=160,
        window_fn=torch.hann_window, 
        wkwargs={'periodic':False}
        )(inverse_mel_pred)
    return pred_audio.view(1,-1).squeeze()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.savefig(f'{title}.png')

if __name__ == '__main__':
    waveform, sr = torchaudio.load('../../figure/you-are-acting-so-weird.wav')
    new_sr = 16000
    transform = torchaudio.transforms.Resample(sr, new_sr)
    waveform = transform(waveform[0])
    plot_waveform(waveform.unsqueeze(0), sample_rate=new_sr, title='in')
    torchaudio.save('in2.wav', waveform.unsqueeze(0), new_sr)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=new_sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
        remove_dc_offset=False)

    raw_audio = sonify(fbank, new_sr)
    plot_waveform(raw_audio.unsqueeze(0), sample_rate=new_sr, title='out')
    torchaudio.save('output2.wav', raw_audio.unsqueeze(0), new_sr)

