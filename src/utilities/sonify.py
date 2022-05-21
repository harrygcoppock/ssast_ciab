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
        window_fn=torch.hann_window, 
        wkwargs={'periodic':False}
        )(inverse_mel_pred)
    return pred_audio.view(1,-1).squeeze()

if __name__ == '__main__':
    waveform, sr = torchaudio.load('/Downloads/you-are-acting-so-weird.wav')
    print(sr)
    transform = torchaudio.transforms.Resample(sr, 16000)
    print(waveform.size())
    torchaudio.save('in.wav', waveform.unsqueeze(0), 16000)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
        remove_dc_offset=False)
    raw_audio = sonify(fbank, 16000)
    torchaudio.save('output.wav', raw_audio, 16000)

