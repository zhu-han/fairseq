# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import torch

from .raw_audio_dataset import FileAudioDataset
import numpy
import scipy
import soundfile as sf
import librosa



logger = logging.getLogger(__name__)


class AugmentCenter(object):
    def __init__(self, seed=8888, rir_dict=None, noise_dice=None, sample_rate=16000):
        self.state = numpy.random.RandomState(seed)
        self.sample_rate = sample_rate
        self.rir_list = [] if rir_dict is None or rir_dict == "None" else self.read_source(rir_dict)
        self.noise_list = []  if noise_dice is None or noise_dice == "None" else self.read_source(noise_dice)
        # self.noise_list = [ noise/numpy.sqrt((noise ** 2).mean()) for noise in self.noise_list]
        
    def read_wav(self, filename):
        signal, rate = sf.read(filename)
        if rate != self.sample_rate: 
            ratio = rate / self.sample_rate
            signal = librosa.resample(signal, ratio, 1, res_type="kaiser_best")
        return signal

        
    def read_source(self, utt_scp):
        utt_dict = []
        with open(utt_scp, "r") as f:
            for line in f:
                utt, filename = line.rstrip().split(None, 1)
                utt_dict.append(filename)
        return utt_dict

    def speed_preturb_samelength(self, x, lower=0.8, upper=1.5):
        # x = x[:,0]
        x = x.astype(numpy.float32)
        ratio = self.state.uniform(lower, upper)

        # Note1: resample requires the sampling-rate of input and output,
        #        but actually only the ratio is used.
        y = librosa.resample(x, ratio, 1, res_type="kaiser_best")

        diff = abs(len(x) - len(y))
        if len(y) > len(x):
            # Truncate noise
            y = y[diff // 2 : -((diff + 1) // 2)]
        elif len(y) < len(x):
            # Assume the time-axis is the first: (Time, Channel)
            pad_width = [(diff // 2, (diff + 1) // 2)] + [
                (0, 0) for _ in range(y.ndim - 1)
            ]
            y = numpy.pad(
                y, pad_width=pad_width, constant_values=0, mode="constant"
            )
        return y
    
    def voice_preturb_const(self, x, lower=0.5, upper=2.0, dbunit=False):
        x = x.astype(numpy.float32)
        ratio = self.state.uniform(lower, upper)
        if dbunit:
            ratio = 10 ** (ratio / 20)
        return ratio * x
        
    def voice_preturb_dynamic(self, x, lower=0.5, upper=2.0, peak=3, dbunit=False):
        # x = x.astype(numpy.float32)
        xlen = len(x)
        if dbunit:
            ratio_peak = [1] +list(10 ** (self.state.uniform(lower, upper, size=peak)/20)) + [1]
        else:
            ratio_peak = [1] + list(self.state.uniform(lower, upper, size=peak)) + [1]
        ratio_pos = [0]
        for i in range(peak):
            ratio_pos.append(self.state.randint(ratio_pos[i], (i+1)*xlen//peak))
        ratio_pos.append(xlen)
        ration_list = numpy.ones_like(x)
        for i in range(peak+1):
            sample_num = ratio_pos[i+1] - ratio_pos[i]
            ration_list[ratio_pos[i]:ratio_pos[i+1]] = numpy.linspace(ratio_peak[i], ratio_peak[i+1], num=sample_num)       

        return ration_list * x

    def random_noise(self, x, lower=0.1, upper=0.5, dbunit=False):
        # noise = self.state.normal(0, 1, len(x))
        noise_pos = self.state.randint(len(self.noise_list) + 1)
        if noise_pos == len(self.noise_list):
            noise = self.state.normal(0, 1, len(x))
        else:
            noise = self.noise_list[noise_pos]
            noise = self.read_wav(noise)
            noise = noise/numpy.sqrt((noise ** 2).mean())            
            diff = abs(len(x) - len(noise))
            offset = self.state.randint(0, diff)
            if len(noise) > len(x):
                # Truncate noise
                noise = noise[offset : -(diff - offset)]
            else:
                noise = numpy.pad(noise, pad_width=[offset, diff - offset], mode="wrap")
            
        ratio = self.state.uniform(lower, upper)
        if dbunit:
            ratio = 10 ** (ratio / 20)
        scale = ratio * numpy.sqrt((x ** 2).mean())
        return x + noise * scale
        
    def fair_filed(self, x):
        if len(self.rir_list)==0:
            return x
        xlen = len(x)
        rir = self.rir_list[self.state.randint(len(self.rir_list))]
        rir = self.read_wav(rir)
        x = scipy.signal.convolve(x, rir, mode="full")[:xlen]
        return x / x.max() * 0.3

    
    def frequence_preturb_dynamic(self, x, lower=0.1, upper=20, peak=5, dbunit=False):
        data = numpy.fft.rfft(x)
        data = self.voice_preturb_dynamic(data, lower=lower, upper=upper, peak=peak, dbunit=dbunit)
        data = numpy.fft.irfft(data)
        return data

    
    def frequence_shift(self, x, lower=20, upper=200, peak=5):
        data = numpy.fft.rfft(x)
        data_len = len(data)
        freq_h = self.state.randint(lower, upper)
        data_off = int(freq_h/self.sample_rate * data_len)
        if self.state.randint(0, 1) == 0:
            data[data_off:] = data[:data_len-data_off]
        else:
            data[:data_len-data_off] = data[data_off:]
        data = numpy.fft.irfft(data)
        return data

    def voice_noarm(self, x):
        return x / x.max() * 0.3


class AugmentFileAudioDataset(FileAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        augment_config={
            "seed": 8888, 
            "rir_dict":None, 
            "noise_dice":None, 
            "sample_rate": 16000
        },
        **mask_compute_kwargs,
    ):
        super().__init__(
            manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            num_buckets=num_buckets,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )
        self.augment_center = AugmentCenter(**augment_config)

    def collater(self, samples):
        out = super().collater(samples)
        raw_wavs = out["net_input"]["source"]
        aug_wavs = torch.zeros_like(out["net_input"]["source"]).to(raw_wavs)
        for i,wav in enumerate(raw_wavs.numpy()):
            aug_wav = self.augment_center.voice_preturb_dynamic(wav, -20, 20, dbunit=True)
            aug_wav = self.augment_center.frequence_preturb_dynamic(aug_wav, -20, 20, dbunit=True)
            aug_wav = self.augment_center.frequence_shift(aug_wav)
            aug_wav = self.augment_center.fair_filed(aug_wav)
            aug_wav = self.augment_center.random_noise(aug_wav, -40, -20, dbunit=True)
            length = min(len(wav), len(aug_wav))
            aug_wavs[i,:length] = torch.from_numpy(aug_wav)[:length]

        out["net_input"]["source"] = [raw_wavs, aug_wavs]

        return out