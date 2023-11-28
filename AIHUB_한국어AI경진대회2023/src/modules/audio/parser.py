# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys, os

import torch
from modules.audio.core import load_audio
from torch import Tensor, FloatTensor
from modules.audio.augment import SpecAugment
from modules.audio.feature import (
    MelSpectrogram,
    MFCC,
    Spectrogram,
    FilterBank,
)

##추가##
import glob, random, soundfile
from scipy import signal
import pickle as pk
##까지##


class AudioParser(object):
    """
    Provides inteface of audio parser.

    Note:
        Do not use this class directly, use one of the sub classes.

    Method:
        - **parse_audio()**: abstract method. you have to override this method.
        - **parse_transcript()**: abstract method. you have to override this method.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def parse_audio(self, *args, **kwargs):
        raise NotImplementedError

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    """
    Parses audio file into (spectrogram / mel spectrogram / mfcc) with various options.

    Args:
        transform_method (str): which feature to use (default: mel)
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction (default: librosa)
        del_silence (bool): flag indication whether to delete silence or not (default: True)
        input_reverse (bool): flag indication whether to reverse input or not (default: True)
        normalize (bool): flag indication whether to normalize spectrum or not (default:True)
        freq_mask_para (int): Hyper Parameter for Freq Masking to limit freq masking length
        time_mask_num (int): how many time-masked area to make
        freq_mask_num (int): how many freq-masked area to make
        sos_id (int): start of sentence token`s identification
        eos_id (int): end of sentence token`s identification
        dataset_path (str): noise dataset path
    """
    VANILLA = 0           # Not apply augmentation
    SPEC_AUGMENT = 1      # SpecAugment

    def __init__(
            self,
            feature_extract_by: str = 'librosa',      # which library to use for feature extraction
            sample_rate: int = 8000,                 # sample rate of audio signal.
            n_mels: int = 80,                         # Number of mfc coefficients to retain.
            frame_length: int = 20,                   # frame length for spectrogram
            frame_shift: int = 10,                    # Length of hop between STFT windows.
            del_silence: bool = False,                # flag indication whether to delete silence or not
            input_reverse: bool = True,               # flag indication whether to reverse input or not
            normalize: bool = False,                  # flag indication whether to normalize spectrum or not
            transform_method: str = 'mel',            # which feature to use [mel, fbank, spect, mfcc]
            freq_mask_para: int = 12,                 # hyper Parameter for Freq Masking to limit freq masking length
            time_mask_num: int = 2,                   # how many time-masked area to make
            freq_mask_num: int = 2,                   # how many freq-masked area to make
            sos_id: int = 1,                          # start of sentence token`s identification
            eos_id: int = 2,                          # end of sentence token`s identification
            dataset_path: str = None,                 # noise dataset path
            audio_extension: str = 'pcm',             # audio extension
            rir_path : str = None,
            musan_path : str = None,
            aug : bool = False,
            mode : str = 'train'
    ) -> None:
        super(SpectrogramParser, self).__init__(dataset_path)
        self.del_silence = del_silence
        self.input_reverse = input_reverse
        self.normalize = normalize
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.spec_augment = SpecAugment(freq_mask_para, time_mask_num, freq_mask_num)
        self.audio_extension = audio_extension
        self.mode = mode
        
        if transform_method.lower() == 'mel':
            self.transforms = MelSpectrogram(sample_rate, n_mels, frame_length, frame_shift, feature_extract_by)
        elif transform_method.lower() == 'mfcc':
            self.transforms = MFCC(sample_rate, n_mels, frame_length, frame_shift, feature_extract_by)
        elif transform_method.lower() == 'spect':
            self.transforms = Spectrogram(sample_rate, frame_length, frame_shift, feature_extract_by)
        elif transform_method.lower() == 'fbank':
            self.transforms = FilterBank(sample_rate, n_mels, frame_length, frame_shift)
        else:
            raise ValueError("Unsupported feature : {0}".format(transform_method))
            
        self.aug = aug
        print('Noise augmentation : ',self.aug)
        if self.aug and mode == 'train':
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
            self.noiselist = {}
            self.musan_path = musan_path
            self.rir_path = rir_path


            augment_files   = glob.glob(os.path.join(self.musan_path,'*/*/*/*.wav'))
            print('MUSAN Files : ',len(augment_files))
            for file in augment_files:
                if file.split('/')[-4] not in self.noiselist:
                    self.noiselist[file.split('/')[-4]] = []
                self.noiselist[file.split('/')[-4]].append(file)
            self.rir_files  = glob.glob(os.path.join(self.rir_path,'*/*/*.wav'))
            print('RIR Files : ',len(self.rir_files))    
            
            
            

    def parse_audio(self, audio_path: str, augment_method: int) -> Tensor:
        """
        Parses audio.

        Args:
             audio_path (str): path of audio file
             augment_method (int): flag indication which augmentation method to use.

        Returns: feature_vector
            - **feature_vector** (torch.FloatTensor): feature from audio file.
        """
        signal = load_audio(audio_path, self.del_silence, extension=self.audio_extension)

        if self.aug and self.mode == 'train':
        
            audio = signal

            length = len(audio)
            audio = np.stack([audio],axis=0)
            # Data Augmentation
            augtype = random.randint(0,11)
            if augtype in [0,6,7,8,9,10,11]:   # Original
                audio = audio
            elif augtype == 1: # Reverberation
                audio = self.add_rev(audio,length)
            elif augtype == 2: # Babble
                audio = self.add_noise(audio, 'speech',length)
            elif augtype == 3: # Music
                audio = self.add_noise(audio, 'music',length)
            elif augtype == 4: # Noise
                audio = self.add_noise(audio, 'noise',length)
            elif augtype == 5: # Television noise
                audio = self.add_noise(audio, 'speech',length)
                audio = self.add_noise(audio, 'music',length)
#        signal = torch.FloatTensor(audio[0])
            signal = audio[0]
        #####################################################################################3 mel spec이면 numpy로 받아야함
        
        if signal is None:
            # print("Audio is None : {0}".format(audio_path))
            return None

        feature = self.transforms(signal)
#        feature = torch.FloatTensor(feature)
#        print(feature.shape)
#        print(feature)
        
        if self.normalize:
            feature -= feature.mean()
            feature /= np.std(feature)

        # Refer to "Sequence to Sequence Learning with Neural Network" paper
        if self.input_reverse:
            feature = feature[:, ::-1]
            feature = FloatTensor(np.ascontiguousarray(np.swapaxes(feature, 0, 1)))
        else:
            feature = FloatTensor(feature).transpose(0, 1)

        if augment_method == SpectrogramParser.SPEC_AUGMENT:
            feature = self.spec_augment(feature)

        return torch.FloatTensor(feature)

    def parse_transcript(self, *args, **kwargs):
        raise NotImplementedError

        
    def add_rev(self, audio, length):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        if rir.shape[1] <= length:
            rep = length//rir.shape[1] +1
            rir = np.tile(rir, (1,rep))
            rir = rir[:,:length]
        else :
            rir = rir[:,:length]
        return signal.convolve(audio, rir, mode='full')[:,:length]

    def add_noise(self, audio, noisecat, length):
        clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            if noiseaudio.shape[0] <= length:
                rep = length//noiseaudio.shape[0] +1
                noiseaudio = np.tile(noiseaudio, rep)
                noiseaudio = noiseaudio[:length]
            else :
                noiseaudio = noiseaudio[:length]
            noiseaudio = np.stack([noiseaudio],axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio