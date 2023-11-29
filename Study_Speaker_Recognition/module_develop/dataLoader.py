'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
import pickle as pk
import msgpack
import msgpack_numpy as m
m.patch()

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			tmp_file_name = line.split('/')[2]
			if tmp_file_name.endswith('wav') :
				file_name     = os.path.join(train_path, line.split()[1].replace('.wav','_noise_0.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
			elif tmp_file_name.endswith('msgpack') :
				if len(tmp_file_name) < 15 :
					file_name     = os.path.join(train_path, line.split()[1].replace('.msgpack','_noise_0.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
				else :                    
					file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		with open(self.data_list[index], 'rb') as f:
			mel = msgpack.unpack(f)
		mel1 = msgpack.unpackb(mel, object_hook = m.decode)
		return torch.FloatTensor(mel1), self.data_label[index]  ##### extractor

	def __len__(self):
		return len(self.data_list)