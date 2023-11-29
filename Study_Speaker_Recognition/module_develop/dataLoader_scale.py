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
			self.data_list_sub  = []
			speaker_label = dictkeys[line.split()[0]]
			tmp_file_name = line.split('/')[2]
			if tmp_file_name.endswith('wav') :
				file_name     = os.path.join(train_path, line.split()[1].replace('.wav','_noise_0.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
				file_name2     = os.path.join(train_path, line.split()[1].replace('.wav','_2scale.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
				file_name3     = os.path.join(train_path, line.split()[1].replace('.wav','_4scale.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
			elif tmp_file_name.endswith('msgpack') :
				if len(tmp_file_name) < 15 :
					file_name     = os.path.join(train_path, line.split()[1].replace('.msgpack','_noise_0.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함
					file_name2     = os.path.join(train_path, line.split()[1].replace('.msgpack','_2scale.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함                    
					file_name3     = os.path.join(train_path, line.split()[1].replace('.msgpack','_4scale.msgpack'))  # _noise_ = musan, rir 포함, # _noise_0 = musan, rir 미포함                    
				else :                    
					file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list_sub.append(file_name) 
			self.data_list_sub.append(file_name2)              
			self.data_list_sub.append(file_name3)              
			self.data_list.append(self.data_list_sub)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		with open(self.data_list[index][0], 'rb') as f:
			mel = msgpack.unpack(f)
		mel1 = msgpack.unpackb(mel, object_hook = m.decode)
		with open(self.data_list[index][1], 'rb') as f2:
			mel2 = msgpack.unpack(f2)
		mel2 = msgpack.unpackb(mel2, object_hook = m.decode)        
		with open(self.data_list[index][2], 'rb') as f3:
			mel3 = msgpack.unpack(f3)
		mel3 = msgpack.unpackb(mel3, object_hook = m.decode)
		data_out = [torch.FloatTensor(mel1),torch.FloatTensor(mel2),torch.FloatTensor(mel3)]        
		return data_out, self.data_label[index]  ##### extractor

	def __len__(self):
		return len(self.data_list)