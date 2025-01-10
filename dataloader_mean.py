import glob, numpy, os, random, sys, soundfile, torch, cv2, wave
from scipy import signal
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
import torchaudio
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
#from dataloader import * 
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import sys 

def init_loader(args):

	#train_dataset = AV_dataset(opt = args)
	#val_dataset = AV_dataset(opt = args, mode='test')

	trainloader = train_loader(opt = args)
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers, drop_last = True)
	evalloader = eval_loader(opt = args, mode='test')
	args.evalLoader = torch.utils.data.DataLoader(evalloader, batch_size = 1, shuffle = False, num_workers = args.num_workers, drop_last = False)
	return args

def getVggoud_proc(filename, idx=None):

	audio_length = 1
	samples, samplerate = sf.read(filename)

	if samples.shape[0] > 16000*(audio_length+0.1):
		sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
		samples = samples[sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]

	else:
		# repeat in case audio is too short
		samples = np.tile(samples,int(self.opt.audio_length))[:int(16000*self.opt.audio_length)]

	samples[samples > 1.] = 1.
	samples[samples < -1.] = -1.

	frequencies, times, spectrogram = signal.spectrogram(samples, samplerate, nperseg=512,noverlap=353)
	spectrogram = np.log(spectrogram+ 1e-7)

	mean = np.mean(spectrogram)
	std = np.std(spectrogram)
	spectrogram = np.divide(spectrogram-mean,std+1e-9)
	
	return torch.tensor(spectrogram).unsqueeze(0).float()



def _wav2fbank(opt, filename, filename2=None, idx=None):
	# mixup
	if filename2 == None:
		waveform, sr = torchaudio.load(filename)
		waveform = waveform - waveform.mean()
	# mixup
	else:
		waveform1, sr = torchaudio.load(filename)
		waveform2, _ = torchaudio.load(filename2)

		waveform1 = waveform1 - waveform1.mean()
		waveform2 = waveform2 - waveform2.mean()

		if waveform1.shape[1] != waveform2.shape[1]:
			if waveform1.shape[1] > waveform2.shape[1]:
				# padding
				temp_wav = torch.zeros(1, waveform1.shape[1])
				temp_wav[0, 0:waveform2.shape[1]] = waveform2
				waveform2 = temp_wav
			else:
				# cutting
				waveform2 = waveform2[0, 0:waveform1.shape[1]]

		# sample lambda from uniform distribution
		#mix_lambda = random.random()
		# sample lambda from beta distribtion
		mix_lambda = np.random.beta(10, 10)
		mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
		waveform = mix_waveform - mix_waveform.mean()
		

	## yb: align ##
	if waveform.shape[1] > 16000*(opt.audio_length+0.1):
		sample_indx = np.linspace(0, waveform.shape[1] -16000*(opt.audio_length+0.1), num=10, dtype=int)
		waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*opt.audio_length)]
	## align end ##


	if opt.vis_encoder_type == 'vit':
		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
		# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
	elif opt.vis_encoder_type == 'swin':
		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

	########### ------> very important: audio normalized
	#fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
	### <--------
	if opt.vis_encoder_type == 'vit':
		target_length = int(1024 * (1/10)) ## for audioset: 10s
	elif opt.vis_encoder_type == 'swin':
		target_length = 192 ## yb: overwrite for swin
	# target_length = 512 ## 5s
	# target_length = 256 ## 2.5s
	n_frames = fbank.shape[0]

	p = target_length - n_frames

	# cut and pad
	if p > 0:
		m = torch.nn.ZeroPad2d((0, 0, 0, p))
		fbank = m(fbank)
	elif p < 0:
		fbank = fbank[0:target_length, :]
	if filename2 == None:
		return fbank, 0
	else:
		return fbank, mix_lambda

class train_loader(object):
	def __init__(self, opt, mode='train'):
		self.opt = opt
		self.train_path = opt.train_path
	
		#self.frame_len = frame_len * 160 + 240
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(self.opt.musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(self.opt.rir_path,'*/*/*.wav'))
		self.data_list = []
		self.data_label = []
		lines = open(self.opt.train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))		
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = line.split()[1]
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

		mapping_file = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/meta_info/voxceleb1_meta/vox1_meta.csv"
		self.audio_path = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1/"
		vid_data = pd.read_csv(mapping_file, header=None)
		video_data = vid_data.to_dict("list")[0][1:]
		self.ids_to_names = {}
		for item in video_data:
			item = item.split('\t')
			item_id = item[0]
			item_name = item[1]
			self.ids_to_names[item_name] =  item_id


		if self.opt.vis_encoder_type == 'vit':
			self.norm_mean = -6.2668
			self.norm_std = 3.2946
		### <----
		
		elif self.opt.vis_encoder_type == 'swin':
			## ---> yb calculate: AVE dataset for 192
			self.norm_mean =  -7.020560264587402 # -4.984795570373535
			self.norm_std =  3.620952606201172 #3.7079780101776123
			## <----

		if self.opt.vis_encoder_type == 'vit':
			self.my_normalize = Compose([
				# Resize([384,384], interpolation=Image.BICUBIC),
				Resize([224,224], interpolation=Image.BICUBIC),
				# Resize([192,192], interpolation=Image.BICUBIC),
				# CenterCrop(224),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
		elif self.opt.vis_encoder_type == 'swin':
			self.my_normalize = Compose([
				Resize([192,192], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])

	def __getitem__(self, index):
		file = self.data_list[index]
		label = self.data_label[index]
		segments = self.load_wav(file_name = file)
		segments = torch.FloatTensor(numpy.array(segments))
		#faces    = self.load_face(file = file)
		#faces = torch.FloatTensor(numpy.array(faces))
		return segments, label

	def load_wav(self, file_name):
		total_audio = []
		for audio_sec in range(10):
			comps = os.path.normpath(file_name).split(os.sep)
			wave_file_path = os.path.join(self.opt.train_audio_folder,self.ids_to_names[comps[0]],comps[2],comps[3].zfill(9))
			fbank, mix_lambda = _wav2fbank(self.opt, wave_file_path, idx=audio_sec)
			total_audio.append(fbank)
		total_audio = torch.stack(total_audio)
		return total_audio


	def load_face(self, file):
		comps1 = os.path.normpath(file.split(" ")[0]).split(os.sep)
		frames = glob.glob("%s/*.jpg"%(os.path.join(self.opt.train_video_folder, os.path.join(comps1[0], comps1[2], comps1[3][:-4]))))
		print(len(frames))
		sys.exit()

		sample_indx = np.linspace(1, len(frames) , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]









			tmp_img = torchvision.io.read_image(self.opt.video_folder+'/'+file_name+'/'+ str("{:04d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)





		frame_path = random.choice(frames)
		#total_num_frames = len(frames)
		try:
			tmp_img = torchvision.io.read_image(frame_path)/255
		except:
			tmp_img = torch.zeros(3,224,224)
		total_img = self.my_normalize(tmp_img)
		#if len(frames)>1:
		#	sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		#	total_img = []
		#	for vis_idx in range(10):
		#		tmp_idx = sample_indx[vis_idx]
		#		tmp_img = torchvision.io.read_image(self.opt.train_video_folder+'/'+file[:-4]+'/'+ str(tmp_idx)+ '.jpg')/255
		#		tmp_img = self.my_normalize(tmp_img)
		#		total_img.append(tmp_img)
		#	total_img = torch.stack(total_img)
		return total_img

	def __len__(self):
		return len(self.data_list)

	def face_aug(self, face):		
		global_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.GaussianBlur(kernel_size=(5, 9),sigma=(0.1, 5)),
			transforms.RandomGrayscale(p=0.2)
		])
		return global_transform(face)

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

class eval_loader(object):
	def __init__(self, opt, mode):  
		self.opt = opt      
		self.data_list, self.data_length = [], []
		self.eval_path = self.opt.eval_path
		#self.num_eval_frames = self.opt.num_eval_frames
		lines = open(self.opt.eval_list).read().splitlines()
		for line in lines:
			data = line.split()
			self.data_list.append(data[-2])
			self.data_length.append(float(data[-1]))

		inds = numpy.array(self.data_length).argsort()
		self.data_list, self.data_length = numpy.array(self.data_list)[inds], \
										   numpy.array(self.data_length)[inds]
		self.minibatch = []
		start = 0
		while True:
			frame_length = self.data_length[start]
			minibatch_size = max(1, int(100 // frame_length)) 
			end = min(len(self.data_list), start + minibatch_size)
			self.minibatch.append([self.data_list[start:end], frame_length])
			if end == len(self.data_list):
				break
			start = end

		mapping_file = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/meta_info/voxceleb1_meta/vox1_meta.csv"
		self.audio_path = "/misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1/"
		vid_data = pd.read_csv(mapping_file, header=None)
		video_data = vid_data.to_dict("list")[0][1:]
		self.ids_to_names = {}
		for item in video_data:
			item = item.split('\t')
			item_id = item[0]
			item_name = item[1]
			self.ids_to_names[item_name] = item_id  

		if self.opt.vis_encoder_type == 'vit':
			self.norm_mean = -6.2668
			self.norm_std = 3.2946
		### <----
		
		elif self.opt.vis_encoder_type == 'swin':
			## ---> yb calculate: AVE dataset for 192
			## ---> yb calculate: AVE dataset for 192
			self.norm_mean =  -7.020560264587402 # -4.984795570373535
			self.norm_std =  3.620952606201172 #3.7079780101776123
			## <----
			
		if self.opt.vis_encoder_type == 'vit':
			self.my_normalize = Compose([
				# Resize([384,384], interpolation=Image.BICUBIC),
				Resize([224,224], interpolation=Image.BICUBIC),
				# Resize([192,192], interpolation=Image.BICUBIC),
				# CenterCrop(224),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
		elif self.opt.vis_encoder_type == 'swin':
			self.my_normalize = Compose([
				Resize([192,192], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])


	def __getitem__(self, index):
		data_lists, frame_length = self.minibatch[index]
		filenames, segments, faces = [], [], []

		for num in range(len(data_lists)):
			file_name = data_lists[num]
			filenames.append(file_name)

			comps1 = os.path.normpath(file_name).split(os.sep)
			audio, mix_lambda = _wav2fbank(self.opt, self.norm_mean, self.norm_std, os.path.join(self.opt.train_audio_folder,self.ids_to_names[comps1[0]],comps1[2],comps1[3].zfill(9)))
			segments.append(audio)
			#total_audio = []
			#for audio_sec in range(10):
			#	fbank, mix_lambda = _wav2fbank(os.path.join('/misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1/', file_name), idx=audio_sec)
			#	total_audio.append(fbank)
			#total_audio = torch.stack(total_audio)


			comps2 = os.path.normpath(file_name.split(" ")[0]).split(os.sep)
			frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path,comps2[0], comps2[2], comps2[3][:-4])))

			#comps1 = os.path.normpath(file_name).split(os.sep)
			#frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path, self.ids_to_names[comps1[0]], comps1[1],comps1[2][:-4].lstrip('0'))))				


			frame_path = random.choice(frames)

			#total_num_frames = len(frames)
			
			tmp_img = torchvision.io.read_image(frame_path)/255
			total_img = self.my_normalize(tmp_img)
			faces.append(total_img)
			#total_num_frames = len(frames)
			#sample_indx = np.linspace(1, total_num_frames , num=5, dtype=int)
			#total_img = []
			#for vis_idx in range(10):
			#	tmp_idx = sample_indx[vis_idx]
			#	tmp_img = torchvision.io.read_image(self.eval_path+'/'+ self.ids_to_names[comps1[0]] + '/' + comps1[1] + '/' + comps1[2][:-4].lstrip('0') +'/'+ str("{:04d}".format(tmp_idx))+ '.jpg')/255
			#	tmp_img = self.my_normalize(tmp_img)
			#	total_img.append(tmp_img)
			#total_img = torch.stack(total_img)
		segments = torch.stack(segments)
		faces = torch.stack(faces)
		return segments, faces, filenames

	def __len__(self):
		return len(self.minibatch)