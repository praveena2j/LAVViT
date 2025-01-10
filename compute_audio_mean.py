import torch, sys, os, numpy, soundfile, time, pickle, cv2, glob, random, scipy
from tqdm import tqdm
import torch.nn as nn
from tools import *
from einops import rearrange, repeat
from loss import *
from audiomodel import *
from visualmodel import *
from models.ASP import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler
from models.LAVISH import MMIL_Net


def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.model = MMIL_Net(args).to('cuda')
		#self.asp = Attentive_Statistics_Pooling(768)

		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
		for name, param in self.model.named_parameters():
				
			param.requires_grad = False
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num

			if 'ViT'in name or 'swin' in name:
				if 'norm' in name and args.is_vit_ln:
					param.requires_grad = bool(args.is_vit_ln)
					total_params += tmp
					train_params += tmp
				else:
					param.requires_grad = False
					total_params += tmp
					
			# ### <----
			# if  'audio_adapter_blocks' in name :  #'my_blocks' in name or 'my_mlp_forward' in name or 'adapter' in name or 'my_mlp_forward' in name 
			# 	print(name)
			# 	param.requires_grad = False
			# 	train_params += tmp
			# 	additional_params += tmp
			# 	total_params += tmp

			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name, param.shape , tmp)
			# elif 'norm' in name:
			# 	param.requires_grad = True
			# 	train_params += tmp
			# print('########### train layer:', name)
			elif 'mlp_class' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			if 'mlp_class' in name:
				param_group.append({"params": param, "lr":args.lr_mlp})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))

		#self.frozen_model = freeze(self.model, args)
		#self.speaker_encoder = ECAPA_TDNN(model = args.model_a).cuda()
		self.loss    = AAMsoftmax(n_class = args.n_class, m = args.margin_a, s = args.scale_a, c = 512).cuda()	

		for name, param in self.loss.named_parameters():
			param_group.append({"params": param, "lr":args.lr_loss})
				

		#self.face_encoder    = IResNet(model = args.model_v).cuda()
		#self.face_loss       = AAMsoftmax(n_class =  args.n_class, m = args.margin_v, s = args.scale_v, c = 512).cuda()
		self.optim           = torch.optim.Adam(param_group) #self.parameters())
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.decay_epoch, gamma=args.decay)
		#self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		#print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
		#print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
		

	def train_network(self, args):
		mean = []
		std = []
		for batch_idx, (speech, labels) in tqdm(enumerate(args.trainLoader, start = 1), total=len(args.trainLoader), position=0, leave=True):

			b,t,w,h = speech.shape
			audio_spec =  rearrange(speech, 'b t w h -> (b t) (w h)')

			cur_mean = torch.mean(audio_spec, dim=-1)
			cur_std = torch.std(audio_spec, dim=-1)
			mean.append(cur_mean)
			std.append(cur_std)

		print('mean: ',torch.hstack(mean).mean().item(),'std: ',torch.hstack(std).mean().item())
		set_trace()
		sys.exit()

		
	def eval_network(self, args):
		self.eval()
		scores_av, scores_a, scores_v, labels, res = [], [], [], [], []
		embeddings = {}
		lines = open(args.eval_trials).read().splitlines()
		for a_data, v_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():
				embedding = self.model(a_data.cuda(), v_data.cuda(), labels, stage='eval')
				embedding = embedding.squeeze(1)
				for num in range(len(filenames)):
					filename = filenames[num][0]
					av = torch.unsqueeze(embedding[num], dim = 0)
					#v = v_embedding[:,num,:]
					embeddings[filename] = F.normalize(av, p=2, dim=1)

		for line in tqdm.tqdm(lines):			
			a1 = embeddings[line.split()[1]]
			a2 = embeddings[line.split()[2]]
			score_a = torch.mean(torch.matmul(a1, a2.T)).detach().cpu().numpy()
			scores_a.append(score_a)
			labels.append(int(line.split()[0]))

		for score in [scores_a]:
			EER = tuneThresholdfromScore(score, labels, [1, 0.1])[1]
			fnrs, fprs, thresholds = ComputeErrorRates(score, labels)
			minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
			res.extend([EER, minDCF])
		
		print('EER_a %2.4f, min_a %.4f\n'%(res[0], res[1]))
		args.score_file.write("EER_a %2.4f, min_a %.4f\n"%(res[0], res[1]))
		args.score_file.flush()
		return

	def save_parameters(self, args):
		torch.save(self.model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
		#if modality == 'A':			
		#	model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))
		#if modality == 'V':
		#	model = OrderedDict(list(self.face_encoder.state_dict().items()) + list(self.face_loss.state_dict().items()))
		#torch.save(model, path)
