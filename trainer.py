import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from audiomodel import *
from visualmodel import *
from models.ASP import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler
from models.LAVISH import MMIL_Net
#from drloss import cal_selfsupervised_loss

def init_trainer(args):
	args.epoch = 1
	s = trainer(args)

	if args.mode == 'Eval':
		savedmodel = 'exps/debug/models/Epoch1.pt'
		print("Model %s loaded from input state!"%(savedmodel))
		state_model = torch.load(savedmodel)
		s.model.load_state_dict(state_model['model_state_dict'])
		s.ASP_model.load_state_dict(state_model['ASP_model_state_dict'])
		state_model['ASP_model_state_dict']
		print(state_model['EER'])
		print(state_model['minDCF'])
	return s

class ASPModel(nn.Module):
	def __init__(self):
		super(ASPModel, self).__init__()
		self.asp = Attentive_Statistics_Pooling(512).to('cuda')
		self.linear = nn.Linear(1024, 512).to('cuda')

	def forward(self, AV_feats):
		AV_feats = self.asp(AV_feats)
		AV_embeddings = self.linear(AV_feats)
		return AV_embeddings


class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.model = MMIL_Net(args).to('cuda')
		self.ASP_model =  ASPModel().cuda()
		#self.asp = Attentive_Statistics_Pooling(512).to('cuda')
		#self.linear = nn.Linear(1024, 512).to('cuda')

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
		#self.drloss = cal_selfsupervised_loss

		for name, param in self.loss.named_parameters():
			param_group.append({"params": param, "lr":args.lr_loss})

		for name, param in self.ASP_model.named_parameters():
			param_group.append({"params": param, "lr":args.lr_loss})
		#self.face_encoder    = IResNet(model = args.model_v).cuda()
		#self.face_loss       = AAMsoftmax(n_class =  args.n_class, m = args.margin_v, s = args.scale_v, c = 512).cuda()
		self.optim           = torch.optim.Adam(param_group) #self.parameters())
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.decay_epoch, gamma=args.decay)
		#self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		#print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
		#print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
		
	def train_network(self, args):
		self.train()
		scaler = GradScaler()
		rand_train_idx = 11
		accum_iter = 8
		#self.scheduler.step(args.epoch - 1)
		index, top1, loss = 0, 0, 0
		cnt = 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()
		for batch_idx, (speech, face, labels) in enumerate(args.trainLoader, start = 1):

			#if batch_idx > 5:
			#	break
			self.optim.zero_grad()
			labels      = torch.LongTensor(labels).cuda()

			#face        = face.div_(255).sub_(0.5).div_(0.5)
			#with autocast():
			embedding = self.model([speech.cuda()], face.cuda(), rand_train_idx=rand_train_idx, stage='train')
			embedding = self.ASP_model(embedding)
			#embedding = self.asp(embedding)
			#embedding = self.linear(embedding)

			#embedding = embedding.squeeze(1)
			#a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = True)	
			#aloss, _ = self.speaker_loss.forward(a_embedding, labels)	
			#v_embedding   = self.face_encoder.forward(face.cuda())	
			loss, prec1 = self.loss.forward(embedding, labels)
			#loss_ssup, ssup_items = self.drloss(outputs, config, lambda_drloc)
			loss.backward()
			#loss += AVloss
			cnt += 1		
			
			#scaler.scale(aloss + vloss).backward()
			#scaler.step(self.optim)
			#scaler.update()

			if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(args.trainLoader)):
				self.optim.step()
				#scaler.update()
			index += len(labels)
			loss += loss.detach().cpu().item()
			top1 += prec1.detach().cpu().item()
			time_used = time.time() - time_start
			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f, ACC %f \r"%\
			(args.epoch, 100 * (batch_idx / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / batch_idx / 60, lr, loss/(batch_idx), top1/batch_idx))
			sys.stderr.flush()
		#self.scheduler.step()
		sys.stdout.write("\n")

		args.score_file.write("%d epoch, LR %f, LOSS %f, ACC %f \n"%(args.epoch, lr, loss/batch_idx, top1/batch_idx))
		args.score_file.flush()
		return
		
	def eval_network(self, args):
		self.eval()
		scores_av, scores_a, scores_v, labels, res = [], [], [], [], []
		embeddings = {}
		lines = open(args.eval_trials).read().splitlines()
		for a_data, v_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():
				v_data = v_data.squeeze(0)
				a_data = a_data.squeeze(0)
				embedding = self.model([a_data.cuda()], v_data.cuda(), stage='eval')
				embedding = self.ASP_model(embedding)
				#embedding = self.asp(embedding)
				#embedding = self.linear(embedding)

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
		return res[0], res[1]

	def save_checkpointer(self, args, EER, minDCF):
		torch.save({
			'model_state_dict': self.model.state_dict(), 
			'ASP_model_state_dict': self.ASP_model.state_dict(),
			'loss': self.loss.state_dict(),
			'optim': self.optim.state_dict(),
			'epoch': args.epoch,
			'scheduler': self.scheduler.state_dict(), 
			'EER' : EER,
			'minDCF' :minDCF,
		},
			args.model_save_dir + "checkpointer.pt")
		#if modality == 'A':			
		#	model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))
		#if modality == 'V':
		#	model = OrderedDict(list(self.face_encoder.state_dict().items()) + list(self.face_loss.state_dict().items()))
		#torch.save(model, path)


	def save_parameters(self, args, EER, minDCF):
		torch.save({
			'model_state_dict': self.model.state_dict(), 
			'ASP_model_state_dict': self.ASP_model.state_dict(),
			'EER' : EER,
			'minDCF' :minDCF,
		}, args.model_save_dir + "Epoch" + str(args.epoch) + ".pt")