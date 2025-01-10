import argparse, glob, os, torch, warnings, time
from tools import *
#from trainer import *
from compute_audio_mean import *
from dataloader_mean import *
from base_options import BaseOptions
args = BaseOptions().parse()

args = init_system(args)
args = init_loader(args)

print("Data loaded")

## Init trainer
s = init_trainer(args)

## Evaluate only
if args.mode == 'Eval':
	s.eval_network(args)
	quit()

## Train only
if args.mode == 'train':
	while args.epoch < args.max_epoch:
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			s.save_parameters(args)
			#s.save_parameters(args.model_save_path_v + "/model_%04d.model"%args.epoch, 'V')
			s.eval_network(args)
		args.epoch += 1
	quit()
