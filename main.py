import argparse, glob, os, torch, warnings, time
from tools import *
#from trainer import *
from trainer import *
from dataLoader import *
from base_options import BaseOptions
args = BaseOptions().parse()

args = init_system(args)
args = init_loader(args)

print("Data loaded")
def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs)/(total_epoch - warmup_epochs))
    return cur_weight

## Init trainer
s = init_trainer(args)

## Evaluate only
if args.mode == 'Eval':
	EER, minDCF = s.eval_network(args)
	quit()

## Train only
if args.mode == 'train':
	while args.epoch < args.max_epoch:
		args.init_lambda_drloc = _weight_decay(
                0.2, 
                args.epoch, 
                20, 
                300)
		s.train_network(args)
		if args.epoch % args.test_step == 0:
			EER, minDCF = s.eval_network(args)
			s.save_checkpointer(args, EER, minDCF)
			s.save_parameters(args, EER, minDCF)
			#s.save_parameters(args.model_save_path_v + "/model_%04d.model"%args.epoch, 'V')
		args.epoch += 1
	quit()
