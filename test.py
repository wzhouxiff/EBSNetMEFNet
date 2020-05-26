import argparse
import os, sys
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from math import log10
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from PIL import Image
import glob

# from dataset import dataset
from models import decision, hdrnet, hdrnet_nobn
from logger import Logger
import utils
from loss import SSIM
import pdb
import random
from torch.nn.functional import pad as tensor_pad

def arg_parse():
	parser = argparse.ArgumentParser(description='PyTorch EBSNetMEFNet')
	parser.add_argument('data', metavar='DIR', help='path to dataset')
	parser.add_argument('ckpt', metavar='DIR', help='path to checkpoints and list')
	parser.add_argument('--data-type', default='none', type=str,
						help='data type: night or day. (default: night)')
	parser.add_argument('--logdir', default='./logs', type=str, metavar='DIR',
						help='log dir')
	parser.add_argument('--results', default='./results', type=str, metavar='DIR',
						help='results')
	parser.add_argument('--score-path', default=None, type=str, metavar='DIR',
						help='save the decision score in a new list')

	args = parser.parse_args()
	return args

def print_args(args):
	print "=========================================="
	print "==========       CONFIG      ============="
	print "=========================================="
	for arg,content in args.__dict__.items():
	    print "{}:{}".format(arg,content)
	print "\n"

def test():
	global args, A
	args = arg_parse()
	print_args(args)

	ckpt = os.path.join(args.ckpt, args.data_type)
	# Get A
	A_path = os.path.join(ckpt, 'A.txt')
	with open(A_path) as fA:
		lines = fA.readlines()
		A = []
		for line in lines:
			line = line.strip().split(':')
			line = line[-1].strip().split(',')
			A.append([ int(v[1]) for v in line])
	print(A)

	num_actions = len(A)
	frames_num = len(A[0])
	print(num_actions, frames_num)

	print('===> Loading the network ...')
	if args.data_type == 'night':
		fusion = hdrnet.HDRNet(inc=frames_num*3, outc=3, is_BN=True)
	else:
		fusion = hdrnet.HDRNet(inc=frames_num*3, outc=3, is_BN=False)
	fusion.cuda()

	policy = decision.DecisionNet(pretrained=False, num_classes=num_actions)
	policy.cuda()

	criterion = nn.MSELoss().cuda()

	print '===> loading weights ...'
	policy_weights = os.path.join(ckpt, 'policy.pth')
	if policy_weights:
		if os.path.isfile(policy_weights):
			print("=====> loading checkpoint '{}".format(policy_weights))
			checkpoint = torch.load(policy_weights)
			policy.load_state_dict(checkpoint)
		else:
			print("=====> no checkpoint found at '{}'".format(policy_weights))

	fusion_weights = os.path.join(ckpt, 'fusion.pth')
	if fusion_weights:
		if os.path.isfile(fusion_weights):
			print("=====> loading checkpoint '{}".format(fusion_weights))
			checkpoint = torch.load(fusion_weights)
			fusion.load_state_dict(checkpoint)
		else:
			print("=====> no checkpoint found at '{}'".format(fusion_weights))

	TT = transforms.ToTensor()
	tmpT = transforms.ToPILImage()

	# The number of bins of illumination distribution
	bins_num = 32

	fin = open(os.path.join(ckpt, 'list.txt'))
	names = fin.readlines()
	print('test num: ', len(names))

	if not os.path.exists(args.results):
		os.makedirs(args.results)
	output_path = args.results

	fout = open(os.path.join(output_path, 'score_'+args.data_type+'.txt'), 'w')

	cnt = 0
	psnr_sum = 0.
	ssim_sum = 0.
	names = names
	for name in names:
		print('{}/{}'.format(cnt,len(names)))
		name = name.strip()
		
		target = Image.open(os.path.join(args.data, name, 'gt.png'))
		target = TT(target)
		target = torch.unsqueeze(target, 0)
		
		normal_path = glob.glob(os.path.join(args.data, name, 'normal*'))
		normal = Image.open(normal_path[0])
		normal = normal.resize((224, 224), resample=Image.BILINEAR)

		normal = np.array(normal)
		input = torch.zeros((bins_num*3, 4, 4))

		# 1*1
		hist = np.histogram(normal.flatten(), bins_num, [0, 255])[0] / (224. * 224.)
		hist = torch.from_numpy(hist.astype(np.float32))
		input[0:bins_num] = hist.view(-1, 1, 1).repeat(1, 4, 4)

		# 2*2
		for i in range(2):
			for j in range(2):
				hist = np.histogram(normal[i*112:(i+1)*112, j*112:(j+1)*112], bins_num, [0, 255])[0] /(112.*112.)
				hist = torch.from_numpy(hist.astype(np.float32))
				input[bins_num:bins_num*2, i*2:(i+1)*2, j*2:(j+1)*2] = hist.view(-1, 1, 1).repeat(1, 2, 2)

		# 4*4
		
		for i in range(4):
			for j in range(4):
				hist = np.histogram(normal[i*56:(i+1)*56, j*56:(j+1)*56], bins_num, [0, 255])[0] /(56.*56.)
				hist = torch.from_numpy(hist.astype(np.float32))
				input[bins_num*2:, i, j] = hist
		
		input = torch.unsqueeze(input, 0)

		normal = TT(normal)
		normal = torch.unsqueeze(normal, 0)

		# Get and sort exposure sequence
		exposures = glob.glob(os.path.join(args.data, name, 'step*'))
		ratio = [float(ex.split('/')[-1].split('_')[1]) for ex in exposures]
		idx = sorted(enumerate(ratio), key=lambda x:x[1])
		idx = [i[0] for i in idx]
		exposures = [exposures[i] for i in idx]

		imgs = []
		for e in exposures:
			img = Image.open(e)
			img = TT(img)
			imgs.append(img)
		imgs = torch.cat(imgs, dim=0)
		imgs = torch.unsqueeze(imgs, 0)

		fusion.eval()
		policy.eval()
		with torch.no_grad():
			normal_var = normal.cuda()
			prob = policy(normal_var, input.cuda())
			prob = F.softmax(prob, dim=1)
			action = torch.argmax(prob)
			print(torch.max(prob))
			print(action)
			print(A[action])
			
			pick_imgs = []
			for i in range(frames_num):
				pick_imgs.append(imgs[:,A[action][i]*3:(A[action][i]+1)*3, :, :])
			pick_imgs = torch.cat(pick_imgs, dim=1)

			target_var = target.cuda()

			fusion_output = fusion(pick_imgs.cuda())
			
			# PSNR
			mse = criterion(fusion_output.cuda(), target_var)
			psnr = 10 * torch.log10(1. / (mse+1e-5))
			psnr = psnr.data.cpu().item()
			psnr_sum += psnr

			# SSIM
			ssim = SSIM.ssim(fusion_output.cuda(), target_var)
			ssim = ssim.data.cpu().item()
			ssim_sum += ssim
			
			output = tmpT(torch.clamp(fusion_output.data.cpu()[0, :, :, :], 0, 1))
			output.save(os.path.join(output_path, name + '.png'))

			fout.write(name + ' ' + str(A[action][0]) + ' ' + str(A[action][1]) + ' ' + str(A[action][2]) + ' ' + str(psnr) + ' ' + str(ssim) + '\n')

		cnt += 1
	psnr_avg = psnr_sum / float(len(names))
	print('psnr: ', psnr_avg)
	ssim_avg = ssim_sum / float(len(names))
	print('ssim: ', ssim_avg)

	fout.write(str(psnr_avg) + ' ' + str(ssim_avg) + '\n')
	fout.close()

if __name__ == '__main__':
	test()