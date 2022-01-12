from absl import app, flags
from torchvision.transforms.transforms import Normalize
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
from models import *
from torch.autograd import Variable

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.hop_skip_jump_attack import  hop_skip_jump_attack
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)


def main(args):
    # Load test data

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    # checkpoint = torch.load('checkpoints/Parallel/vgg19.pth') 
    # model = VGG('VGG19').to('cuda') # data parallel moment cuda cannot change to 'device'
    # model = torch.nn.DataParallel(model) 

    checkpoint = torch.load('checkpoints/MobileNet.pth')
    model = MobileNet().to(device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0
    correct_fgm,correct_pgd,correct_pgdl1,correct_spsa=0,0,0,0
    nb_test = 0
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        nb_test += targets.size(0)
        
        if args.attack_type == 'FGSM': 
            x_fgm = fast_gradient_method(model, inputs, args.eps, np.inf)
            _, y_pred_fgm = model(x_fgm).max(1) # model prediction on FGM adversarial examples
            correct_fgm += y_pred_fgm.eq(targets).sum().item()
        elif args.attack_type == 'PGD':
            x_pgd = projected_gradient_descent(model, inputs, args.eps, 0.01, 40, np.inf)
            _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
            correct_pgd += y_pred_pgd.eq(targets).sum().item()

        elif args.attack_type == 'PGDL1':
            x_pgdl1 = sparse_l1_descent(model, inputs) #args.eps
            _, y_pred_pgdl1 = model(x_pgdl1).max(1)  # model prediction on PGD adversarial examples
            correct_pgdl1 += y_pred_pgdl1.eq(targets).sum().item()
        
        elif args.attack_type == 'SPSA': # Cost a lot of time
            x_spsa = spsa(model, inputs, args.eps,nb_iter=5)
            _, y_pred_spsa = model(x_spsa).max(1)  # model prediction on SPSA adversarial examples
            correct_spsa += y_pred_spsa.eq(targets).sum().item()

        
        
    print("test acc on clean examples (%): {:.3f}".format(correct / nb_test * 100.0))
    if args.attack_type == 'FGSM':
        print("test acc on FGSM adversarial examples (%): {:.3f}".format(correct_fgm / nb_test * 100.0))
    elif args.attack_type == 'PGD':
        print("test acc on PGD adversarial examples (%): {:.3f}".format(correct_pgd / nb_test * 100.0))
    elif args.attack_type == 'PGDL1':
        print("test acc on pgdl1 adversarial examples (%): {:.3f}".format(correct_pgdl1 / nb_test * 100.0))
    elif args.attack_type == 'SPSA':
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(correct_spsa / nb_test * 100.0))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--cuda', type=int, default=1,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size (default: 128)')
    parser.add_argument('--eps', type=float, default=8/255,
                        help='Total epsilon for FGM and PGD attacks (default: 8/255)')
    parser.add_argument('--attack_type', type=str, default='FGSM',
                        help='Adversarial attack type.(FGSM,PGD,IFGSM,CW,HSJ,PGDL1)') # CW = carlini Wagner, HSJ = hop_skip_jump_attack  
    parser.add_argument('--checkpoint', type=str, default='model_08.pt',
                        help='Checkpoint')              
    args = parser.parse_args()

    main(args)
