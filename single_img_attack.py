import numpy as np
from PIL import Image
import numpy as np
import torch
from torch._C import device
from torchvision.utils import save_image
from torchvision import transforms
from torch.autograd import Variable
from pathlib import Path
from bat.attacks import SimBA
from bat.apis.deepapi import VGG16Cifar10
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.hop_skip_jump_attack import  hop_skip_jump_attack
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from models import VGG
import os
from argparse import ArgumentParser
from torch import nn

classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_transform = transforms.Compose([transforms.ToTensor()])
test_path = 'test'
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

def testing(PTH_file):

    checkpoint = torch.load(PTH_file)
    model = VGG('VGG19').to(device)
    model = torch.nn.DataParallel(model)  
    model.load_state_dict(checkpoint['net'])
    model.eval()
    Softmax = nn.Softmax()
    print("Start Inference!!")
    dirs = os.listdir(test_path)
    for file in dirs:
        image = Image.open(Path(test_path).joinpath(file))
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.to(device))
        outputs = model(inputs)
        probability = list(Softmax(outputs.data))
        _, preds = torch.max(outputs.data, 1)
        print(f"{file} Predicted:",classes[preds[0]],"Probability: ",probability[0][preds[0]].item())

def simba_att(img):
    # Load Image [0.0, 1.0]
    x = np.asarray(Image.open(img).resize((32, 32))) / 255.0

    # Initialize the Cloud API Model
    DEEP_API_URL = 'https://api.wuhanstudio.uk'
    model = VGG16Cifar10(DEEP_API_URL + "/vgg16_cifar10")

    # SimBA Attack
    simba = SimBA(model)
    x_adv = simba.attack(x, epsilon=0.1, max_it=1000)
    img = Image.fromarray((x_adv * 255).astype(np.uint8)).save('test/simba_adv1.jpg')

    # Distributed SimBA Attack
    x_adv = simba.attack(x, epsilon=0.1, max_it=1000, distributed=True , batch=50, max_workers=10)
    img = Image.fromarray((x_adv * 255).astype(np.uint8)).save('test/simba_adv2.jpg')

def fgsm(model,inputs,eps):

    x_fgm = fast_gradient_method(model, inputs, eps, np.inf)
    
    return x_fgm

def pgd(model,inputs,eps,step_size=0.01,num_iter=40):

    x_pgd = projected_gradient_descent(model, inputs, eps, step_size, num_iter, np.inf)
    return x_pgd
    # :param model_fn: a callable that takes an input tensor and returns the model logits.
    # :param x: input tensor.
    # :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    # :param eps_iter: step size for each attack iteration
    # :param nb_iter: Number of attack iterations.
    # :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    
def Ifgsm(model,x,epsilon,num_iter = 20):
    # def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    alpha = epsilon/num_iter
    for i in range(num_iter):
        x_adv = fgsm(model,x_adv,alpha)
        x_adv = torch.min(torch.max(x_adv,x-epsilon),x+epsilon)
    return x_adv


def PGDL1(model,inputs):
    x_pgdl1 = sparse_l1_descent(model, inputs)
    return x_pgdl1

def SPSA(model,inputs,eps,num_iter=5):
    x_spsa = spsa(model, inputs, eps,nb_iter=num_iter)
    return x_spsa


if __name__ == '__main__':
    img = 'test/plane.png'
    PTH_file = 'checkpoints/Parallel/vgg19.pth'
    eps = 8/255

    checkpoint = torch.load(PTH_file)
    model = VGG('VGG19').to('cuda')
    model = torch.nn.DataParallel(model)  
    model.load_state_dict(checkpoint['net'])
    model.eval()

    image = Image.open(Path(img))
    image = data_transform(image).unsqueeze(0)
    inputs = Variable(image.cuda())

    print("start attack PGDL1 ~")
    x_pgdl1 = sparse_l1_descent(model, inputs)
    save_image(x_pgdl1,'test/pgdl1.png')
    print("start attack SPSA ~")
    x_spsa = spsa(model, inputs, eps,nb_iter=5)
    save_image(x_spsa,'test/spsa.png')

    
    print("start attack fgsm ~")
    x_fgsm = fgsm(model,inputs,eps)
    save_image(x_fgsm,'test/fgsm_adv.png')
    print("start attack PGD ~")
    x_pgd = pgd(model,inputs,eps)
    save_image(x_pgd,'test/pgd_adv.png')
    print("start attack IFGSM ~")
    x_Ifgsm = Ifgsm(model,inputs,eps)
    save_image(x_Ifgsm,'test/Ifgsm_adv.png')
    # simba_att(img)

    testing(PTH_file)
