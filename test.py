import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
import os
classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def test():
    parser = ArgumentParser()
    parser.add_argument('--test-path', type=str, default='./test/',
                        help='testing dataset path (default: ./test/)')
    parser.add_argument('--weight-path', type=str, default='./model_08.pt',
                        help='testing dataset path (default: ./model.pth)')
    parser.add_argument('--cuda', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    # dataset argument
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')
    # training argument
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    

    args = parser.parse_args()
    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    model = torch.load(args.weight_path)
    model = model.to(device)
    model.eval()


    print("Start Inference!!")
    dirs = os.listdir( args.test_path )



    for file in dirs:
        image = Image.open(Path(args.test_path).joinpath(file))
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.cuda(args.cuda))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        print(f"{file} Predicted:",classes[preds[0]])
        # if preds[0] > 12:
        #     preds[0] = 0 
        #     cnt+=1


if __name__ == '__main__':
    test()
