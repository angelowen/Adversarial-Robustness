import numpy as np
from PIL import Image
import numpy as np
from bat.attacks import SimBA
from bat.apis.deepapi import VGG16Cifar10

# Load Image [0.0, 1.0]
x = np.asarray(Image.open('test/plane.png').resize((32, 32))) / 255.0

# Initialize the Cloud API Model
DEEP_API_URL = 'https://api.wuhanstudio.uk'
model = VGG16Cifar10(DEEP_API_URL + "/vgg16_cifar10")

# SimBA Attack
simba = SimBA(model)
x_adv = simba.attack(x, epsilon=0.5, max_it=1000)
img = Image.fromarray((x_adv * 255).astype(np.uint8)).save('test/adv1.jpg')

# Distributed SimBA Attack
x_adv = simba.attack(x, epsilon=0.5, max_it=1000, distributed=True , batch=50, max_workers=10)
img = Image.fromarray((x_adv * 255).astype(np.uint8)).save('test/adv2.jpg')

