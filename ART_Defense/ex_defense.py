
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.defences.detector.evasion import BinaryInputDetector,BinaryActivationDetector
from art.defences.preprocessor import SpatialSmoothing
from art.defences.trainer import AdversarialTrainerFBFPyTorch
from art.attacks.evasion import ZooAttack,TargetedUniversalPerturbation
# Step 0: Define the neural network model, return logits instead of activation in forward method

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)
# Step 6: Generate adversarial test examples

print("Normal Classifier")

classifier.fit(x_train, y_train, batch_size=512, nb_epochs=3)
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

attack = FastGradientMethod(estimator=classifier, eps=0.1)
x_test_adv = attack.generate(x=x_test)
# attack = TargetedUniversalPerturbation(classifier)
# x_test_adv = attack.generate(x=x_test,y = y_test)

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


# Initalize the SpatialSmoothing defence. 
ss = SpatialSmoothing(window_size=3,clip_values=(min_pixel_value, max_pixel_value),apply_fit = True)
x_test_adv_def, _ = ss(x_test_adv)
predictions = classifier.predict(x_test_adv_def)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples after SpatialSmoothing: {}%".format(accuracy * 100))

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

print("AdversarialTrainerFBFPyTorch~")

# Adversarial Training Fast is Better than Free
FBF = AdversarialTrainerFBFPyTorch(classifier,eps = 0.1)
FBF.fit(x_train, y_train, batch_size=512, nb_epochs=5)
predictions = FBF.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# attack = TargetedUniversalPerturbation(classifier)
# x_test_adv = attack.generate(x=x_test,y = y_test)
attack = FastGradientMethod(estimator=classifier, eps=0.1)
x_test_adv = attack.generate(x=x_test)

predictions = FBF.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
