from art.defences.postprocessor import HighConfidence
from art.defences.postprocessor import ClassLabels
import numpy as np

H = HighConfidence(cutoff = 0.25, apply_fit= False, apply_predict = True)
l1 = [[0.3,0.3,0.4],[0.1,0.4,0.2]]
arr = np.array(l1)
print(arr, arr.shape)
# l = H(arr)
# print(l, l.shape)

c = ClassLabels(apply_fit = False, apply_predict = True)
l = c(arr)
print(l)
# l2 = [[1,5,8],[18,9,2]]
# arr_d = np.array(l2)
# print(type(arr_d), arr_d.shape)