import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import DBN_Exact_Inference as Inference

logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

training_data = Inference.training_data

X = [training_data[]

logreg.fit()