import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from new_models import SiameseLSTMClassifier, DoubleLSTMClassifier
import load_data as d
from sklearn.metrics import roc_auc_score, roc_curve
import csv


def read_data():
    pass


def train_autoencoder():
    pass


def find_most_similar(tcr):
    pass


def predict_peptide(tcr):
    pass


# train TCR autoencoder

# given test TCR
# find the closest TCR (that binds to peptide)
# take this peptide for prediction

