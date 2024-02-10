#!/usr/bin/env python3
from .matrix import Matrix
from .neural_network import NeuralNetwork
from .training_data import XOR, AND, NAND, OR
import random
from datetime import datetime

#random.seed(69)
random.seed(datetime.now().timestamp() )
