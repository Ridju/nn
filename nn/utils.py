#!/usr/bin/env python3
import math
import random

def random_float(start=0, stop=1):
    return random.uniform(start,stop)

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))
