#Use clustering and classification algorithms to get insights from the dataset wines_Header.csv .
#The file wines_Metadata.csv contains the "metadata" of the dataset, meaning information about the variables in the dataset.

import numpy as np  
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

with open('wines_Header.csv', 'r') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {line.strip().count(',') + 1} fields")

