'''
Use clustering and classification algorithms to get insights from the dataset wines_Header.csv .

The file wines_Metadata.csv contains the "metadata" of the dataset, meaning information about the variables in the dataset.
'''
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

# read
headers = pd.read_csv('wines_Header.csv')
meta_data = pd.read_csv('wines_Metadata.csv')

print(headers.head())


