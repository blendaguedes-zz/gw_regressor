import pandas as pd
from data_maker import GWGeneratorWiki
from sklearn import preprocessing
import numpy as np

#read the `masses` file
masses = pd.read_csv("masses", delimiter=' ')

#drop the third unnecessary column
masses = masses.drop(columns='NaN')

gwg = GWGeneratorWiki()
mass_ratio = []

#calculationg mass ratio
for m1, m2 in zip(masses.m1, masses.m2):
    mass_ratio.append(gwg.calculate_mass_ratio(m1, m2))

mass_ratio = np.array(mass_ratio)

#normalizing mass ratio
#mass_ratio_normalized = mass_ratio / np.linalg.norm(mass_ratio)
#print(max(mass_ratio_normalized))
#f= open("guru99.txt","w+")

waves_histogram = pd.read_csv("full_size_waves_hist", delimiter=' ', nrows=10, header=None)
#print(len(waves_histogram))
for hist in waves_histogram.values:
   wave_form_histogram_normalized = hist / np.linalg.norm(hist)
   print(wave_form_histogram_normalized)

