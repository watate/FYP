#This script is used to test functions before deploying them in other scripts

#Seaborn Documentation for relplot:
	#https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot

#Changing wide-format data to long-format (which seaborn prefers)
	#use pandas.melt()
	#https://stackoverflow.com/questions/52308749/how-do-i-create-a-multiline-plot-using-seaborn

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

############## VISUALIZATION ##################
#Initial settings
sb.set(style="darkgrid")
fig_dpi = 300

constant = 100
a = np.linspace(1, constant, constant)
b = np.linspace(1, 200, constant)
c = np.random.randint(0,100,size=(100,))
d = np.random.randint(0,100,size=(100,))

data = {"a": a,
		"b": b,
		"c": c,
		"d": d,}

#Generate dataframe
df = pd.DataFrame(data, columns=["a", "b", "c", "d"])

#Generating line graphs
g = sb.relplot(x="a", y="value", hue="variable", kind="scatter", data=pd.melt(df, id_vars=['a']))
#g.fig.autofmt_xdate()
fig = g.fig