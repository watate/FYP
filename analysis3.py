import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#list
result = list()

#get data
for e in tf.train.summary_iterator('/home/watate/Documents/AsDDPG/ddpg_summary/20200106-214225/events.out.tfevents.1578318156.ubuntu'):
    for v in e.summary.value:
        if v.tag == 'Result':
            result.append(v.simple_value)

crash = 0
get_target = 0
time_out = 0
for i in result:
	if i == 1: time_out = time_out + 1;
	if i == 2: crash = crash + 1;
	if i == 3: get_target = get_target + 1;

print("Crash: {}".format(crash))
print("Get Target: {}".format(get_target))
print("Time Out: {}".format(time_out))