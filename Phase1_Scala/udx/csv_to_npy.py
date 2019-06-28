import tensorflow as tf
import numpy as np
import pandas as pd
import os

filename = '/Users/ashwinravishankar/Work/WineQuality/Dataset/winequality.csv'
df = pd.read_csv(filename)

train = df.sample(frac=0.8, random_state=0)
test = df.drop(train.index)

t_stats = train.describe()
t_stats.pop("quality")
t_stats = t_stats.transpose()

train_target = train.pop("quality")
test_target = test.pop("quality")

def normalize(x):
    return (x - t_stats['mean']) / t_stats['std']

normal_train = normalize(train)
normal_test = normalize(test)

train = train.values
test = test.values
t_stats = t_stats.values
train_target = train_target.values
test_target = test_target.values

save_dir = '/Users/ashwinravishankar/Work/WineQuality/Dataset/NPY'
np.save(os.path.join(save_dir,'train.npy'), train)
np.save(os.path.join(save_dir,'test.npy'), test)
np.save(os.path.join(save_dir,'t_stats.npy'), t_stats)
np.save(os.path.join(save_dir,'train_target.npy'), train_target)
np.save(os.path.join(save_dir,'test_target.npy'), test_target)

# x = np.load(os.path.join(save_dir,'train.npy'))
