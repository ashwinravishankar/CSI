# https://stackoverflow.com/questions/49335184/reuse-variables-and-model-encapsulated-in-class

from TensorflowClass import TFCLass_Regression
import itertools
import operator 

print("============================ Computing Statistics  ===============================")

#   Parse data one time to get all components to calculate statistics
import csv
file_path = '/Users/ashwinravishankar/Work/WineQuality/Dataset/winequality_train.csv'
data=[]
with open(file_path) as csvfile:
    reader=csv.reader(csvfile)
    header=next(reader)
    running_total = [0.0 for i in header]
    running_max = [float("-inf") for i in header]
    running_min = [float("inf") for i in header]
    running_count = 0
    for row in reader:
        row = list(map(float, row))
        running_total = list(map(operator.add, running_total, row))
        running_max = list(map(max, running_max, row))
        running_min = list(map(min, running_min, row))
        running_count = running_count+1

running_mean = [i/running_count for i in running_total]
running_minmax_diff = list(map(operator.sub, running_max, running_min))


##Send NumberOfFeatures=11, NumberOfTargetLabels=1, LearningRate=0.01
N_Features = len(header)-1
N_TargetLabel = 1
LearningRate=0.01

print("\n\n============================ Creating Model and Setting Parameters  ===============================")

model=TFCLass_Regression()
model.set_params(N_Features, N_TargetLabel, LearningRate)


#Set data Statistics on the worker - all values except last value which is the target value
model.set_stats(running_count, running_total[:-1], running_min[:-1], running_max[:-1], running_minmax_diff[:-1])

print("\n\n============================ Training Phase - Starting ===============================")

#  Training step
first_row=None
for i in range(30):
    print("Starting Iteration/Epoch {0}".format(i+1))
    with open(file_path) as csvfile:
        reader=csv.reader(csvfile)
        header=next(reader)
        first_row=next(reader)
        for row in reader:
            row=list(map(float,row))
            model.fit_train(row[:-1],row[-1])

print("============================ Training Phase - Completed  ===============================")

print("\n\n============================ Prediction Phase - Starting  ===============================")

# Prediction step
test_file='/Users/ashwinravishankar/Work/WineQuality/Dataset/winequality_test.csv'
with open(test_file) as csvfile:
    reader = csv.reader(csvfile)
    header=next(reader)
    for row in reader:
        row=list(map(float, row))
        _,_ = model.predict(row[:-1], row[-1])

print("============================ Prediction Phase - Completed  ===============================")


print("\n\n============================  Model Weights ===============================")
# Print weights
print(model.get_weights())


