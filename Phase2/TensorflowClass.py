
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import os

from AbstractClass_ML_Model import ML_Model

class TFCLass_Regression(ML_Model):
    def __init__(self):
        self.train_data=None
        self.train_label=None
        self.weights=None
        self.output=None
        self.cost=None
        self.optimizer=None
        self.sess=None

        #For statistics... 
        # count is scalar.. All other stats are rows of length - #columns
        self.total_count=None
        self.min_Row=None
        self.max_Row=None
        self.minmax_diff=None
        self.total_val=None

        #For parameters...
        # More params added with *args, **kwargs
        self.N_features=None
        self.N_targets=None

    def set_params(self, num_features, num_target,learning_rate):
        self.N_features=num_features
        self.N_targets=num_target
        self.l_rate=learning_rate
        self.create_model(num_features, num_target,learning_rate)

    def set_stats(self, count_t, total_d, min_r, max_r, minmax_r):
        self.total_count=count_t
        self.min_Row=min_r
        self.max_Row=max_r
        self.minmax_diff=minmax_r
        self.total_val=total_d

    def create_model(self, num_features, num_target,learning_rate):
        self.train_data=tf.compat.v1.placeholder(tf.float32,name="train_data",shape=(1,num_features))
        self.train_label=tf.compat.v1.placeholder(tf.float32,name="train_label")    #,shape=(1,1))
        # weights=tf.zeros([num_features,1])
        #weights=tf.compat.v1.placeholder(tf.float32,shape=(num_features,-1),name="weights")
        self.weights=tf.compat.v1.get_variable("weights",(num_features,1))    #(tf.zeros([num_features,1], tf.float32), name="weights")
        self.output=tf.matmul(self.train_data,self.weights)
        # cost=tf.reduce_sum(tf.pow(output-train_label, tf.const(2.0,tf.float32)))
        self.cost=tf.reduce_sum(tf.abs(self.output - self.train_label))
        # self.cost=tf.losses.mean_squared_error(self.train_label,self.output)
        self.optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        # epochs=num_epochs
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def fit_train(self, train_x, train_y):
        _ = self.sess.run(self.optimizer, feed_dict={self.train_data:self.normalize_data(train_x), self.train_label:self.toTensorFloat32(train_y)})
        c = self.sess.run(self.cost,feed_dict={self.train_data:self.normalize_data(train_x), self.train_label:self.toTensorFloat32(train_y)})  #, self.weights:self.toTensorFloat32(w)})
        # print(c)
        w = self.sess.run(self.weights)
        # return w,c

    def get_weights(self):
        return self.sess.run(self.weights)
    
    def predict(self, test_x, test_y):
        self.test_data=tf.compat.v1.placeholder(tf.float32,name="test_data",shape=(1,self.N_features))
        op = tf.matmul(self.test_data, self.weights)
        pred=self.sess.run(op,feed_dict={self.test_data:self.normalize_data(test_x)})
        # print("Weight: ", self.sess.run(self.weights))``
        print("TrueOutput: ", test_y, "\t PredictedOutput: ",int(round(pred[0][0])))
        return test_y, pred


    def normalize_data(self, data_row):
        """
                x - min(x)
        z = ------------------
              max(x) - min(x)
        """
        a = [ [(data_row[i] - self.min_Row[i])/self.minmax_diff[i] for i in range(len(data_row))  ] ]
        return a

    ##Other local helper methods
    def toTensorFloat32(self, obj):
        return obj
        # return tf.convert_to_tensor(obj, tf.float32)

    def serialize(self):
        pass
    
    def deserialize(self):
        pass
    
    def ensemble(self):
        pass
        

