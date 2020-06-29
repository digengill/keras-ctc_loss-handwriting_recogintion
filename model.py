from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise, Conv2D, MaxPooling2D, MaxPool2D, Reshape, Lambda, BatchNormalization, add,concatenate, GRU
from keras.optimizers import Adam
from keras.models import Model
import string
import pickle
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K

class my_Model:

	def __init__(self):
		self.ctcm = None
		self.predict_model = None		

	def ctc_lambda_func(self,args):
	    y_pred, labels, input_length, label_length = args
	    # the 2 is critical here since the first couple outputs of the RNN
	    # tend to be garbage:
	    #y_pred = y_pred[:, 2:, :]
	    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

	def create_model(self,input_shape,label_shape):
		# Using CTCModel
		"""
		required input_shape = (48,96,1)
		output = 53
		"""
		inputs = Input(shape=input_shape) 
		conv_1 = Conv2D(32, (3,3), activation = 'relu', padding='same')(inputs)
		pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2,1))(conv_1) 
		conv_2 = Conv2D(64, (3,3), activation = 'relu', padding='same')(pool_1)
		conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(conv_2)

		pool_2 = MaxPool2D(pool_size=(2, 2), strides=(3,2))(conv_2)
		conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
		pool_3 = MaxPool2D(pool_size=(2, 2), strides=3)(conv_2)
		conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_3)
		pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

		conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
		pool_5 = MaxPool2D(pool_size=(2, 1), strides=(2,1))(conv_5)
		batch_norm_5 = BatchNormalization()(pool_5)
		conv_6 = Conv2D(512, (2,2), activation = 'relu', padding='same')(batch_norm_5)
		batch_norm_6 = BatchNormalization()(conv_6)

		pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
		conv_7 = Conv2D(512, (2,2), activation = 'relu', padding='same')(pool_6)

		squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

		blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(squeezed)
		blstm_2 = Bidirectional(LSTM(256, return_sequences=True))(blstm_1)
		outputs = Dense(53, activation = 'softmax')(blstm_2)

		self.predict_model = Model(inputs, outputs)
		self.predict_model.summary()

		labels = Input(name='the_labels', shape=[label_shape], dtype='float32') # (None ,8)
		input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
		label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)
		# Keras doesn't currently support loss funcs with extra parameters
		# so CTC loss is implemented in a lambda layer
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length]) #(None, 1)

		self.ctcm = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

	def my_compile(self):
		self.ctcm.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam())

	def my_fit(self,x_train,y_train, x_train_len, y_train_len):
		history = self.ctcm.fit(x=[x_train,y_train, x_train_len, y_train_len], y=np.zeros(len(x_train)), epochs=5, verbose = 1)
		return history
	
	def my_load_weights(self, w_path):
		self.predict_model.load_weights(w_path)
		print("Weights loaded!")

	def my_predict(self,img, char_list):

		prediction = self.predict_model.predict([img])
		out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=True)[0][0])
		i = 0
		#print("out=",out)
		answer = ""
		answer_int = ""
		for x in out:
		    #print("predicted text = ", end = '')
		    for p in x:  
		        if int(p) != -1:
		        	answer = answer + char_list[int(p)]
		        	answer_int = answer_int + str(p) + " "
		            #print(char_list[int(p)], end = '')       
		    #print('\n')
		    i+=1
		print("predicted=",answer)
		#print("predicted_int=",answer_int)
		return out, answer