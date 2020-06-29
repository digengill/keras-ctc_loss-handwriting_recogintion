
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from data_loader import DataLoader
from model import my_Model
from app import App
import pandas as pd
import string



def load_save(max_label_len):
	pickle_in = open("X.pickle","rb")
	X = pickle.load(pickle_in)
	pickle_in = open("y.pickle","rb")
	y = pickle.load(pickle_in)  # !! Change sparse tensor to another character encoding

	x_len = len(X)
	x_train = []
	y_train = []
	x_train_len = []
	y_train_len = []
	for i in range(x_len):
		x_train.append(X[i][0])
		x_train_len.append(X[i][1])
		y_train.append(y[i][0])
		y_train_len.append(y[i][1])

		#print(x_train_len)
	x_train = np.array(x_train).reshape(-1, 48, 96, 1)
	x_train = x_train / 255.0
	y_train = pad_sequences(y_train, maxlen=max_label_len,padding='post')#, value=52)
	y_train = np.array(y_train)
	x_train_len = np.array(x_train_len)
	y_train_len = np.array(y_train_len)

	print("y shape=",y_train.shape)
	print("x shape=",x_train.shape)
	print("x_len_shape=",x_train_len.shape)
	print("y_len_shape=",y_train_len.shape)
	
	return x_train, y_train, x_train_len, y_train_len




def main():
	#save = False
	#dsize = 0
	#data = DataLoader()

	#df = pd.read_csv("words_csv.csv")
	#char_list = string.ascii_letters
	#X, y, max_label_len = data.data_generate(df, dsize, char_list)
	#print("max_label_len=",max_label_len)
	#if save:
	#	data.data_save(X,y)

	#x_train, y_train, x_train_len, y_train_len = load_save(max_label_len)

	my_model = my_Model()
	my_model.create_model(input_shape=(48,96,1),label_shape=17)#max_label_len)
	my_model.my_compile()

	#history = my_model.my_fit(x_train, y_train, x_train_len, y_train_len)
	my_model.my_load_weights('weights/7-conv-2-blstm-53-out/10-epochs-15-validation/ctcm3.hdf5')

	app = App(model=my_model, check_accuracy=False)

if __name__ == '__main__':
	main()