import os
import pandas as pd
from tqdm import tqdm
import re
import cv2
import tensorflow as tf
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import string

class DataLoader:

    #path = "images//{}\{}//{}".format(pl[0],pl[0]+"-"+pl[1], row['filename'])
    
    def encode_to_labels(self, txt, char_list):
        dig_lst = []
        for i, char in enumerate(txt):
            try:
                dig_lst.append(char_list.index(char))
            except:
                print(char)
        return dig_lst

    def image_crop(self, img_path, dims):
        img = cv2.imread(img_path)
        img = img[dims[2]:dims[3],dims[0]:dims[1]]
        cv2.imwrite(img_path, img)

    def image_extract(self, img):
        ext_im = img
        x,y = ext_im.shape
        ext_im[ext_im < 140] = 0;
        ext_im[ext_im > 230] = 255;

        ext_im = cv2.resize(ext_im, (y,48))
              #image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
        diff = 96 - y
        if diff > 0:
            ext_im = cv2.copyMakeBorder(ext_im,0,0,0,diff, cv2.BORDER_CONSTANT, value=(255,255,255))
        else:
            ext_im = cv2.resize(ext_im, (96,48))

        #ext_im = cv2.bitwise_not(ext_im)


        ext_im = np.expand_dims(ext_im , axis = 2)
        return ext_im

    def data_generate(self,df,data_size, char_list):
        d_X = []
        d_y = []
        max_label_len = 0
        regex = re.compile('[ .,\'\"-@_!#$%^&*()<>?/\\|}{~:]') 

        for i in tqdm(df.index):
            pl = df['filename'][i].split("-")
            word = df['transcription'][i]
            if not regex.search(word):
                path = "image_data/handwriting_data/{}/{}/{}.png".format(pl[0],pl[0]+"-"+pl[1],df['filename'][i])
                #print(path)
                try:
                    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    im = self.image_extract(im)
                    #print(im.shape)
                    #print(word)
                    #cv2.imshow('img',im)
                    #cv2.waitKey(0)
                    d_X.append([im, 32])
                    d_y.append([self.encode_to_labels(word, char_list), len(word)])
                    #print(self.encode_to_labels(word, char_list))

                    if len(word) > max_label_len:
                        max_label_len = len(word)
                except:
                    print("Image not read")
            if i == data_size and data_size != 0:
                break
        return d_X, d_y, max_label_len

    def data_save(self,X,y):
        pickle_out = open("X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
