from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
#import cv2



class dataProcess(object):

	def __init__(self, out_rows, out_cols, txt_path = "./CamVid", npy_path = "./npydata", img_type = "png", num_class = 12):

		"""
		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.img_type = img_type
		self.npy_path = npy_path
		self.txt_path = txt_path
		self.num_class = num_class

	def binarylab(self, labels):
		x = np.zeros([360,480,self.num_class])
		for i in range(360):
			for j in range(480):
				x[i,j,int(labels[i][j])]=1
		return x

	def create_train_data(self):
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		with open(self.txt_path + '/train.txt') as f:
			txt = f.readlines()
			txt = [line.split(' ') for line in txt]
		train_data = np.ndarray((len(txt),self.out_rows,self.out_cols,3), dtype=np.uint8)
		train_label = np.ndarray((len(txt),self.out_rows,self.out_cols,self.num_class), dtype=np.uint8)
		for i in range(len(txt)):
			train_data[i] = img_to_array(load_img(os.getcwd() + txt[i][0][7:]))
			train_label[i] = self.binarylab(img_to_array(load_img(os.getcwd() + txt[i][1][7:][:-1]))[:,:,0])
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, len(txt)))
			i += 1
		print('loading done')
		train_label = np.reshape(train_label, (len(txt),self.out_cols * self.out_rows,self.num_class))
		np.save(self.npy_path + '/imgs_train.npy', train_data)
		np.save(self.npy_path + '/imgs_mask_train.npy', train_label)
		print('Saving to .npy files done.')

	def create_test_data(self):
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		with open(self.txt_path + '/test.txt') as f:
			txt = f.readlines()
			txt = [line.split(' ') for line in txt]
		test_data = np.ndarray((len(txt),self.out_rows,self.out_cols,3), dtype=np.uint8)
		test_label = np.ndarray((len(txt),self.out_rows,self.out_cols,self.num_class), dtype=np.uint8)
		for i in range(len(txt)):
			test_data[i] = img_to_array(load_img(os.getcwd() + txt[i][0][7:]))
			test_label[i] = self.binarylab(img_to_array(load_img(os.getcwd() + txt[i][1][7:][:-1]))[:,:,0])
			if i % 10 == 0:
				print('Done: {0}/{1} images'.format(i, len(txt)))
			i += 1
		print('loading done')
		test_label = np.reshape(test_label, (len(txt),self.out_cols * self.out_rows,self.num_class))
		np.save(self.npy_path + '/imgs_test.npy', test_data)
		np.save(self.npy_path + '/imgs_mask_test.npy', test_label)
		print('Saving to .npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean	
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean	
		return imgs_test

if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(360,480)
	mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#print imgs_train.shape,imgs_mask_train.shape