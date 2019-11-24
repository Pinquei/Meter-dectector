from dataloader import DataLoader,load_test_data
from keras.layers import Input, BatchNormalization,MaxPooling2D, UpSampling2D, Conv2D, Activation,ZeroPadding2D
from keras.applications.vgg19 import VGG19
from keras.layers import Dense,Flatten,Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
import datetime
import numpy as np
from keras import layers
import csv
from keras.models import load_model
import scipy.misc
import random
from io import BytesIO


class ConvolutionalNeuralNetworks():
    def __init__(self):
        # Input shape
        self.img_rows = 100
        self.img_cols = 100
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

        # Build the network
        optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
        self.CNN_Network = self.build_CNN_Network()
        # print(self.CNN_Network.summary())
        self.CNN_Network.load_weights('./model/CNN_Network_on_epoch_1320.h5')
        self.CNN_Network.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # Input images
        Xtr = Input(shape=self.img_shape)

        # Output images
        Y = self.CNN_Network(Xtr)

    def build_CNN_Network(self):
        def conv2d(layer_input, filters, f_size=3, bn=True, dropout_rate=0):

            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if dropout_rate:
                d = Dropout(dropout_rate)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            return d

        def maxpooling2d(layer_input, f_size, stride=2):
            d = MaxPooling2D(pool_size=f_size, strides=stride, padding='same')(layer_input) #更改
            return d

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='linear')(layer_input)
                d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        d0 = Input(shape=self.img_shape)
        d1 = conv2d(d0, filters=32, f_size=3)
        d2 = maxpooling2d(d1, f_size=2, stride=2)
        d3 = conv2d(d2, filters=64, f_size=3)
        d4 = maxpooling2d(d3, f_size=2, stride=2)
        d5 = conv2d(d4, filters=64, f_size=3)
        d6 = maxpooling2d(d5, f_size=2, stride=2)
        d7 = Dropout(0.5)(d6)
        d8 = flatten(d7)
        d9 = dense(d8, f_size=1024, lastLayer=False)
        d10 = Dropout(0.5)(d9)
        d11 = dense(d10, f_size=101)

        return Model(d0,d11)

    def train(self, epochs, batch_size=1):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                #  Training
                loss = self.CNN_Network.train_on_batch(imgs_A, imgs_B)
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [Training clsLoss: %f, Training acc: %f] time: %s" % (
                    epoch + 1, epochs,
                    batch_i + 1, self.data_loader.n_batches,
                    loss[0],100*loss[1],
                    elapsed_time))
            # self.validation(epoch)
            if (epoch + 1) % 10 == 0:
                self.validation(epoch)
                self.CNN_Network.save_weights('./model/CNN_Network_on_epoch_%d.h5' % (epoch + 1321))


    def validation(self, epoch):
        ans = 0
        name, imgs_A, imgs_B_cls_map = self.data_loader.load_data(batch_size=100, is_testing=False)
        pred_labels=self.CNN_Network.predict(imgs_A)
        gt=np.argmax(imgs_B_cls_map,axis=1)
        prd=np.argmax(pred_labels,axis=1)
        print(gt)
        print(prd)
        print("Validation acc: " + str(int(accuracy_score(gt, prd) * 100)) + "%")

if __name__ == '__main__':
    # training model
    my_CNN_Model = ConvolutionalNeuralNetworks()
    my_CNN_Model.train(epochs=1000, batch_size=216)

    # testing
    # my_CNN = ConvolutionalNeuralNetworks()
    # my_CNN_Model = my_CNN.build_CNN_Network()
    # my_CNN_Model.load_weights('./model/CNN_Network_on_epoch_200.h5')
    # with open('output1.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     pred_labels = my_CNN_Model.predict(load_test_data(batch_size=1225))
    #     labels = np.argmax(pred_labels,axis=1)
    #     for i in labels:
    #             writer.writerow([i])
