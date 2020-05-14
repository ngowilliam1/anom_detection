# Import system packages
import time as time
#Import data manipulation libraries
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import initializers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, BatchNormalization, LeakyReLU, Dense, Reshape, Flatten, Activation
from tensorflow.keras.layers import Dropout, multiply, GaussianNoise, MaxPooling2D, concatenate
import random
import tensorflow.keras.backend as kb


class GAN:
    def __init__(self, learning_rate = 0.00001, batch_size = 512, total_epochs = 10):
        self.generator = Sequential()
        self.discriminator = Sequential()

        # Hyper Paramaters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.evaluationAtEpochs = []
        self.optimizer = Adam(lr = self.learning_rate, beta_1 = 0.5)

    def save_discriminator_model(self,filename):
        self.discriminator.save(filename)

    def fit(self, x, x_valid=None, dictFit=None):
        start = time.time()
        self.inp_shape = x.shape[1]
        self.initGenerator(inp_shape=self.inp_shape)
        self.initDiscriminator(inp_shape=self.inp_shape)
        self.ganNetwork = self.get_gan_network(input_dim=self.inp_shape)

        # Calculating the number of batches based on the batch size
        self.batch_count = x.shape[0] // self.batch_size
        self.pbar = tqdm(total=self.total_epochs * self.batch_count)
        self.gan_loss = []
        self.discriminator_loss = []
        self.train(x, x_valid, dictFit)
        runtime = time.time()-start
        return runtime, self.evaluationAtEpochs

    def predict(self, image_batch, batch_size=128, verbose=0):
        return self.discriminator.predict(x=image_batch, batch_size=batch_size, verbose=verbose)

    def evaluate(self, x):
        nr_batches_test = np.ceil(x.shape[0] // self.batch_size).astype(np.int32)
        results = []
        for t in range(nr_batches_test +1):
            ran_from = t * self.batch_size
            ran_to = (t + 1) * self.batch_size
            image_batch = x[ran_from:ran_to]
            tmp_rslt = self.predict(image_batch=image_batch, batch_size=128, verbose=0)
            results = np.append(results, tmp_rslt)
        # Do (1 - results) because results is likelihood that x is "normal"
        return (1-results)

    def train(self, x, x_valid = None, dictFit=None):
        x_train = x
        for epoch in range(self.total_epochs):
            for index in range(self.batch_count):
                self.pbar.update(1)
                # Creating a random set of input noise and images
                noise = np.random.normal(0, 1, size=[self.batch_size, self.inp_shape])

                # Generate fake samples
                generated_images = self.generator.predict_on_batch(noise)

                # Obtain a batch of normal network packets
                image_batch = x_train[index * self.batch_size: (index + 1) * self.batch_size]

                X = np.vstack((generated_images, image_batch))
                y_dis = np.ones(2 * self.batch_size)
                y_dis[:self.batch_size] = 0

                # Train discriminator
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.uniform(0, 1, size=[self.batch_size, self.inp_shape])
                y_gen = np.ones(self.batch_size)
                self.discriminator.trainable = False
                g_loss = self.ganNetwork.train_on_batch(noise, y_gen)

                # Record the losses
                self.discriminator_loss.append(d_loss)
                self.gan_loss.append(g_loss)

            print("Epoch %d Batch %d/%d [D loss: %f] [G loss:%f]" % (epoch, index, self.batch_count, d_loss, g_loss))
            if (x_valid is not None) and ((epoch % 2) == 0):
                self.evaluationAtEpochs.append(self.evaluate(x_valid))
                if dictFit['save_model']:
                    model_filename = f'models/{dictFit["dataset"]}_GAN_perc_{dictFit["percentage"]}_epoch_{epoch}_{dictFit["timestr"]}.h5'
                    self.save_discriminator_model(model_filename)


    def get_gan_network(self,input_dim):
        self.discriminator.trainable = False
        gan_input = Input(shape=(input_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)

        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        return gan


    def initGenerator(self, inp_shape=None):
        self.generator.add(Dense(64, input_dim=inp_shape, kernel_initializer=initializers.glorot_normal(seed=42)))
        self.generator.add(Activation('tanh'))

        self.generator.add(Dense(128))
        self.generator.add(Activation('tanh'))

        self.generator.add(Dense(256))
        self.generator.add(Activation('tanh'))

        self.generator.add(Dense(256))
        self.generator.add(Activation('tanh'))

        self.generator.add(Dense(512))
        self.generator.add(Activation('tanh'))

        self.generator.add(Dense(inp_shape, activation='tanh'))
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def initDiscriminator(self,inp_shape=None):
        self.discriminator.add(Dense(256, input_dim=inp_shape, kernel_initializer=initializers.glorot_normal(seed=42)))
        self.discriminator.add(Activation('relu'))
        self.discriminator.add(Dropout(0.2))

        self.discriminator.add(Dense(128))
        self.discriminator.add(Activation('relu'))
        self.discriminator.add(Dropout(0.2))

        self.discriminator.add(Dense(128))
        self.discriminator.add(Activation('relu'))
        self.discriminator.add(Dropout(0.2))

        self.discriminator.add(Dense(128))
        self.discriminator.add(Activation('relu'))
        self.discriminator.add(Dropout(0.2))

        self.discriminator.add(Dense(128))
        self.discriminator.add(Activation('relu'))
        self.discriminator.add(Dropout(0.2))

        self.discriminator.add(Dense(1))
        self.discriminator.add(Activation('sigmoid'))

        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

