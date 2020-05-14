from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time

class AutoEncoderCollection:
    def __init__(self, layers, latent_dim, learning_rate = 0.00001, batch_size = 512, total_epochs = 10):
        self.layers = layers
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.autoEncoders = {}
        self.learning_rate = learning_rate

    def get(self, layer, latent_dim):
        return self.autoEncoders[(layer,latent_dim)]

    def addAE(self, layer, latent_dim):
        self.autoEncoders[(layer,latent_dim)] = AutoEncoder(latent_dim = latent_dim,
                                                            layer = layer,
                                                            learning_rate = self.learning_rate,
                                                            batch_size = self.batch_size,
                                                            total_epochs = self.total_epochs
                                                            )

    def fit(self, preprocessed_data):
        start = time.time()
        for layer in self.layers:
            for latent in self.latent_dim:
                print('Latent Dimension Nodes:', latent)
                self.addAE(layer = layer, latent_dim = latent)
                runtime_ae = self.autoEncoders[(layer,latent)].fit(preprocessed_data)
                print(f"AE layer:{layer}, latent:{latent}, Training time took: {runtime_ae}")
        return (time.time()-start)


class AutoEncoder:
    def __init__(self, latent_dim, layer, learning_rate, batch_size, total_epochs):
        self.latent_dim = latent_dim
        self.layer = layer
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.enc_layer_ids = ['encoder_'+str(i) for i in range(1,self.layer+1)]
        self.dec_layer_ids = ['decoder_'+str(i) for i in range(1,self.layer+1)]
        self.start_layer = (2+self.layer)*16
        self.layer_sizes = [int(self.start_layer/i) for i in range(1,self.layer+1)]
        self.learning_rate = learning_rate
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.runtime = 0

    def fit(self, x):
        start = time.time()
        x_train, x_valid = train_test_split(x, test_size=0.2, random_state=42)
        input_dim = x_train.shape[1]
        input_data = Input(shape=(input_dim,), name='encoder_input')

        for idx, enc_layer_id in enumerate(self.enc_layer_ids):
            print(enc_layer_id)
            if idx == 0:
                self.encoder = Dense(self.layer_sizes[idx], activation='tanh', name=enc_layer_id)(input_data)
                self.encoder = Dropout(.1)(self.encoder)
            else:
                self.encoder = Dense(self.layer_sizes[idx], activation='tanh', name=enc_layer_id)(self.encoder)
                self.encoder = Dropout(.1)(self.encoder)
        # bottleneck layer
        self.latent_encoding = Dense(self.latent_dim, activation='linear', name='latent_encoding')(self.encoder)
        self.encoder_model = Model(input_data, self.latent_encoding)
        for idx, dec_layer_id in enumerate(self.dec_layer_ids):
            print(self.layer_sizes[::-1][idx])
            if idx==0:
                self.decoder = Dense(self.layer_sizes[::-1][idx], activation='tanh', name=dec_layer_id)(self.latent_encoding)
                self.decoder = Dropout(.1)(self.decoder)
            else:
                self.decoder = Dense(self.layer_sizes[::-1][idx], activation='tanh', name=dec_layer_id)(self.decoder)
                self.decoder = Dropout(.1)(self.decoder)

        self.reconstructed_data = Dense(input_dim, activation='linear', name='reconstructed_data')(self.decoder)

        self.autoencoder_model = Model(input_data, self.reconstructed_data)
        


        self.autoencoder_model.compile(optimizer=self.optimizer, loss='mse')
        callback = EarlyStopping(monitor='val_loss', patience=3)
        # NOTE: validation_data is only used to evaluate the loss at each epoch, it is NOT USED for training
        self.train_history = self.autoencoder_model.fit(x_train, x_train,
                                                    shuffle=True,
                                                    callbacks=[callback],
                                                    epochs=self.total_epochs,
                                                    batch_size=self.batch_size,
                                                    validation_data =(x_valid, x_valid))
      
        return time.time() - start
    
    def predict(self,x):
        return self.autoencoder_model.predict(x)
    
    def evaluate(self, x):
        x_test_recon = self.predict(x)
        reconstruction_scores = np.mean((x - x_test_recon) ** 2, axis=1)
        return reconstruction_scores