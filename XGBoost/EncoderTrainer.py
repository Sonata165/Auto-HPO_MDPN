import keras
from keras.layers import *
from keras.models import *
from keras.utils import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import *


class AutoEncoder:
    def __init__(self, input_shape, first_output_shape, second_output_shape, encoder1=None, encoder2=None,
                 auto_encoder1=None, auto_encoder2=None):
        '''
        Two-layer auto encoder, train seperately
        input_size->first_output_shape->second_output_shape
        The data need to be normalized to the range (-1, 1)
        :param input_shape: (x1,)
        :param first_output_shape: (x2,)
        :param second_output_shape: (x3,)
        '''
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.auto_encoder1 = auto_encoder1
        self.auto_encoder2 = auto_encoder2
        self.input_shape = input_shape
        self.first_output_shape = first_output_shape
        self.second_output_shape = second_output_shape
        self.build_encoder_decoder()

    def set_model(self):
        '''
        Read pretrained model from file
        '''
        self.encoder1 = load_model('encoder1.h5')
        self.auto_encoder1 = load_model('auto_encoder1.h5')
        self.encoder2 = load_model('encoder2.h5')
        self.auto_encoder2 = load_model('auto_encoder2.h5')

    def build_encoder_decoder(self):
        '''
        Construct the models of auto encoders
        :return: The first encoder, the second encoder,
            the first encoder + decoder (for training), the second encoder + decoder (for training)
        '''
        # The first encoder
        input_data = Input(self.input_shape)
        encoder = Dense(self.first_output_shape[0])(input_data)
        """
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        """
        encoder = Activation('tanh')(encoder)
        self.encoder1 = Model(inputs=[input_data], outputs=[encoder])
        decoder = Dense(self.input_shape[0])(encoder)
        """
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        """
        decoder = Activation('tanh')(decoder)
        self.auto_encoder1 = Model(inputs=[input_data], outputs=[decoder])
        self.auto_encoder1.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mse)
        # The seconde Encoder
        input_data = Input(self.first_output_shape)
        encoder = Dense(self.second_output_shape[0])(input_data)
        """
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        """
        encoder = Activation('tanh')(encoder)
        self.encoder2 = Model(inputs=[input_data], outputs=[encoder])
        decoder = Dense(self.first_output_shape[0])(encoder)
        """
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        """
        decoder = Activation('tanh')(decoder)
        self.auto_encoder2 = Model(inputs=[input_data], outputs=[decoder])
        self.auto_encoder2.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.mse)
        return self.encoder1, self.encoder2, self.auto_encoder1, self.auto_encoder2

    def predict(self, input_data):
        '''
        Encode data
        :param input_data: data to be encoded
        :return: Encoding results
        '''
        return self.encoder2.predict(self.encoder1.predict(input_data))

    def train(self, x, epoch, batch_size):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=50, min_lr=0.0001)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True,
                                                   min_delta=0.01)
        self.auto_encoder1.fit(x, x, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=0,
                               callbacks=[reduce_lr, early_stop])
        x1 = self.encoder1.predict(x)
        self.auto_encoder2.fit(x1, x1, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=0,
                               callbacks=[reduce_lr, early_stop])

    def get_feature(self):
        '''
        Return the weights of the encoder as meta-feature of the datasets
        :return: encoder1 weight,encoder2 weight
                type: list(numpy array)
                eg. the parameter of encoder1 is
                [(1000,200,1),(201,50,1)]
        '''
        res = [self.encoder1.get_weights(), self.encoder2.get_weights()]
        return [np.expand_dims(np.concatenate((res[0][0], res[0][1].reshape(1, -1)), axis=0), axis=-1),
                np.expand_dims(np.concatenate((res[1][0], res[1][1].reshape(1, -1)), axis=0), axis=-1)]
