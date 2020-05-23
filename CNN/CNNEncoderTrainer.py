import keras
from keras.layers import *
from keras.models import *
import keras.backend as K
from keras.utils import multi_gpu_model


class AutoEncoder:
    def __init__(self, input_shape, label_shape, auto_encoder=None, encoder=None):
        '''
        Two-layer auto encoder, train seperately
        input_size->first_output_shape->second_output_shape
        Data need to be normalized to range (-1, 1)
        :param input_shape: (width,height,channel)
        :param feature_shape: (width,height,channel)
        :param label_shape: (label_size,)
        '''
        # print(input_shape)
        self.auto_encoder = auto_encoder
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.encoder = encoder
        self.build_encoder_decoder()

    def set_model(self):
        '''
        Read pretrained model from file
        '''
        # self.encoder = load_model('encoder.h5')
        self.auto_encoder = load_model('auto_encoder.h5')

    def build_encoder_decoder(self):
        '''
        Construct auto encoder model
        :return: encoder encoder + decoder (used for training)
        '''
        # encoder 1
        input_data = Input(self.input_shape)
        input_label = Input(self.label_shape)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='encoder1')(input_data)
        x = MaxPooling2D((2, 2), padding='same', name='encoder2')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder3')(x)
        x = MaxPooling2D((2, 2), padding='same', name='encoder4')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same', name='encoder5')(x)
        x = MaxPooling2D((2, 2), name='encoder6')(x)
        # Save the shape before flattening
        x_shape = K.int_shape(x)
        # Flatten
        x = Flatten()(x)
        x_size = K.int_shape(x)[1]
        label = Dense(self.label_shape[0], name='encoder7')(input_label)
        # Join X and label
        x_with_label = Concatenate()([x, label])
        # Unified encoding
        encoded = Dense(x_size, activation='relu', name='encoder8')(x_with_label)

        # Unified decoding
        x_with_label = Dense(x_size, activation='relu')(encoded)
        # Extract label from results
        label = Dense(self.label_shape[0])(x_with_label)
        # Reconstruct high dimention x from results (undo flattening)
        x = Reshape((x_shape[1], x_shape[2], x_shape[3]))(x_with_label)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same', name='decoded')(x)  # 32x32

        self.auto_encoder = Model(inputs=[input_data, input_label], outputs=[decoded, label])

        ''' parallel '''
        # n_GPUs = 4
        # model = multi_gpu_model(model, n_GPUs)
        # n_GPUs = 4
        # self.auto_encoder = multi_gpu_model(self.auto_encoder, n_GPUs)

        self.auto_encoder.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.mse)
        self.auto_encoder.summary()
        # visualize_activation(self.auto_encoder,utils.find_layer_idx(self.auto_encoder,'decoded'),filter_indices=0,input_range=(0,1))
        return self.auto_encoder

    def predict(self, input_data):
        '''
        Encode the data
        :param input_data: data to be encoded
        :return: encoding results
        '''
        return self.encoder.predict(input_data)

    def train(self, x, epoch, batch_size):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # check_point = keras.callbacks.ModelCheckpoint(filepath='./auto_encoder_checkPoint.h5', monitor='val_loss',
        #                                               save_best_only=True, verbose=1)
        # tensor_board = keras.callbacks.TensorBoard(log_dir='./auto_encoder_tensor_board_logs', write_grads=True,
        #                                            write_graph=True,
        #                                            write_images=True)
        self.auto_encoder.fit(x, x, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=2,
                              callbacks=[reduce_lr, early_stop])
        # self.auto_encoder.save('to_encoder.h5')
        self.encoder = Model(inputs=self.auto_encoder.input, outputs=self.auto_encoder.get_layer('encoder8').output)
        # self.encoder.save('encoder.h5')

    def get_feature(self):
        '''
        Return the weights of the encoder as the meta-feature of the dataset
        :return: encoder1 weights, encoder2
                type: list(numpy array)
                eg. encoder1's parameter is
                [the results of 'get_weights' of encoder's each layer]
        '''
        res = []
        st = 'encoder'
        # generate the name of encoder's each layer, then extract weights
        for i in range(1, 9):
            res.append(self.auto_encoder.get_layer(st + str(i)).get_weights())
        return res
