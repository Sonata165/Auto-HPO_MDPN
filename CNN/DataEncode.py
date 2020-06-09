import keras.backend as bk
import keras
import keras.backend as K
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Concatenate, Reshape, UpSampling2D
from keras.models import *

from CNN.ReadDataset import *
from CNN.Constants import *

flag = 0


def main():
    compute_feature()


def compute_feature():
    dataset_path = 'data/subset/'
    result_path = 'data/feature/'
    files = os.listdir(dataset_path)
    for file in files:
        print(file)
        filename = file.split('.')[0]
        dataset = read_dataset(dataset_path + file, padding=True)

        # Else, do Encoding
        feature = encode(dataset)

        # save as pkl file
        with open(result_path + filename + '.pkl', 'wb') as f:
            pk.dump(feature, f)
        print(file + " finish!")


def encode(dataset):
    '''
    Encode dataset
    :param dataset: format is [x_train, y_train, x_test, y_test]
    '''
    x_train, y_train, x_test, y_test = dataset
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    print('x shape:', x[0].shape)
    encoder = AutoEncoder(input_shape=x[0].shape, label_shape=(10,))
    encoder.train([x, y], epoch=ENCODER_EPOCHS, batch_size=ENCODER_BATCH_SIZE)
    ret = encoder.get_feature()
    bk.clear_session()
    return ret


def encode_xy(dataset, epochs=ENCODER_EPOCHS, batchsize=ENCODER_BATCH_SIZE):
    x, y = dataset
    encoder = AutoEncoder(input_shape=x[0].shape, label_shape=(10,))
    encoder.train([x, y], epoch=epochs, batch_size=batchsize)
    ret = encoder.get_feature()
    bk.clear_session()
    return ret


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
        self.auto_encoder = load_model('model/auto_encoder.h5')

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

        self.auto_encoder.compile(optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.mse)
        # self.auto_encoder.summary()
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
        self.auto_encoder.fit(x, x, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=2,
                              callbacks=[reduce_lr, early_stop])
        self.encoder = Model(inputs=self.auto_encoder.input, outputs=self.auto_encoder.get_layer('encoder8').output)

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
        for i in range(1, 9):
            res.append(self.auto_encoder.get_layer(st + str(i)).get_weights())
        return res


if __name__ == "__main__":
    main()
