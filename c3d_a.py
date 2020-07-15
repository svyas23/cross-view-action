from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv3D,Conv2D,ConvLSTM2D,MaxPooling2D,Reshape,Bidirectional,Input,MaxPooling3D,Flatten,Activation,TimeDistributed,Lambda,UpSampling3D,Concatenate
from keras.regularizers import l2
from keras.models import Model
import h5py
import params as params
import keras.backend as K

def rep_model(input_shape1, input_shape2, weights=False, summary=True):
    # input_shape = (16,112,112,3)
    inputs1 = Input(input_shape1)
    inputs2 = Input(input_shape2)
    # 1st layer group
    x = TimeDistributed(Conv3D(64, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name ='c3d1')(inputs1)
    x = TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'), name='c3d2')(x)
    # 2nd layer group
    x = TimeDistributed(Conv3D(128, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d3')(x)
    x = TimeDistributed(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same', name='c3d4'))(x)
    # Concat input layer
    v = TimeDistributed(UpSampling3D(size = (1,28,28), data_format=None))(inputs2)
    x = Concatenate(axis=5)([v,x])

    # 3rd layer group
    # x = TimeDistributed(Conv3D(256, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d5')(x)
    x = TimeDistributed(Conv3D(256, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d6')(x)
    x = TimeDistributed(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding='same', name='pool3'))(x)
    # 4th layer group
    # x = TimeDistributed(Conv3D(512, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d7')(x)
    x = TimeDistributed(Conv3D(512, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d8')(x)
    x = TimeDistributed(MaxPooling3D(pool_size=(3, 1, 1), strides=(3, 1, 1), padding='same', name='pool4'))(x)
    # 5th layer group
    # x = TimeDistributed(Conv3D(512, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d9')(x)
    x = TimeDistributed(Conv3D(256, (3, 3, 3), activation='relu', padding='same', dilation_rate=(1, 1, 1)), name='c3d10')(x)
    # x = TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', name='pool5'))(x)
    x = TimeDistributed(Reshape((28, 28, 256)))(x)
    x = Bidirectional(ConvLSTM2D(128, (3,3), activation='relu', padding='same', return_sequences=True), name='blending')(x)
    y = ConvLSTM2D(256, (2,2), activation='relu', name='blending1', padding='same', return_sequences=False)(x)
    model = Model([inputs1, inputs2], [y])

    if weights:
        model.load_weights(params.enc_model, by_name=True)

    if summary:
        model.summary(line_length=120)

    return model

if __name__ == '__main__':

    c3d = rep_model(input_shape1 = (params.num_views*params.num_clips, params.num_frames, params.crop_size, params.crop_size, params.num_channels), input_shape2 = (params.num_views*params.num_clips, params.num_frames, 1, 1, params.view_dims), weights=True, summary=True)
    c3d.summary(line_length=120)

