from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv3D,Conv2D,ConvLSTM2D,MaxPooling2D,Reshape,Bidirectional,Input,MaxPooling3D,Flatten,Activation,TimeDistributed,Lambda,UpSampling3D,Concatenate
from keras.regularizers import l2
from keras.models import Model
import h5py
import params as params
import keras.backend as K

def clas_model(r, weights=False, summary=False):
    x = Conv2D(128, (3, 3), activation='tanh', padding='same', name='conv6a_2')(r)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='pool16')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv6a_3')(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(1024, activation='sigmoid', name='fc61')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='sigmoid', name='fc71')(x)
    
    x = Dense(params.num_classes, activation='softmax', name='prediction')(x)
    
    if weights:
        model.load_weights(params.enc_model, by_name=True)

    if summary:
        model.summary(line_length=120)

    return x

if __name__ == '__main__':

    c3d = rep_model(input_shape1 = (params.num_views*params.num_clips, params.num_frames, params.crop_size, params.crop_size, params.num_channels), input_shape2 = (params.num_views*params.num_clips, params.num_frames, 1, 1, params.view_dims), weights=True, summary=True)
    c3d.summary(line_length=120)

