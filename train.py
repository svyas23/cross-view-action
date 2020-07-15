import tensorflow as tf
from keras import backend as K
tf.set_random_seed(1)
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, merge, Lambda
from keras.optimizers import Adam
from keras import backend as K
from data_generator import DataGenerator
from data_generator_1 import DataGenerator_1
import params as params
from generator import generator_model
from c3d_a import rep_model
from c3d_b import clas_model
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def train_model(weights=True):
    # Standardizing the input shape order
    K.set_image_dim_ordering('tf')

    # create the representation model
    input_shape1 = (params.num_views*params.num_clips, params.num_frames, params.crop_size, params.crop_size, params.num_channels)
    input_shape2 = (params.num_views*params.num_clips, params.num_frames, 1, 1, params.view_dims)
    
    model1 = rep_model(input_shape1, input_shape2)
    
    inputs1 = Input(input_shape1)
    inputs2 = Input(input_shape2)
    
    #clas = model1([inputs1, inputs2])
    #clas, r = model1([inputs1, inputs2])
    r = model1([inputs1, inputs2])
    
    clas = clas_model(r)

    v = Input(shape=params.v_shape)
    z = Input(shape=params.z_shape)
    
    g = generator_model(r, v, z)
    
    model = Model(inputs=[inputs1,inputs2,v,z], outputs=[clas, g])

    model.compile(loss=['categorical_crossentropy','mse'], optimizer=Adam(lr=params.lr), metrics=['accuracy'])
    #model1.compile(loss=['categorical_crossentropy'], optimizer=Adam(lr=params.lr), metrics=['accuracy'])
    
    if weights:
        model.load_weights(params.model_weights1, by_name=True)
    
    model.summary(line_length=120)
    """
    video_list = np.loadtxt(params.video_list, dtype=str)
    np.random.shuffle(video_list)

    train_list = video_list[:params.num_train_samples]
    validation_list = video_list[params.num_train_samples:]

    train_steps_per_epoch = len(train_list)/params.batch_size
    val_steps = len(validation_list)/params.batch_size
    """
    # train_list = np.loadtxt(params.train_list, dtype=str)
    # validation_list = np.loadtxt(params.val_list, dtype=str)
    train_list = np.loadtxt(params.train_list1, dtype=str)
    validation_list = np.loadtxt(params.val_list1, dtype=str)

    train_steps_per_epoch = len(train_list)/params.batch_size
    val_steps = len(validation_list)/params.batch_size

    training_generator = DataGenerator(train_list, 
                                batch_size=params.batch_size,
                                num_frames=params.num_frames, 
                                num_channels=params.num_channels,
                                num_views=params.num_views,
                                num_clips=params.num_clips,
                                num_classes=params.num_classes,
                                shuffle=params.shuffle,
                                crop_size=params.crop_size)

    validation_generator = DataGenerator_1(validation_list,
                                batch_size=params.batch_size,
                                num_frames=params.num_frames, 
                                num_channels=params.num_channels,
                                num_views=params.num_views,
                                num_clips=params.num_clips,
                                num_classes=params.num_classes,
                                shuffle=params.shuffle,
                                crop_size=params.crop_size)


    checkpoint = ModelCheckpoint(filepath=params.out_dir+"/2views_{epoch:03d}-{val_loss:.6f}-{val_prediction_acc:.3f}-1.h5", verbose=1, monitor='val_prediction_acc', save_best_only=True)

    model.fit_generator(generator=training_generator, 
                        validation_data=validation_generator,
                        steps_per_epoch=train_steps_per_epoch, 
                        # validation_steps=val_steps, 
                        validation_steps=200, 
                        epochs=params.epochs,
                        callbacks=[checkpoint],
                        max_queue_size=4,
                        use_multiprocessing=True,
                        initial_epoch=101,
                        workers=4)

    model.save(params.out_dir+'/model.h5')
    model.save_weights(params.out_dir+'/model_weights.h5')

if __name__=='__main__':
    train_model()

