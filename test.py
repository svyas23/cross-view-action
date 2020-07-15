import tensorflow as tf
from keras import backend as K
tf.set_random_seed(1)
from keras.callbacks import TensorBoard, ModelCheckpoint

import os
#import matplotlib
# matplotlib.use('qt5agg')
#import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, merge, Lambda
from keras.optimizers import Adam
from keras import backend as K
from data_generator_test_single import DataGenerator
import params_test as params
from c3d import rep_model
from generator import generator_model
import numpy as np
from sklearn.metrics import accuracy_score
"""
if not os.path.exists(params.dump_path):
    os.makedirs(params.dump_path)

def save_frame(frame, _id):
    plt.imshow(frame)
    spath = params.dump_path + '/{:05d}.jpg'.format(_id)
    plt.savefig(spath)
    plt.close()

def save_predictions(preds):
    for i, p in enumerate(preds):
        save_frame(p, i)
"""

def save_frames(frame, vido):
    frame = (frame*128+128).astype(int)
    for j in range(frame.shape[0]):
        plt.imshow(frame[j][:,:,::-1])
        if not os.path.exists('./dump1/test_single/epoch63/1epoch63/'+ vido):
            os.makedirs('./dump1/test_single/epoch63/1epoch63/'+ vido)
        name = os.path.join('dump1/test_single/epoch63/1epoch63', vido, '{:03d}.png'.format(j))
        plt.savefig(name)
    # print name
    plt.close()

def test_model(weights=True):
    # Standardizing the input shape order
    K.set_image_dim_ordering('tf')

    # create the representation model
    input_shape1 = (params.num_views*params.num_clips, params.num_frames, params.crop_size, params.crop_size, params.num_channels)
    input_shape2 = (params.num_views*params.num_clips, params.num_frames, 1, 1, params.view_dims)
    
    model1 = rep_model(input_shape1, input_shape2, summary=False)
    
    inputs1 = Input(input_shape1)
    inputs2 = Input(input_shape2)
    
    clas, r = model1([inputs1, inputs2])

    v = Input(shape=params.v_shape)
    z = Input(shape=params.z_shape)
    
    g = generator_model(r, v, z)
    
    model = Model(inputs=[inputs1,inputs2,v,z], outputs=[clas, g])

    model.compile(loss=['categorical_crossentropy','mse'], optimizer=Adam(lr=params.lr), metrics=['accuracy'])
    # model.summary(line_length=120)
    
    if weights:
        model.load_weights(params.model_weights2, by_name=True)
    
    
    validation_list = np.loadtxt(params.val_list, dtype=str)
    #validation_list = np.loadtxt(params.val_list1, dtype=str)[:5000]

    #batch_size = int(64/num_clips)
    val_steps = int(len(validation_list)/params.batch_size)
    
    fp = open('./dump/score30.txt', 'w')

    validation_generator = DataGenerator(validation_list,
                            batch_size=params.batch_size,
                            num_frames=params.num_frames, 
                            num_channels=params.num_channels,
                            num_views=params.num_views,
                            num_clips=params.num_clips,
                            num_classes=params.num_classes,
                            shuffle=False,
                            crop_size=params.crop_size)
   
    predictions = model.predict_generator(generator=validation_generator, 
                    steps=val_steps, 
                    # steps=3, 
                    max_queue_size=6,
                    use_multiprocessing=False,
                    workers=1)
                    
    # print predictions[0]    
    pred = np.argmax(predictions[0], axis = 1)

    np.savetxt(fp, pred, fmt='%d') # 20 clips in name implies 40 clips from test view
    #np.savetxt(fp, pred, fmt='%d') # 20 clips in name implies 40 clips from test view
   
    gt = [int(os.path.split(x)[0])-1 for x in validation_list[:,0]]
   
    #print(num_clips)
    #print(accuracy_score(gt[:], pred))
    print(accuracy_score(gt[:params.batch_size*val_steps], pred))
    
    fp.close()
         
    # evaluation = model.evaluate_generator(generator=validation_generator, steps=3, max_queue_size=8, workers=1, use_multiprocessing=False)
    # np.savetxt('./dump1/test_pair/2-3epoch63/scores.txt', evaluation)  
    """  
    metrics_scores = open('./dump1/test_pair/2-3epoch63/scores.txt', 'w')
    dic0 = str(evaluation)
    metrics_scores.write(dic0+'\n') 
    metrics_scores.close()
    """
    # loss, model_1_loss, conv4_loss, model_1_acc, conv4-acc 
    # evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

    # save predictions to dump_path
    # save_predictions(predictions)
    """
    for i in range(predictions[1].shape[0]):
        save_frames(predictions[1][i], validation_list[i][0])
    """
if __name__=='__main__':
    #i = int(os.sys.argv[1])
    #print(i, 'starting new set...')
    test_model()

