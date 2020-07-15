import numpy as np
import keras
import pickle
import random
import params
import cv2
import os

import params_test as params
cam = 1

def decrypt_vid_name(vid):

    scene = int(vid[1:4])
    pid = int(vid[5:8])
    rid = int(vid[9:12])
    action = int(vid[13:16])

    return scene, pid, rid, action

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, num_clips=3, num_frames=8, 
                crop_size=112, num_channels=3, num_views=3,
                num_classes=60, shuffle=False):
        'Initialization'
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.num_views = num_views
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.view_dims = params.view_dims
        self.noise_dims = params.noise_dims
        self.on_epoch_end()

        self.view_params = self.__load_view_params()

    def __load_view_params(self):
        view_params = np.loadtxt(params.view_params)

        # normalize the distances
        view_params /= view_params.max(axis=0)

        return view_params

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch. Index is stored by the parent class to indicate the batch number.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        clips, view, target, t_class, t_view, t_noise = self.__data_generation(list_IDs_temp)

        return [clips, view, t_view, t_noise], [t_class, target]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_viewing_angle(self, rid, cam):
        vpt = 0.
        pi = 22/7.
        # rid-1 implies face towards cam3; rid-2 implies face towards cam2; cam1 is the center camera
        if rid == 1:
            if cam == 1:
                vpt = pi/4.
            elif cam == 2:
                vpt = pi/2.
            elif cam == 3:
                vpt = 0.00
        elif rid == 2:
            if cam == 1:
                vpt = -pi/4.
            elif cam == 2:
                vpt = 0.00
            elif cam == 3:
                vpt = -pi/2.
    
        return vpt

    def get_scene_parameters(self, scene):
        return self.view_params[scene][1], self.view_params[scene][2]

    def get_view(self, vid, cam, x_pos, y_pos, fh, fw, _id, fcount):
        v_name = os.path.split(vid)[1]
        scene, pid, rid, action = decrypt_vid_name(v_name)

        vpt = self.get_viewing_angle(rid, cam)

        ele, dis = self.get_scene_parameters(scene)
        
        pan = 1.*x_pos/(fh - self.crop_size)
        van = 1.*y_pos/(fw - self.crop_size)

        pos = 1.*_id/fcount
        
        if rid ==1:
            rid1=-0.5
        elif rid ==2:
            rid1=0.5
            
        return np.array([vpt, rid1, ele, dis, pan, van, pos])

    def get_target(self, ID):
        
        global cam
        
        vid = ID[0]
        target = np.empty((self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        # random_view = np.random.randint(1,4)
        random_view=1
        fcount = int(ID[random_view])
        skip_rate = 3
        # r_id = np.random.randint(0, fcount-(self.num_frames+1)*skip_rate)
        r_id = np.random.randint(30, 35)
        if fcount<53:
            r_id = np.random.randint(15, 20)
            
        # r_id =6
        f_path = os.path.join(params.rgb_data, vid, str(random_view), '{:03d}.jpg'.format(r_id))
        img = cv2.imread(f_path)
        height, width, channels = img.shape

        crop_pos_x = np.random.randint(0, height-self.crop_size)
        crop_pos_y = np.random.randint(0, width-self.crop_size)
        if params.center_crop:
            # if we need to crop only from the center of the frame
            crop_pos_y = np.random.randint(50, width-self.crop_size-50)
            
        for l in range(params.num_frames):
            target[l,] = img[crop_pos_x:crop_pos_x+self.crop_size, crop_pos_y:crop_pos_y+self.crop_size]
            f_path = os.path.join(params.rgb_data, vid, str(random_view), '{:03d}.jpg'.format(r_id+(l+1)*skip_rate))
            img = cv2.imread(f_path)

        t_class = int(os.path.split(vid)[0])
        
        # keras.utils.to_categorical(t_class, num_classes=params.num_classes, dtype='int32')
        t_class = keras.utils.to_categorical(t_class-1, num_classes=params.num_classes)
        # print t_class.shape
        t_view = self.get_view(vid, random_view, crop_pos_x, crop_pos_y, height, width, r_id, fcount)
        t_noise = np.random.rand()
        print(vid, 'r_id=', r_id, 'target')
        return (target-128.)/128., t_class, t_view, t_noise

    def get_frames(self, ID):
        
        vid = ID[0]
        global cam
        clips = np.empty((self.num_views*self.num_clips, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        view = np.empty((self.num_views*self.num_clips, self.num_frames, 1, 1, self.view_dims))

        # iterate through all views and collect frames
        cnt = 0
        skip_rate = 3
        cam_ids = params.cam_ids
        # np.random.shuffle(cam_ids)
        cam = cam_ids[0]
        for cam in cam_ids[1:3]:
            v_path = os.path.join(vid, str(cam))
            fcount = int(ID[cam])
            
            # select random frames
            # ids = np.random.randint(0, fcount-(self.num_frames+1)*skip_rate, self.num_clips)
            ids = np.random.randint(15, 20, self.num_clips)
            if fcount<53:
                ids = np.random.randint(5, 10, self.num_clips)
            # ids = [6,6]
            # collect random frames from this view
            for _id in ids:
                f_path = os.path.join(params.rgb_data, vid, str(cam), '{:03d}.jpg'.format(_id))
                img = cv2.imread(f_path)
                height, width, channels = img.shape

                crop_pos_x = np.random.randint(0, height-self.crop_size)
                crop_pos_y = np.random.randint(0, width-self.crop_size)
                # view[cnt, 0, 0, 0, ] = self.get_view(vid, cam, crop_pos_x, crop_pos_y, height, width, _id, fcount)
                if params.center_crop:
                    # if we need to crop only from the center of the frame
                    crop_pos_y = np.random.randint(50, width-self.crop_size-50)

                for j in range(params.num_frames):
                    
                    img_sample = img[crop_pos_x:crop_pos_x+self.crop_size, crop_pos_y:crop_pos_y+self.crop_size]    
                    clips[cnt, j, ] = (img_sample-128.)/128.
                    f_path = os.path.join(params.rgb_data, vid, str(cam), '{:03d}.jpg'.format(_id+(j+1)*skip_rate))
                    img = cv2.imread(f_path)
                    
                    view[cnt, j, 0, 0, ] = self.get_view(vid, cam, crop_pos_x, crop_pos_y, height, width, _id, fcount)
                
                cnt += 1
            print(ids, v_path)

        return clips, view

    def _get_sample(self, ID):

        # get the input clips
        clips, view = self.get_frames(ID)
            
        # get the target frames
        target, t_class, t_view, t_noise = self.get_target(ID)
        
        return clips, view, target, t_class, t_view, t_noise

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        clips = np.empty((self.batch_size, self.num_views*self.num_clips, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        view = np.empty((self.batch_size, self.num_views*self.num_clips, self.num_frames, 1, 1, self.view_dims))
        # sq1 = np.empty((self.batch_size, self.num_views*self.num_clips, self.num_frames, 1, 1, 1))
        target = np.empty((self.batch_size, self.num_frames, self.crop_size, self.crop_size, self.num_channels))
        t_class = np.empty((self.batch_size, params.num_classes))
        t_view = np.empty((self.batch_size, 1, 1, 1, self.view_dims))
        t_noise = np.empty((self.batch_size, 1, 1, 1, self.noise_dims))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            clips[i,], view[i,], target[i,], t_class[i,], t_view[i,0,0,], t_noise[i, 0, 0, ] = self._get_sample(ID)
            

        return clips, view, target, t_class, t_view, t_noise

