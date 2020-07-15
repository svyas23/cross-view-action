# data preprocessing parameters
raw_videos = '/groups/mshah/data/ntu-ard/videos'
# araw_videos = '/home/c2-2/yogesh/datasets/ntu-ard/videos'
#raw_videos = '/home/shruti/crcv/home/c2-2/yogesh/datasets/ntu-ard/videos'
rgb_data = '/groups/mshah/data/ntu-ard/frames-240x135'
# rgb_data = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135'
#rgb_data = '/home/shruti/crcv/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135'
num_classes = 60
dataset_name = 'nturgb+d_rgb'
cam_ids = [1, 2, 3]
video_list = 'data/video.list'
view_params = 'data/view.params'
train_list = 'data/train.list'
val_list = 'data/val.list'
train_list1 = 'data/video.list'
val_list1 = 'data/video.list'

# training parameters
batch_size=32
crop_size=112
num_clips=2
num_frames=6
view_dims=7
noise_dims=1
num_views = 2
num_channels=3
r_shape=(num_frames,28,28,256)
v_shape=(1,1,view_dims)
z_shape=(1,1,1)

shuffle=True

num_train_samples=16000
center_crop=True

lr = 1e-5
epochs=10000
out_dir='test'

model_weights1='test/2views_101-0.604319-0.851-1.h5'
model_weights='test/2views_303-0.735551-0.801-1.h5'
model_weights2='test/2views_063-0.702150-0.851-1.h5'
dump_path = 'dump'
enc_model1 ='sports1M_weights_tf.h5'
enc_model ='test/2views_303-0.735551-0.801-1.h5'
