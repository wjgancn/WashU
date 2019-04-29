from dataset import DataLoader
import tensorflow.keras as keras
import tensorflow as tf
from method import generator, discriminator, CallBackFun
import os
from tools import copy_code

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
experiment_path = '/export/project/SAVE_PATH/'
train_folder = 'apr28-photo2ce-128-par1'
copy_code(src='/export/project/CODE_PATH/', dst=experiment_path + train_folder + '/')

train_data = DataLoader(path_x='../cezanne2photo/trainB/', path_y='../cezanne2photo/trainA/', is_pair=False)
test_data = DataLoader(path_x='../cezanne2photo/testB/', path_y='../cezanne2photo/testA/', is_pair=False)

# Different lr for generative network and discriminate network
optimizer_dis = tf.train.AdamOptimizer(1e-6)
optimizer_gen = tf.train.AdamOptimizer(1e-4)

loss_fn_mse = keras.losses.MeanSquaredError()
loss_fn_mae = keras.losses.MeanAbsoluteError()

gen_xy = generator(input_shape=[128, 128, 3])
gen_yx = generator(input_shape=[128, 128, 3])

dis = discriminator(input_shape=[128, 128, 3])

# Print Network Information
gen_xy.summary()
gen_yx.summary()
dis.summary()

batch_size = 4
epoch_run = 1000

callback_fun = CallBackFun(experiment_path + train_folder + '/', train_data.tf_imgs_dataset, test_data.tf_imgs_dataset)
callback_fun.on_train_begin()

real_patch = tf.ones((batch_size, ) + (8, 8, 1))  # 128 / (2^4), determined by the discriminator. Idea comes from patchGAN.
fake_patch = tf.zeros((batch_size, ) + (8, 8, 1))

for epoch in range(epoch_run):

    for batch, (x, y) in enumerate(train_data.tf_dataset.batch(batch_size).prefetch(batch_size)):

        with tf.GradientTape() as disc_tape:
            logits_real = loss_fn_mse(dis(y), real_patch)
            logits_fake = loss_fn_mse(dis(gen_xy(x)), fake_patch)
            loss_dis = logits_real + logits_fake

        grads_dis = disc_tape.gradient(loss_dis, dis.variables)
        optimizer_dis.apply_gradients(zip(grads_dis, dis.variables))

        with tf.GradientTape() as gen_tape:

            loss_gen = loss_fn_mse(dis(gen_xy(x)), real_patch)
            loss_gen += 1*(loss_fn_mae(gen_yx(gen_xy(x)), x) + loss_fn_mae(gen_xy(gen_yx(y)), y))

        grads_gen = gen_tape.gradient(loss_gen, gen_xy.variables + gen_yx.variables)
        optimizer_gen.apply_gradients(zip(grads_gen, gen_xy.variables + gen_yx.variables))

        logs = {'loss_gen': loss_gen, 'loss_dis': loss_dis}
        callback_fun.on_train_batch_end(logs)
        print('[Epoch %d Batch %d] [D Loss %.4f] [G Loss %.4f]' % (epoch, batch, loss_dis, loss_gen))

    callback_fun.on_epoch_end(epoch=epoch, gen_xy=gen_xy, gen_yx=gen_yx)

    # Save Network
    if epoch % 20 == 0:
        gen_xy.save(filepath=experiment_path + train_folder + '/gen_xy.model.epoch%d.h5' % epoch)
        gen_yx.save(filepath=experiment_path + train_folder + '/gen_yx.model.epoch%d.h5' % epoch)
        dis.save(filepath=experiment_path + train_folder + '/dis.model.epoch%d.h5' % epoch)
