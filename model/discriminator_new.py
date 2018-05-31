import numpy as np
import tensorflow as tf
from utils.util import conv, fc, upscale, avg_pool, leaky_relu, instance_norm
slim = tf.contrib.slim

# # def discriminator_image(reuse, image_size):
# #     """
# #     Architecture copy from https://github.com/carpedm20/BEGAN-tensorflow/models.py
# #     """
# #     with variable_scope('discriminator_image', reuse=reuse) as vs:
# #         input_real_image = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
# #         input_fake_image = tf.placeholder(tf.float32, (None, image_size, image_size, 3))

# #         x = tf.concat([input_real_image, input_fake_image], 0)
# #         # Encoder
# #         x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
# #         for idx in range(repeat_num):
# #             channel_num = hidden_num * (idx + 1)
# #             x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
# #             x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
# #             if idx < repeat_num - 1:
# #                 x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu)

# #         x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
# #         z = x = slim.fully_connected(x, num_output, activation_fn=None)

# #         # Decoder
# #         num_output = int(np.prod([8, 8, hidden_num]))
# #         x = slim.fully_connected(x, num_output, activation_fn=None)
# #         x = reshape(x, 8, 8, hidden_num)

# #         for idx in range(repeat_num):
# #             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
# #             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
# #             if idx < repeat_num - 1:
# #                 x = upscale(x, 2, data_format)

# #         out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None)

# #     variables = tf.contrib.framework.get_variables(vs)
# #     return out, z, variables

# def discriminator_began(input_real_image, input_fake_image, name, z_num, hidden_num=128, reuse=False, image_size=227, resize_size=64):
#     """
#     Architecture copy from https://github.com/carpedm20/BEGAN-tensorflow/models.py
#     """

#     repeat_num=int(np.log2(resize_size))-2
#     with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
#         # input_real_image = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
#         # input_fake_image = tf.placeholder(tf.float32, (None, image_size, image_size, 3))

#         assert input_real_image.get_shape().as_list()[1:] == [image_size, image_size, 3]
#         assert input_fake_image.get_shape().as_list()[1:] == [image_size, image_size, 3]
#         resize_real_image = tf.image.resize_images(input_real_image, [resize_size, resize_size])
#         resize_fake_image = tf.image.resize_images(input_fake_image, [resize_size, resize_size])

#         x = tf.concat([resize_real_image, resize_fake_image], 0)
#         # Encoder
#         x = conv(x, 3, 3, hidden_num, 1, 1, 'encoder_', activation='elu')
#         for idx in range(repeat_num):
#             channel_num = hidden_num * (idx + 1)
#             x = conv(x, 3, 3, channel_num, 1, 1, 'encoder%d_0'%(idx), activation='elu')
#             x = conv(x, 3, 3, channel_num, 1, 1, 'encoder%d_1'%(idx), activation='elu')
#             if idx < repeat_num - 1:
#                 x = conv(x, 3, 3, channel_num, 2, 2, 'encoder%d_2'%(idx), activation='elu')

#         x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
#         z = x = fc(x, z_num, 'enc_fc', activation=None )

#         # Decoder
#         num_output = int(np.prod([8, 8, hidden_num])) 
#         x = fc(x, num_output, 'dec_fc', activation=None)
#         x = tf.reshape(x, [-1, 8, 8, hidden_num])

#         for idx in range(repeat_num):
#             x = conv(x, 3, 3, hidden_num, 1, 1, 'decode%d_0'%(idx), activation='elu')
#             x = conv(x, 3, 3, hidden_num, 1, 1, 'decode%d_1'%(idx), activation='elu')
#             if idx < repeat_num - 1:
#                 x = upscale(x, 2)

#         out = conv(x, 3, 3, 3, 1, 1, 'decode_', activation=None)
#         output_real_image, output_fake_image = tf.split(out, 2)

#     variables = tf.contrib.framework.get_variables(vs)
#     return resize_real_image, resize_fake_image, output_real_image, output_fake_image, variables

def discriminator_patch(image, map_feature, df_dim, reuse=False, dropout=False, keep_prob=1.0, name="default"):

    assert len(map_feature.get_shape().as_list()) == 3
    map_feature = tf.nn.l2_normalize(map_feature, dim=2, name='normed_map_feature')
    with tf.variable_scope("discriminator_patch_%s"%(name), reuse=reuse) as vs:

        h0 = leaky_relu(instance_norm(conv(image, 4, 4, df_dim, 2, 2, name='d_h0_conv', init='random', activation=None, biased=False)), alpha=0.2)
        h0 = tf.concat([tf.nn.l2_normalize(h0, dim=3, name='l2_normalized_h0'), tf.tile(map_feature[:, :, None, :], [1, 128, 128, 1])], axis=3)

        # # h0 is (128 x 128 x self.df_dim)
        # h1 = leaky_relu(conv(h0, 4, 4, df_dim*2, 2, 2, name='d_h1_conv', init='random', activation=None, biased=False), alpha=0.2)
        # # h1 is (64 x 64 x self.df_dim*2)
        # h2 = leaky_relu(conv(h1, 4, 4, df_dim*4, 2, 2, name='d_h2_conv', init='random', activation=None, biased=False), alpha=0.2)
        # # h2 is (32x 32 x self.df_dim*4)
        # h3 = leaky_relu(conv(h2, 4, 4, df_dim*8, 1, 1, name='d_h3_conv', init='random', activation=None, biased=False), alpha=0.2)
        # # h3 is (32 x 32 x self.df_dim*8)
        # h4 = conv(h3, 4, 4, 1, 1, 1, name='d_h3_pred')
        # # h4 is (32 x 32 x 1)

        # h0 is (128 x 128 x self.df_dim)
        h1 = leaky_relu(instance_norm(conv(h0, 4, 4, df_dim*2, 2, 2, name='d_h1_conv', init='random', activation=None, biased=False),'d_bn1'), alpha=0.2)
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = leaky_relu(instance_norm(conv(h1, 4, 4, df_dim*4, 2, 2, name='d_h2_conv', init='random', activation=None, biased=False), 'd_bn2'), alpha=0.2)
        # h2 is (32x 32 x self.df_dim*4)
        h3 = leaky_relu(instance_norm(conv(h2, 4, 4, df_dim*8, 1, 1, name='d_h3_conv', init='random', activation=None, biased=False), 'd_bn3'), alpha=0.2)
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv(h3, 4, 4, 1, 1, 1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)

        variables = tf.contrib.framework.get_variables(vs)

    return h4, variables
"""
def discriminator_patch(image, df_dim, reuse=False, dropout=False, keep_prob=1.0, name="default"):
    with tf.variable_scope("discriminator_patch_%s"%(name), reuse=reuse) as vs:

        h0 = leaky_relu(conv(image, 4, 4, df_dim, 2, 2, name='d_h0_conv', init='random', activation=None, biased=False), alpha=0.2)
        # h0 is (128 x 128 x self.df_dim)
        h1 = leaky_relu(instance_norm(conv(h0, 4, 4, df_dim*2, 2, 2, name='d_h1_conv', init='random', activation=None, biased=False),'d_bn1'), alpha=0.2)
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = leaky_relu(instance_norm(conv(h1, 4, 4, df_dim*4, 2, 2, name='d_h2_conv', init='random', activation=None, biased=False), 'd_bn2'), alpha=0.2)
        # h2 is (32x 32 x self.df_dim*4)
        h3 = leaky_relu(instance_norm(conv(h2, 4, 4, df_dim*8, 1, 1, name='d_h3_conv', init='random', activation=None, biased=False), 'd_bn3'), alpha=0.2)
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv(h3, 4, 4, 1, 1, 1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)   

        variables = tf.contrib.framework.get_variables(vs)

    return h4, variables
"""
# def discriminator_image(input_image, name, reuse=False):
# def discriminator_image(input_image, input_semantic, name, reuse=False):
def discriminator_image(input_image, real_input_label, fake_input_label, num_train_classes, name='image', reuse=False):
    with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
        dconv1 = conv(input_image, 7, 7, 32, 4, 4, pad='VALID', name='dconv1')
        dconv2 = conv(dconv1, 5, 5, 64, 1, 1, pad='VALID', name='dconv2')
        dconv3 = conv(dconv2, 3, 3, 128, 2, 2, pad='VALID', name='dconv3')
        dconv4 = conv(dconv3, 3, 3, 256, 1, 1, pad='VALID', name='dconv4')
        dconv5 = conv(dconv4, 3, 3, 256, 2, 2, pad='VALID', name='dconv5')
        dpool5 = avg_pool(dconv5, 11, 11, 11, 11, name='dpool5')
        dpool5_reshape = tf.reshape(dpool5, [-1, 256], name='dpool5_reshape')

        # label information
        real_label_feat = tf.one_hot(real_input_label, depth=num_train_classes)
        fake_label_feat = tf.one_hot(fake_input_label, depth=num_train_classes)
        label_feat = tf.concat([real_label_feat, fake_label_feat], axis=0)
        # Ffc1 = fc(label_feat, 512, name='Ffc1')
        # Ffc2 = fc(Ffc1, 256, name='Ffc2')
        # SAVE GPU MEMORY
        Ffc1 = fc(label_feat, 128, name='Ffc1')
        Ffc2 = fc(Ffc1, 128, name='Ffc2')

        concat5 = tf.concat([dpool5_reshape, Ffc2], axis=1, name='concat5')
        # drop5 = tf.nn.dropout(dpool5_reshape, keep_prob=0.5, name='drop5')
        drop5 = tf.nn.dropout(concat5, keep_prob=0.5, name='drop5')
        # dfc6 = fc(drop5, 512, name='dfc6')
        # SAVE GPU MEMORY
        dfc6 = fc(drop5, 256, name='dfc6')
        dfc7 = fc(dfc6, 1, name='dfc7', activation=None)

    variables = tf.contrib.framework.get_variables(vs)

    return dfc7, variables

def discriminator_cycle(vector, name='vector', reuse=False):
    with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
        output = fc(vector, 1, name='output', biased=True, activation=None, trainable=True)
    variables = tf.contrib.framework.get_variables(vs)
    return output, variables


def discriminator_vector(real_vector, fake_vector, name='vector', reuse=False):
    with tf.variable_scope('discriminator_%s'%(name), reuse=reuse) as vs:
        input_  = tf.concat([real_vector, fake_vector], axis=0)
        output = fc(input_, 1, name='output', biased=True, activation=None, trainable=True)
        # fc1 = fc(input_, 1000, name='fc1', trainable=True)
        # fc2 = fc(fc1, 1000, name='fc2', trainable=True)
        # output = fc(fc2, 1, name='output', activation=None, trainable=True)
    variables = tf.contrib.framework.get_variables(vs)
    return output, variables

# def discriminator_noise(real_noise, fake_noise, name='noise'):
#     with tf.variable_scope('discriminator_%s'%(name)) as vs:
#         input_ = tf.concat([real_noise, fake_noise], axis=0)

#         fc1 = fc(input_, 1000, name='fc1', trainable=True)
#         fc2 = fc(fc1, 1000, name='fc2', trainable=True)
#         output = fc(fc2, 1, name='output', activation=None, trainable=True)

#     variables = tf.contrib.framework.get_variables(vs)
#     return output, variables
