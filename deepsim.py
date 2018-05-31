import os
import scipy
from tqdm import tqdm
import numpy as np
import random
import h5py
import tensorflow as tf
from utils import patchShow
from data.preprocessing import read_data
from data.data_utils import norm_feat
from utils.util import hwc2chw, load_pretrained_model, fc
# from model.encoder import encoder_caffenet
# from model.resnet import encoder_resnet
from model.generator import generator_caffenet_fc6
# from model.comparator import comparator_caffenet
# from model.classifier import classifier_caffenet
# from model.discriminator_new import discriminator_image, discriminator_vector
from model.discriminator_new import discriminator_cycle
from model.mapping_new import visual2semantic_2layer, semantic2visual_2layer, semantic2semantic_2layer
from utils.path import get_pretrain_generator, get_pretrain_comparator, get_pretrain_classifier

FLAGS = tf.app.flags.FLAGS 

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image

def _concat_mat(matrix_a, matrix_b):
    return tf.concat([matrix_a, matrix_b], axis=0)

def _split_mat(matrix):
    return tf.split(matrix, 2)

def _checkpath(path):
    assert os.path.exists(path)
    return path

class ZSLGAN(object):
    def __init__(self, dataset_info, net_inputsize, output_imagefeatsize, net_imagefeatsize, recon_size, num_train_classes, num_classes, \
                    semantic_size, summary_path, checkpoint_path, cachefile_path, test1_total_iter, test2_total_iter, data_path):
        margin = FLAGS.margin
        self.num_classes = num_classes
        self.dataset_info = dataset_info
        self.net_inputsize = net_inputsize
        self.semantic_size = semantic_size
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.cachefile_path = cachefile_path
        # self.train_total_iter = train_total_iter
        self.test1_total_iter = test1_total_iter
        self.test2_total_iter = test2_total_iter

        self.train_feat_path = _checkpath(os.path.join(data_path, 'ps_trainval_feat.h5'))
        self.train_image_path = _checkpath(os.path.join(data_path, 'ps_trainval_image.h5'))
        self.test1_feat_path = _checkpath(os.path.join(data_path, 'ps_test_seen_feat.h5'))
        self.test1_image_path = _checkpath(os.path.join(data_path, 'ps_test_seen_image.h5'))
        self.test2_feat_path = _checkpath(os.path.join(data_path, 'ps_test_unseen_feat.h5'))
        self.test2_image_path = _checkpath(os.path.join(data_path, 'ps_test_unseen_image.h5'))
        # self.trainset_path = trainset_path
        # self.seen_testset_path = seen_testset_path
        # self.unseen_testset_path = unseen_testset_path

        self.global_step = tf.Variable(0, trainable=False)

        # self.input_image = tf.placeholder(tf.float32, shape=(None, net_inputsize, net_inputsize, 3), name='input_image')
        self.image_feat = tf.placeholder(tf.float32, shape=(None, output_imagefeatsize), name='image_feat')        
        self.recon_image_feat = tf.placeholder(tf.float32, shape=(None, 4096), name='recon_image_feat')
        self.input_real_semantic = tf.placeholder(tf.float32, shape=(None, 1, semantic_size), name='input_real_semantic')
        self.input_fake_semantic = tf.placeholder(tf.float32, shape=(None, num_train_classes-1, semantic_size), name='input_fake_semantic')
        # self.pretrain_label = tf.placeholder(tf.int32, shape=(None,), name='pretrain_label')

        self.fake_rank_semantic_dis = tf.placeholder(tf.float32, shape=(None, semantic_size), name='fake_rank_semantic_dis')
        # self.fake_recon_semantic_dis = tf.placeholder(tf.float32, shape=(None, 4096), name='fake_recon_semantic_dis')

        self.rank_semantic_pool = ImagePool(50)
        # self.recon_semantic_pool = ImagePool(50)

        self.keep_prob = tf.placeholder(tf.float32, shape=())

        self.recon_topleft = ((recon_size - net_inputsize)/2, (recon_size - net_inputsize)/2)

        self.norm_image_feat = tf.nn.l2_normalize(self.image_feat, dim=1, name='upper_norm_image_feat')
        self.rank_semantic, self.v2s_rank_vars = visual2semantic_2layer(self.norm_image_feat, output_imagefeatsize, semantic_size, name='ranksemantic', reuse=False, dropout=FLAGS.dropout, keep_prob=self.keep_prob)
        self.rank_semantic_sum = tf.summary.scalar('rank_semantic_norm', tf.reduce_mean(tf.norm(self.rank_semantic, axis=1)))

       # Cycle-GAN:
        self.fake_rank_semantic, self.cycle_G_vars = semantic2semantic_2layer(self.recon_image_feat, semantic_size, name='cycle_G', reuse=False, dropout=False)
        self.fake_recon_semantic_, self.cycle_F_vars = semantic2semantic_2layer(self.fake_rank_semantic, 4096, name='cycle_F', reuse=False, dropout=False)

        # self.fake_recon_semantic, _ = semantic2semantic_2layer(self.rank_semantic, 4096, name='cycle_F', reuse=True, dropout=False)
        # self.fake_rank_semantic_, _ = semantic2semantic_2layer(self.fake_recon_semantic, semantic_size, name='cycle_G', reuse=True, dropout=False)

        # self.L_cycle_consist = FLAGS.alpha_cycle_consist * ((tf.losses.mean_squared_error(labels=self.recon_image_feat, predictions=self.fake_recon_semantic_) + \
                                                            # tf.losses.mean_squared_error(labels=self.rank_semantic, predictions=self.fake_rank_semantic_)))

        self.L_cycle_consist = FLAGS.alpha_cycle_consist * tf.losses.mean_squared_error(labels=self.recon_image_feat, predictions=self.fake_recon_semantic_) 



        # self.real_recon_dis_logits, self.discriminator_recon_vars = discriminator_cycle(self.recon_image_feat, name='recon', reuse=False)
        # self.fake_recon_dis_logits, _ = discriminator_cycle(self.fake_recon_semantic_dis, name='recon', reuse=True) 
        # self.dis_loss_recon_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real_recon_dis_logits), logits=self.real_recon_dis_logits))
        # self.dis_loss_recon_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fake_recon_dis_logits), logits=self.fake_recon_dis_logits))
        # self.L_recon_dis = FLAGS.alpha_recon_dis * ((self.dis_loss_recon_real + self.dis_loss_recon_fake) / 2.0)

        # self.fake_recon_gen_logits, _ = discriminator_cycle(self.fake_recon_semantic, name='recon', reuse=True)
        # self.L_recon_gen = FLAGS.alpha_recon_gen * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.fake_recon_gen_logits), logits=self.fake_recon_gen_logits))
        # self.L_recon_dis_sum = tf.summary.scalar('L_recon_dis', self.L_recon_dis / (FLAGS.alpha_recon_dis + 1e-8))
        # self.L_recon_gen_sum = tf.summary.scalar('L_recon_gen', self.L_recon_gen / (FLAGS.alpha_recon_gen + 1e-8))

        self.real_rank_dis_logits, self.discriminator_rank_vars = discriminator_cycle(self.rank_semantic, name='rank', reuse=False)
        self.fake_rank_dis_logits, _ = discriminator_cycle(self.fake_rank_semantic_dis, name='rank', reuse=True) 
        self.dis_loss_rank_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real_rank_dis_logits), logits=self.real_rank_dis_logits))
        self.dis_loss_rank_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fake_rank_dis_logits), logits=self.fake_rank_dis_logits))
        self.L_rank_dis = FLAGS.alpha_rank_dis * ((self.dis_loss_rank_real + self.dis_loss_rank_fake) / 2.0)

        self.fake_rank_gen_logits, _ = discriminator_cycle(self.fake_rank_semantic, name='rank', reuse=True)
        self.L_rank_gen = FLAGS.alpha_rank_gen * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.fake_rank_gen_logits), logits=self.fake_rank_gen_logits))
        self.L_rank_dis_sum = tf.summary.scalar('L_rank_dis', self.L_rank_dis / (FLAGS.alpha_rank_dis + 1e-8))
        self.L_rank_gen_sum = tf.summary.scalar('L_rank_gen', self.L_rank_gen / (FLAGS.alpha_rank_gen + 1e-8))

        rank_positive_term = tf.reduce_sum(self.rank_semantic[:, None, :] * self.input_real_semantic, axis=2)
        rank_negative_term = tf.reduce_sum(self.rank_semantic[:, None, :] * self.input_fake_semantic, axis=2)
        all_rank_values = margin - rank_positive_term + rank_negative_term

        if FLAGS.rank_loss_type == 'mean':
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, all_rank_values))
        elif FLAGS.rank_loss_type == 'random':
            break_rules = tf.cast((all_rank_values > 0), tf.float32)
            break_prob = (break_rules / tf.reduce_sum(break_rules, axis=1)[:, None])
            break_ind = tf.multinomial(tf.log(break_prob), 1)
            N = tf.cast(tf.shape(all_rank_values)[0], dtype=tf.int64) # for train, FLAGS.batch_size, for test, FLAGS.test_batch_size
            select_ind = tf.transpose(tf.concat([tf.range(N, dtype=tf.int64)[tf.newaxis, :], \
                                                 tf.squeeze(break_ind, [1])[tf.newaxis, :]], axis=0))
            select_rank_values = tf.gather_nd(params=all_rank_values, indices=select_ind)
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, select_rank_values))
        elif FLAGS.rank_loss_type == 'max':
            max_rank_negative_term = tf.reduce_max(rank_negative_term, axis=1)
            self.rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, margin - rank_positive_term + max_rank_negative_term))
        else:
            raise NotImplementedError

        # for test, rank loss use mean
        self.test_rank_loss = FLAGS.alpha_rank * tf.reduce_mean(tf.maximum(0.0, all_rank_values))

        # Generator
#        # if FLAGS.generator == 'caffenet' and FLAGS.feat == 'fc6':
#        #     self._recon_image, self.generator_vars = generator_caffenet_fc6(self.recon_image_feat, reuse=False, trainable=False)
#        #     self._recon_image = tf.reshape(self._recon_image, (-1, recon_size, recon_size, 3))
#        #     self.recon_image = self._recon_image[:, self.recon_topleft[0]:self.recon_topleft[0]+net_inputsize, 
#        #                                         self.recon_topleft[1]:self.recon_topleft[1]+net_inputsize, :]

#        #     self._fake_recon_image, _ = generator_caffenet_fc6(self.fake_recon_semantic_, reuse=True, trainable=False)
#        #     self._fake_recon_image = tf.reshape(self._fake_recon_image, (-1, recon_size, recon_size, 3))
#        #     self.fake_recon_image = self._fake_recon_image[:, self.recon_topleft[0]:self.recon_topleft[0]+net_inputsize, 
#        #                                         self.recon_topleft[1]:self.recon_topleft[1]+net_inputsize, :]

        # # Comparator
        # if FLAGS.comparator == 'caffenet':
        #     if FLAGS.encoder == 'caffenet':         
        #         self.recon_image_comp, self.comparator_vars = comparator_caffenet(self.recon_image, reuse=False, trainable=False)
        #     elif FLAGS.encoder == 'resnet':
        #         self.recon_image_comp, self.comparator_vars = comparator_caffenet(tf.image.resize_image_with_crop_or_pad(self.recon_image, 227, 227), reuse=False, trainable=False)

        # # Classifier
        # if FLAGS.classifier == 'caffenet':
        #     if FLAGS.encoder == 'caffenet':
        #         self.recon_image_classfc8, self.classifier_vars= classifier_caffenet(self.recon_image, num_train_classes, reuse=False, trainable=False)
        #     elif FLAGS.encoder == 'resnet':
        #         self.recon_image_classfc8, self.classifier_vars = classifier_caffenet(tf.image.resize_image_with_crop_or_pad(self.recon_image, 227, 227), \
        #                                                                                 num_train_classes, reuse=False, trainable=False)

        print "Finish Build Model!"

        # self.L_recon_img = FLAGS.alpha_recon_img * tf.losses.mean_squared_error(labels=self.input_image, predictions=self.recon_image) 
        # self.L_recon_feat = FLAGS.alpha_recon_feat * tf.losses.mean_squared_error(labels=self.comp_image_feat, predictions=self.recon_image_comp)
        # self.L_class = FLAGS.alpha_class * tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pretrain_label, logits=(self.recon_image_classfc8 + 1e-8)))

        # self.L_recon_img_sum = tf.summary.scalar('L_recon_img', self.L_recon_img / (FLAGS.alpha_recon_img + 1e-8))
        # self.L_recon_feat_sum = tf.summary.scalar('L_recon_feat', self.L_recon_feat / (FLAGS.alpha_recon_feat + 1e-8))
        # self.L_class_sum = tf.summary.scalar('L_class', self.L_class / (FLAGS.alpha_class + 1e-8))
       
        self.rank_loss_sum = tf.summary.scalar('rank_loss', self.rank_loss / (FLAGS.alpha_rank + 1e-8))

        for var in self.v2s_rank_vars: 
            if 'weights' in var.op.name:
                tf.add_to_collection('rank_regularization', tf.nn.l2_loss(var))
        self.rank_regul_loss = FLAGS.alpha_rank_regularization * tf.add_n(tf.get_collection('rank_regularization'))

        # for var in v2s_recon_vars+s2s_recon_vars+s2s_rank_vars+s2v_vars:
        # for var in self.v2s_recon_vars+self.s2s_rank_vars+self.s2s_recon_vars+self.s2v_vars+self.cycle_G_vars+self.cycle_F_vars:
        for var in self.cycle_G_vars+self.cycle_F_vars:
            if 'weights' in var.op.name:
                tf.add_to_collection('cycle_regularization', tf.nn.l2_loss(var))
        self.recon_regul_loss = FLAGS.alpha_cycle_regularization * tf.add_n(tf.get_collection('cycle_regularization'))

        # self.cycle_loss = self.L_cycle_consist + self.L_rank_gen + self.L_recon_gen
        self.cycle_loss = self.L_cycle_consist + self.L_rank_gen
        self.cycle_loss_sum = tf.summary.scalar('Cycle_loss', self.cycle_loss)
        self.total_loss = self.rank_loss + self.cycle_loss + self.rank_regul_loss + self.recon_regul_loss
        self.total_loss_sum = tf.summary.scalar('Total_Loss', self.total_loss)

        # self.L_SUM = tf.summary.merge([self.total_loss_sum, self.L_recon_img_sum, self.L_recon_feat_sum, self.L_class_sum, self.rank_loss_sum, self.L_rank_gen_sum, self.L_recon_gen_sum])
        # self.L_SUM = tf.summary.merge([self.total_loss_sum, self.L_rank_gen_sum, self.L_recon_gen_sum, self.cycle_loss_sum])
        self.L_SUM = tf.summary.merge([self.total_loss_sum, self.L_rank_gen_sum, self.cycle_loss_sum])
        self.NORM_SUM = tf.summary.merge([self.rank_semantic_sum])

        # self.DIS_SUM = tf.summary.merge([self.L_recon_dis_sum, self.L_rank_dis_sum])
        self.DIS_SUM = tf.summary.merge([self.L_rank_dis_sum])

        # Optimizers
        if FLAGS.decay_type == 'ed':
            self.decayed_learning_rate = tf.train.exponential_decay(FLAGS.lr, self.global_step, FLAGS.ed_decay_steps, FLAGS.ed_decay_rate, \
                                                                    staircase=True, name='decayed_learning_rate') 
        elif FLAGS.decay_type == 'pc':
            lr_boundaries = [FLAGS.decay_curriculum]
            lr_values = [FLAGS.cycle_lr, FLAGS.cycle_lr*0.1]
            self.decayed_learning_rate = tf.train.piecewise_constant(self.global_step, lr_boundaries, lr_values)


        if FLAGS.optimizer == 'adam':
            cycle_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        elif FLAGS.optimizer == 'sgd':
            cycle_trainer = tf.train.MomentumOptimizer(self.decayed_learning_rate, momentum=0.9, use_nesterov=True)

        cycle_grads_and_vars = cycle_trainer.compute_gradients(self.total_loss, self.cycle_G_vars+self.cycle_F_vars+self.v2s_rank_vars)
        cycle_trainer_clip = [(tf.clip_by_norm(cycle_op_grad, 10.), cycle_op_var) for cycle_op_grad, cycle_op_var in cycle_grads_and_vars]
        self.cycle_optimizer = (cycle_trainer.apply_gradients(cycle_trainer_clip))

        if FLAGS.optimizer == 'adam':
            dis_rank_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        elif FLAGS.optimizer == 'sgd':
            dis_rank_trainer = tf.train.MomentumOptimizer(self.decayed_learning_rate, momentum=0.9, use_nesterov=True)
        dis_rank_grads_and_vars = dis_rank_trainer.compute_gradients(self.L_rank_dis, self.discriminator_rank_vars)
        dis_rank_trainer_clip = [(tf.clip_by_norm(dis_rank_op_grad, 10.), dis_rank_op_var) for dis_rank_op_grad, dis_rank_op_var in dis_rank_grads_and_vars]
        self.dis_rank_optimizer = (dis_rank_trainer.apply_gradients(dis_rank_trainer_clip))

        # if FLAGS.optimizer == 'adam':
        #     dis_recon_trainer = tf.train.AdamOptimizer(self.decayed_learning_rate, beta1=FLAGS.beta1, beta2=FLAGS.beta2)
        # elif FLAGS.optimizer == 'sgd':
        #     dis_recon_trainer = tf.train.MomentumOptimizer(self.decayed_learning_rate, momentum=0.9, use_nesterov=True)
        # dis_recon_grads_and_vars = dis_recon_trainer.compute_gradients(self.L_recon_dis, self.discriminator_recon_vars)
        # dis_recon_trainer_clip = [(tf.clip_by_norm(dis_recon_op_grad, 10.), dis_recon_op_var) for dis_recon_op_grad, dis_recon_op_var in dis_recon_grads_and_vars]
        # self.dis_recon_optimizer = (dis_recon_trainer.apply_gradients(dis_recon_trainer_clip))

        self.update_global_step = tf.assign(self.global_step, self.global_step+1)
        self.clear_global_step = tf.assign(self.global_step, 0)
        # self.recon_global_step = tf.assign(self.global_step, FLAGS.recon_curriculum)

        self.summary_writer = tf.summary.FileWriter(summary_path)
        # self.pretrain_saver = tf.train.Saver(v2s_vars)

#        # generator_pretrain_vars = {}
#        # for var in self.generator_vars:
#        #     assert 'generator' in var.op.name
#        #     new_var_name = "/".join(var.op.name.split('/')[1:])
#        #     generator_pretrain_vars[new_var_name] = var
#       # self.generator_saver = tf.train.Saver(generator_pretrain_vars)

        # comparator_pretrain_vars = {}
        # for var in self.comparator_vars:
        #     assert 'comparator' in var.op.name
        #     new_var_name = "/".join(var.op.name.split('/')[1:])
        #     comparator_pretrain_vars[new_var_name] = var
        # self.comparator_saver = tf.train.Saver(comparator_pretrain_vars)

        # classifier_pretrain_vars = {}
        # for var in self.classifier_vars:
        #     assert 'classifier' in var.op.name
        #     new_var_name = "/".join(var.op.name.split('/')[1:])
        #     classifier_pretrain_vars[new_var_name] = var
        # self.classifier_saver = tf.train.Saver(classifier_pretrain_vars)

        # self.save_vars = v2s_rank_vars + v2s_recon_vars + s2s_recon_vars + s2s_rank_vars + s2v_vars

        # self.without_cycle_vars = []
        # for each_save_var in tf.global_variables():
        #     # if not each_save_var in self.generator_vars + self.comparator_vars + self.classifier_vars:
        #     if (not 'discriminator' in each_save_var.op.name) and (not 'cycle' in each_save_var.op.name):
        #         self.without_cycle_vars.append(each_save_var)
            
        # self.without_cycle_saver = tf.train.Saver(self.without_cycle_vars, max_to_keep=10)

        self.saver = tf.train.Saver(max_to_keep=100)


        self.unseen_test_mask = np.array(self.dataset_info['test_unseen_class_num'].keys())
        self.trainval_mask = np.array(self.dataset_info['trainval_class_num'].keys())

    def get_fake_semantic(self, all_fake_semantic, except_class_list):
        fake_semantic = []
        for class_i in except_class_list:
            temp_trainval_mask = self.trainval_mask
            fake_semantic.append(all_fake_semantic[np.delete(temp_trainval_mask,
                                                            np.where(temp_trainval_mask == class_i))-1])
        return np.array(fake_semantic)

    def get_NN(self, query_vector, test_mask, semantic_type, mode):
        if semantic_type == 'glove':
            assert query_vector.shape[-1] == self.dataset_info['class_glove'].shape[-1]
            if mode == 'zsl':
                dist = np.sum((self.dataset_info['class_glove'][test_mask-1] - query_vector) ** 2, axis=1)
                return test_mask[np.argmin(dist)]
            elif mode == 'gzsl':
                dist = np.sum((self.dataset_info['class_glove'] - query_vector) ** 2, axis=1)
                return (np.argmin(dist) + 1).astype('int32')
            else:
                raise NotImplementedError
        elif semantic_type == 'word2vec':
            assert query_vector.shape[-1] == self.dataset_info['class_word2vec'].shape[-1]
            if mode == 'zsl':
                dist = np.sum((self.dataset_info['class_word2vec'][test_mask-1] - query_vector) ** 2, axis=1)
                return test_mask[np.argmin(dist)]
            elif mode == 'gzsl':
                dist = np.sum((self.dataset_info['class_word2vec'] - query_vector) ** 2, axis=1)
                return (np.argmin(dist) + 1).astype('int32')
            else:
                raise NotImplementedError
        elif semantic_type == 'class_attr':
            assert query_vector.shape[-1] == self.dataset_info['class_attr'].shape[-1]
            if mode == 'zsl':
                dist = np.sum((self.dataset_info['class_attr'][test_mask-1] - query_vector) ** 2, axis=1)
                return test_mask[np.argmin(dist)]
            elif mode == 'gzsl':
                dist = np.sum((self.dataset_info['class_attr'] - query_vector) ** 2, axis=1)
                return (np.argmin(dist) + 1).astype('int32')
            else:
                raise NotImplementedError
        elif semantic_type == 'image_attr':
            assert query_vector.shape[-1] == self.dataset_info['image_attr'].shape[-1]
            if mode == 'zsl':
                dist = np.sum((self.dataset_info['image_attr'][test_mask-1] - query_vector) ** 2, axis=1)
                return test_mask[np.argmin(dist)]
            elif mode == 'gzsl':
                dist = np.sum((self.dataset_info['image_attr'] - query_vector) ** 2, axis=1)
                return (np.argmin(dist) + 1).astype('int32')
            else:
                raise NotImplementedError               
        else:
            raise NotImplementedError

    def get_RANK(self, query_semantic, test_mask, semantic_type):
        if semantic_type == 'class_attr':
            if FLAGS.negative_class_attr:
                dist = np.dot(query_semantic, self.dataset_info['negative_norm_class_attr'][test_mask-1].transpose())
            else:
                if FLAGS.dataset == 'cub':
                    dist = np.dot(query_semantic, self.dataset_info['positive_norm_class_attr'][test_mask-1].transpose())                    
                else:
                    dist = np.dot(query_semantic, self.dataset_info['class_attr'][test_mask-1].transpose())
            return test_mask[np.argmax(dist, axis=1)]
        else:
            raise NotImplementedError

    def get_RANK_combine(self, query_rank_semantic, query_fake_semantic, test_mask, semantic_type, weight):
        # combine two type of semantic vector to get a high score
        if semantic_type == 'class_attr':
            if FLAGS.negative_class_attr:
                dist1 = np.dot(query_rank_semantic, self.dataset_info['negative_norm_class_attr'][test_mask-1].transpose())
                dist2 = np.dot(query_fake_semantic, self.dataset_info['negative_norm_class_attr'][test_mask-1].transpose())
                dist = dist1 + weight * dist2
            else:
                if FLAGS.dataset == 'cub':
                    dist1 = np.dot(query_rank_semantic, self.dataset_info['positive_norm_class_attr'][test_mask-1].transpose())                    
                    dist2 = np.dot(query_fake_semantic, self.dataset_info['positive_norm_class_attr'][test_mask-1].transpose())
                    dist = dist1 + weight * dist2
                else:
                    dist1 = np.dot(query_rank_semantic, self.dataset_info['class_attr'][test_mask-1].transpose())
                    dist2 = np.dot(query_fake_semantic, self.dataset_info['class_attr'][test_mask-1].transpose())
                    dist = dist1 + weight * dist2 
            return test_mask[np.argmax(dist, axis=1)]
        else:
            raise NotImplementedError

    def compute_class_accuracy(self, total_number_dict, right_number_dict):
        class_accuracy = {}
        assert set(total_number_dict.keys()) == set(right_number_dict.keys())
        for key in total_number_dict.keys():
            assert key not in class_accuracy
            class_accuracy[key] = 1.0 * right_number_dict[key] / total_number_dict[key]
        return class_accuracy, np.mean(class_accuracy.values())

    def compute_class_accuracy_total(self, true_label, predict_label, classes):
        true_label = true_label[:, 0]
        nclass = len(classes)
        acc_per_class = np.zeros((nclass, 1))
        for i, class_i in enumerate(classes):
            idx = np.where(true_label == class_i)[0]
            acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))

        return np.mean(acc_per_class)

    def load(self, sess):
        # Load pretrained model
#        # generator_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_generator(FLAGS.generator, load_type='tf')))
#        # if not generator_stats:
#        #     load_pretrained_model('generator', get_pretrain_generator(FLAGS.generator, load_type='np'), sess, ignore_missing=True)
#        #     self.generator_saver.save(sess, get_pretrain_generator(FLAGS.generator, load_type='tf'))
#        # self.generator_saver.restore(sess, get_pretrain_generator(FLAGS.generator, load_type='tf'))

        # # load_pretrained_model('comparator', get_pretrain_comparator(FLAGS.comparator), sess, ignore_missing=True)
        # comparator_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_comparator(FLAGS.comparator, load_type='tf')))
        # self.comparator_saver.restore(sess, comparator_stats.model_checkpoint_path)

        # #load_pretrained_model('classifier', get_pretrain_classifier(FLAGS.classifier), sess, ignore_missing=True)
        # classifier_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_classifier(FLAGS.classifier, load_type='tf')))
        # if not classifier_stats:
        #     load_pretrained_model('classifier', get_pretrain_classifier(FLAGS.classifier, load_type='np'), sess, ignore_missing=True)
        #     self.classifier_saver.save(sess, get_pretrain_classifier(FLAGS.classifier, load_type='tf'))
        #     classifier_stats = tf.train.get_checkpoint_state(os.path.dirname(get_pretrain_classifier(FLAGS.classifier, load_type='tf')))
        # self.classifier_saver.restore(sess, classifier_stats.model_checkpoint_path)
        pass

    def sae(self, sess, sae_name):
        if FLAGS.retrain_model and (FLAGS.test_checkpoint != ''):
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            # self.pretrain_saver.restore(sess, ckpt_state.model_checkpoint_path)
            self.saver.restore(sess, ckpt_state.model_checkpoint_path)
            print "Restore parameters from checkpoint!"

        print "Load data to memory >>>>>>>>>>>>"

        # train_image_h5 = h5py.File(self.train_image_path, 'r')
        train_feat_h5 = h5py.File(self.train_feat_path, 'r')
        # test1_image_h5 = h5py.File(self.test1_image_path, 'r')
        test1_feat_h5 = h5py.File(self.test1_feat_path, 'r')
        # test2_image_h5 = h5py.File(self.test2_image_path, 'r')
        test2_feat_h5 = h5py.File(self.test2_feat_path, 'r')

        # self.test2_cl_list = test2_feat_h5['class'][:]

        train_img_feat_list = train_feat_h5['image_feat'][:]
        train_recon_encoder_img_feat_list = train_feat_h5['recon_image_feat'][:]
        train_img_len = len(train_img_feat_list)
        train_img_feat_list = train_img_feat_list.reshape((10*train_img_len, -1))
        train_recon_encoder_img_feat_list = train_recon_encoder_img_feat_list.reshape((10*train_img_len, -1))

        test1_img_feat_list = test1_feat_h5['image_feat'][:]
        test1_recon_encoder_img_feat_list = test1_feat_h5['recon_image_feat'][:]
        test2_img_feat_list = test2_feat_h5['image_feat'][:]
        test2_recon_encoder_img_feat_list = test2_feat_h5['recon_image_feat'][:]

        print 'Finish load data to memory'       

        _train_rank_semantic = []
        _train_fake_rank_semantic = []
        for each_train_img_feat, each_recon_encoder_img_feat in tqdm(zip(train_img_feat_list, train_recon_encoder_img_feat_list)):
            [_each_train_rank_semantic, _each_train_fake_rank_semantic] = sess.run([self.rank_semantic, self.fake_rank_semantic],
                        feed_dict={self.image_feat:  each_train_img_feat[None, :],
                        self.recon_image_feat: each_recon_encoder_img_feat[None, :],
                        self.keep_prob: 1.0})
            _train_rank_semantic.append(_each_train_rank_semantic)
            _train_fake_rank_semantic.append(_each_train_fake_rank_semantic)
        _train_rank_semantic = np.concatenate(_train_rank_semantic, axis=0)
        _train_fake_rank_semantic = np.concatenate(_train_fake_rank_semantic, axis=0)

        [_test1_rank_semantic, _test1_fake_rank_semantic] = sess.run([self.rank_semantic, self.fake_rank_semantic],
                    feed_dict={self.image_feat: test1_img_feat_list,
                    self.recon_image_feat: test1_recon_encoder_img_feat_list,
                    self.keep_prob: 1.0})

        [_test2_rank_semantic, _test2_fake_rank_semantic] = sess.run([self.rank_semantic, self.fake_rank_semantic],
                    feed_dict={self.image_feat: test2_img_feat_list,
                    self.recon_image_feat: test2_recon_encoder_img_feat_list,
                    self.keep_prob: 1.0})

        np.savez(sae_name, train_semantic=_train_rank_semantic, train_visual=train_recon_encoder_img_feat_list, train_fake_semantic = _train_fake_rank_semantic,
                        test1_semantic=_test1_rank_semantic, test1_visual=test1_recon_encoder_img_feat_list, test1_fake_semantic = _test1_fake_rank_semantic,
                        test2_semantic=_test2_rank_semantic, test2_visual=test2_recon_encoder_img_feat_list, test2_fake_semantic = _test2_fake_rank_semantic)        

        # np.savez(sae_name, test2_semantic=_test2_rank_semantic, test2_visual=test2_recon_encoder_img_feat_list, test2_fake_semantic = _test2_fake_rank_semantic)


    def train_h5(self, sess):

        if FLAGS.retrain_model and (FLAGS.test_checkpoint != ''):
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
            # self.pretrain_saver.restore(sess, ckpt_state.model_checkpoint_path)
            self.saver.restore(sess, ckpt_state.model_checkpoint_path)
            print "Restore parameters from checkpoint!"

        sess.run([self.clear_global_step])

        zsl_best_test2_accuracy = -1
        gzsl_best_test1_accuracy = -1
        gzsl_best_test2_accuracy = -1
        best_H = -1

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        print "Load data to memory >>>>>>>>>>>>"
        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}

        train_image_h5 = h5py.File(self.train_image_path, 'r')
        train_feat_h5 = h5py.File(self.train_feat_path, 'r')
        test1_image_h5 = h5py.File(self.test1_image_path, 'r')
        test1_feat_h5 = h5py.File(self.test1_feat_path, 'r')
        test2_image_h5 = h5py.File(self.test2_image_path, 'r')
        test2_feat_h5 = h5py.File(self.test2_feat_path, 'r')

        self.train_cl_list = train_feat_h5['class'][:]
        self.train_semantic_list = train_feat_h5['class_attr'][:]
        self.train_img_feat_list = train_feat_h5['image_feat'][:]
        self.train_recon_encoder_img_feat_list = train_feat_h5['recon_image_feat'][:]

        train_image_num = len(self.train_cl_list)
        self.train_img_feat_list = np.reshape(self.train_img_feat_list, (10*train_image_num, 2048))
        self.train_recon_encoder_img_feat_list = np.reshape(self.train_recon_encoder_img_feat_list, (10*train_image_num, 4096))
        train_sample_num = 10*train_image_num

        self.test1_cl_list = test1_feat_h5['class'][:]
        self.test1_semantic_list = test1_feat_h5['class_attr'][:]
        self.test1_img_feat_list = test1_feat_h5['image_feat'][:]
        self.test1_recon_encoder_img_feat_list = test1_feat_h5['recon_image_feat'][:]

        self.test2_cl_list = test2_feat_h5['class'][:]
        self.test2_semantic_list = test2_feat_h5['class_attr'][:]
        self.test2_img_feat_list = test2_feat_h5['image_feat'][:]
        self.test2_recon_encoder_img_feat_list = test2_feat_h5['recon_image_feat'][:]
        print 'Finish load data to memory'

        read_offset = 0
        shuffle_sample_index = np.arange(train_sample_num)
        np.random.shuffle(shuffle_sample_index)
        for i in xrange(FLAGS.training_step):

            if i % FLAGS.ckpt_interval == 0  and i > 0:
                save_path = os.path.join(self.checkpoint_path, '%4d'%(i), FLAGS.model_name)
                print "save checkpoint in %s"%(save_path)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                self.saver.save(sess, save_path, global_step=i)

      
            if (i % FLAGS.validate_interval == 0): 

                zsl_test_unseen_total_acc = -1
                gzsl_test_unseen_total_acc = -1
                gzsl_test_seen_total_acc = -1

                [_test1_rank_semantic] = sess.run([self.rank_semantic],
                            feed_dict={self.image_feat: self.test1_img_feat_list,
                            self.recon_image_feat: self.test1_recon_encoder_img_feat_list,
                            self.input_real_semantic: self.test1_semantic_list[:, None, :],
                            self.keep_prob: 1.0})

                # S->S
                zsl_test_seen_pred_label = self.get_RANK(_test1_rank_semantic, self.trainval_mask, FLAGS.semantic)
                zsl_test_seen_total_acc = self.compute_class_accuracy_total(self.test1_cl_list[:, None], zsl_test_seen_pred_label, self.trainval_mask)

                print 'S->S Accuracy: %.6f' %(zsl_test_seen_total_acc)

                gzsl_test_seen_pred_label= self.get_RANK(_test1_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
                gzsl_test_seen_total_acc = self.compute_class_accuracy_total(self.test1_cl_list[:, None], gzsl_test_seen_pred_label, self.trainval_mask)

                print "Total GZSL Accuracy: %.6f" %(gzsl_test_seen_total_acc)

                [_test2_rank_semantic] = sess.run([self.rank_semantic],
                            feed_dict={self.image_feat: self.test2_img_feat_list,
                            self.recon_image_feat: self.test2_recon_encoder_img_feat_list,
                            self.input_real_semantic: self.test2_semantic_list[:, None, :],
                            self.keep_prob: 1.0})


                zsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, self.unseen_test_mask, FLAGS.semantic)
                zsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list[:, None], zsl_test_unseen_pred_label, self.unseen_test_mask) 

                gzsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
                gzsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list[:, None], gzsl_test_unseen_pred_label, self.unseen_test_mask)

                gzsl_H = (2*gzsl_test_unseen_total_acc*gzsl_test_seen_total_acc)/(gzsl_test_unseen_total_acc+gzsl_test_seen_total_acc+1e-8)
                print "Total ZSL Accuracy: %.6f, Total GZSL Accuracy: %.6f, GZSL H: %.6f" %(zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc, gzsl_H)

                if gzsl_H > best_H and FLAGS.save_checkpoint:
                    gzsl_best_test1_accuracy = gzsl_test_seen_total_acc
                    zsl_best_test2_accuracy = zsl_test_unseen_total_acc
                    gzsl_best_test2_accuracy = gzsl_test_unseen_total_acc
                    best_H = gzsl_H
                    if best_H > 0.0:
                        gzsl_save_path = os.path.join(self.checkpoint_path, \
                                                      '%.6f_%.6f_%.6f_%.6f'%(gzsl_best_test1_accuracy, zsl_best_test2_accuracy, gzsl_best_test2_accuracy, best_H), \
                                                      FLAGS.model_name)
                        print "Save checkpoint in %s"%(gzsl_save_path)
                        if not os.path.exists(os.path.dirname(gzsl_save_path)):
                            os.makedirs(os.path.dirname(gzsl_save_path))
                        self.saver.save(sess, gzsl_save_path, global_step=i)
                        self.gzsl_best_save_path = gzsl_save_path

                summary_eval = tf.Summary(value=[tf.Summary.Value(tag="test1_gzsl_accuracy", simple_value=gzsl_test_seen_total_acc),
                                                tf.Summary.Value(tag="test2_gzsl_accuracy", simple_value=gzsl_test_unseen_total_acc),
                                                tf.Summary.Value(tag="test2_zsl_accuracy", simple_value=zsl_test_unseen_total_acc),
                                                tf.Summary.Value(tag='gzsl_H', simple_value=gzsl_H)])

                if not os.path.exists(self.summary_path):
                    os.makedirs(self.summary_path)
                self.summary_writer.add_summary(summary_eval, i)

            # read train batch data
            if read_offset + FLAGS.batch_size > train_sample_num:
                read_offset = 0
                np.random.shuffle(shuffle_sample_index)
            
            batch_sample_index = shuffle_sample_index[read_offset: read_offset+FLAGS.batch_size]
            batch_image_index = batch_sample_index / 10

            train_cl = self.train_cl_list[batch_image_index]
            train_img_feat = self.train_img_feat_list[batch_sample_index]
            train_semantic = self.train_semantic_list[batch_image_index]
            train_recon_img_feat = self.train_recon_encoder_img_feat_list[batch_image_index]

            read_offset = read_offset + FLAGS.batch_size


            [_, _fake_rank_semantic, _rank_semantic, _train_loss, _rank_loss, _L_SUM, _lr, _rank_regul_loss, _recon_regul_loss, \
                _L_cycle_consist, _L_rank_gen] = \
                sess.run([self.cycle_optimizer, self.fake_rank_semantic, self.rank_semantic, self.total_loss, self.rank_loss, self.L_SUM, self.decayed_learning_rate, \
                        self.rank_regul_loss, self.recon_regul_loss, \
                        self.L_cycle_consist, self.L_rank_gen],
                                                        feed_dict={self.image_feat: train_img_feat,
                                                                   self.recon_image_feat: train_recon_img_feat,
                                                                   # self.input_image: train_img1,
                                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                   # self.pretrain_label: train_pretrain_cl1,
                                                                   self.keep_prob: FLAGS.keep_prob})

            [_, _DIS_SUM, _L_rank_dis]= \
                sess.run([self.dis_rank_optimizer, self.DIS_SUM, self.L_rank_dis],
                                                        feed_dict={self.image_feat: train_img_feat,
                                                                   self.recon_image_feat: train_recon_img_feat,
                                                                   # self.recon_image_feat: train_recon_img_feat,
                                                                   # self.input_image: train_img1,
                                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                   # self.pretrain_label: train_pretrain_cl1,
                                                                   self.fake_rank_semantic_dis: self.rank_semantic_pool.query(_fake_rank_semantic),
                                                                   # self.fake_recon_semantic_dis: self.recon_semantic_pool.query(_fake_recon_semantic),
                                                                   self.keep_prob: FLAGS.keep_prob})



            if i % FLAGS.summary_interval == 0:

#                # [_NORM_SUM, _save_recon_image, _save_fake_recon_image] = sess.run([self.NORM_SUM, self.recon_image, self.fake_recon_image],
#                #                                         feed_dict={self.image_feat: train_img_feat,
#                #                                                    self.recon_image_feat: train_recon_img_feat,
#                #                                                    # self.input_image: train_img1,
#                #                                                    self.input_real_semantic: train_semantic[:, None, :],
#                #                                                    self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
#                #                                                    # self.pretrain_label: train_pretrain_cl1,
#                #                                                    self.keep_prob: FLAGS.keep_prob})

                [_NORM_SUM] = sess.run([self.NORM_SUM],
                                                        feed_dict={self.image_feat: train_img_feat,
                                                                   self.recon_image_feat: train_recon_img_feat,
                                                                   # self.input_image: train_img1,
                                                                   self.input_real_semantic: train_semantic[:, None, :],
                                                                   self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl),
                                                                   # self.pretrain_label: train_pretrain_cl1,
                                                                   self.keep_prob: FLAGS.keep_prob})                


                self.summary_writer.add_summary(_NORM_SUM, i)

            self.summary_writer.add_summary(_DIS_SUM, i)
            self.summary_writer.add_summary(_L_SUM, i)

            if i % 50 == 0:
                this_step_accuracy = np.mean(np.equal(self.get_RANK(_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic), train_cl))            
                # this_step_accuracy = np.mean(np.equal(self.get_RANK(_rank_semantic, trainval_mask, FLAGS.semantic, 'zsl'), train_cl))
                print "Step %d, Training loss: %.6f, Rank loss: %.6f, Learning rate: %.6f, Accuracy: %.6f "%(i, _train_loss, _rank_loss, _lr, this_step_accuracy)

                if FLAGS.debug:
                    # print"Debug>>> Upper recon image loss: %.6f, Upper recon feat loss: %.6f, Upper class loss: %.6f"%(_L_upper_recon_img, _L_upper_recon_feat, _L_upper_class)
                    # print"Debug>>> Below recon image loss: %.6f, Below recon feat loss: %.6f, Below class loss: %.6f"%(_L_below_recon_img, _L_below_recon_feat, _L_below_class)
                    # # print"Debug>>> Below semantic rank: %.6f, Below semantic recon: %.6f"%(_L_below_semantic_rank, _L_below_semantic_recon)
                    # print"Debug>>> Rank Regul loss: %.6f, Recon Regul loss: %.6f"%( _rank_regul_loss, _recon_regul_loss)
                    # print"Debug>>> Dis1 loss: %.6f, Gen1 loss: %.6f"%(_L_recon_adv1, _gen1_loss)
                    # print"Debug>>> Dis2 loss: %.6f, Gen2 loss: %.6f"%(_L_recon_adv2, _gen2_loss)
                    # print"Debug>>> Dis3 loss: %.6f, Gen3 loss: %.6f"%(_L_recon_adv3, _gen3_loss)

                    # print"Debug>>> recon image loss: %.6f, recon feat loss: %.6f, class loss: %.6f"%(_L_recon_img, _L_recon_feat, _L_class)
                    print"Debug>>> Rank Regul loss: %.6f, Recon Regul loss: %.6f"%( _rank_regul_loss, _recon_regul_loss)
                    # print"Debug>>> Cycle consist loss: %.6f, Recon dis loss: %.6f, Recon gen loss: %.6f, Semantic l2 loss: %.6f"%(_L_cycle_consist, _L_recon_dis, _L_recon_gen, _L_semantic_l2)
                    print"Debug>>> Cycle consist loss: %.6f, Rank dis loss: %.6f, Rank gen loss: %.6f"%(_L_cycle_consist, _L_rank_dis, _L_rank_gen)
     
#            # if i % FLAGS.saveimage_interval == 0:

#            #     collage = patchShow.patchShow(np.concatenate((hwc2chw(train_img[:, :, :, ::-1]), hwc2chw(_save_recon_image[:, :, :, ::-1]), hwc2chw(_save_fake_recon_image[:, :, :, ::-1])), axis=3), in_range=(-120, 120))
#            # #     # collage1 = patchShow.patchShow(np.concatenate((hwc2chw(train_img1[:, :, :, ::-1]), hwc2chw(_save_upper_recon_image[:, :, :, ::-1])), axis=3), in_range=(-120, 120))
#            # #     # collage2 = patchShow.patchShow(np.concatenate((hwc2chw(train_img2[:, :, :, ::-1]), hwc2chw(_save_below_recon_image[:, :, :, ::-1])), axis=3), in_range=(-120, 120))

#            #     if not os.path.exists(self.cachefile_path):
#            #         os.makedirs(self.cachefile_path)
#            #     scipy.misc.imsave(os.path.join(self.cachefile_path, 'reconstructions_%s_%s_%d.png'%(FLAGS.encoder, FLAGS.feat, i)), collage)
#            # #     # scipy.misc.imsave(os.path.join(self.cachefile_path, 'below_reconstructions_%s_%s_%d.png'%(FLAGS.encoder, FLAGS.feat, i)), collage2)

            sess.run(self.update_global_step)

        coord.request_stop()
        coord.join(threads)

 
    def visualization_cycle(self, sess, visualize_path, visualization_set='test1', visualization_num=32):
        print "Start loading test to memory..."
        # load test1 and test2 to memory
        if visualization_set == 'train':
            train_index, train_class, train_pretrain_class, train_positive_norm_class_attr, train_negative_norm_class_attr, \
                train_image_feat, train_recon_encoder_image_feat, train_image = read_data(trainset_path, \
                    batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='train')

            train_class_semantic = train_positive_norm_class_attr

        elif visualization_set == 'test1':
            test1_index, test1_class, test1_pretrain_class, test1_positive_norm_class_attr, test1_negative_norm_class_attr, \
                test1_image_feat, test1_recon_encoder_image_feat, test1_image = read_data(self.seen_testset_path, \
                    batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')
            test1_class_semantic = test1_positive_norm_class_attr

        elif visualization_set == 'test2':
            test2_index, test2_class, test2_pretrain_class, test2_positive_norm_class_attr, test2_negative_norm_class_attr, \
                test2_image_feat, test2_recon_encoder_image_feat, test2_image = read_data(self.unseen_testset_path, \
                    batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')
            test2_class_semantic = test2_positive_norm_class_attr

        else:
            raise NotImplementedError

        if FLAGS.dataset == 'cub':
            all_fake_semantic = self.dataset_info['positive_norm_class_attr']
        else:
            all_fake_semantic = self.dataset_info['class_attr']

        # if FLAGS.semantic == 'class_attr':
        #     if FLAGS.negative_class_attr:
        #         train_class_semantic = train_negative_norm_class_attr
        #         test1_class_semantic = test1_negative_norm_class_attr
        #         test2_class_semantic = test2_negative_norm_class_attr
        #         all_fake_semantic = self.dataset_info['negative_norm_class_attr']
        #     else:
        #         train_class_semantic = train_positive_norm_class_attr
        #         test1_class_semantic = test1_positive_norm_class_attr
        #         test2_class_semantic = test2_positive_norm_class_attr
        #         if FLAGS.dataset == 'cub':
        #             all_fake_semantic = self.dataset_info['positive_norm_class_attr']
        #         else:
        #             all_fake_semantic = self.dataset_info['class_attr']
        # elif FLAGS.semantic == 'glove':
        #     train_class_semantic = train_class_glove
        #     test1_class_semantic = test1_class_glove
        #     test2_class_semantic = test2_class_glove
        #     all_fake_semantic = self.dataset_info['class_glove']
        # elif FLAGS.semantic == 'word2vec':
        #     train_class_semantic = train_class_word2vec
        #     test1_class_semantic = test1_class_word2vec
        #     test2_class_semantic = test2_class_word2vec
        #     all_fake_semantic = self.dataset_info['class_word2vec']
        # else:
        #     raise NotImplementedError

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}

        if visualization_set == 'train':
            train_img_list = []
            train_cl_list = []
            train_img_feat_list = []
            train_semantic_list = []
            train_pretrain_cl_list = []
            train_recon_encoder_img_feat_list = []
            for train_iter in tqdm(xrange(visualization_num)):
                [train_img, train_cl, train_img_feat, train_semantic, train_pretrain_cl, train_recon_encoder_img_feat] = \
                     sess.run([train_image, train_class, train_image_feat, train_class_semantic, train_pretrain_class, train_recon_encoder_image_feat])
                train_img_list.append(train_img)
                train_cl_list.append(train_cl)
                train_img_feat_list.append(train_img_feat)
                train_semantic_list.append(train_semantic)
                train_pretrain_cl_list.append(train_pretrain_cl)
                train_recon_encoder_img_feat_list.append(train_recon_encoder_img_feat)
            self.vis_img_list = np.array(train_img_list)
            self.vis_cl_list = np.array(train_cl_list)
            self.vis_img_feat_list = np.concatenate(train_img_feat_list, axis=0)
            self.vis_semantic_list = np.array(train_semantic_list)
            self.vis_pretrain_cl_list = np.array(train_pretrain_cl_list)
            self.vis_recon_encoder_img_feat_list = np.array(train_recon_encoder_img_feat_list)

        elif visualization_set == 'test1':
            test1_img_list = []
            test1_cl_list = []
            test1_img_feat_list = []
            test1_semantic_list = []
            test1_pretrain_cl_list = []
            test1_recon_encoder_img_feat_list = []
            for test1_iter in tqdm(xrange(visualization_num)):
                [test1_img, test1_cl, test1_img_feat, test1_semantic, test1_pretrain_cl, test1_recon_encoder_img_feat] = \
                    sess.run([test1_image, test1_class, test1_image_feat, test1_class_semantic, test1_pretrain_class, test1_recon_encoder_image_feat])
                test1_img_list.append(test1_img)
                test1_cl_list.append(test1_cl)
                test1_img_feat_list.append(test1_img_feat)
                test1_semantic_list.append(test1_semantic)
                test1_pretrain_cl_list.append(test1_pretrain_cl)
                test1_recon_encoder_img_feat_list.append(test1_recon_encoder_img_feat)
            self.vis_img_list = np.array(test1_img_list)
            self.vis_cl_list = np.array(test1_cl_list)
            self.vis_img_feat_list = np.concatenate(test1_img_feat_list, axis=0)
            self.vis_semantic_list = np.array(test1_semantic_list)
            self.vis_pretrain_cl_list = np.array(test1_pretrain_cl_list)
            self.vis_recon_encoder_img_feat_list = np.array(test1_recon_encoder_img_feat_list)

        elif visualization_set == 'test2':
            test2_img_list = []
            test2_cl_list = []
            test2_img_feat_list = []
            test2_semantic_list = []
            test2_pretrain_cl_list = []
            test2_recon_encoder_img_feat_list = []
            for test2_iter in tqdm(xrange(visualization_num)):
                [test2_img, test2_cl, test2_img_feat, test2_semantic, test2_pretrain_cl, test2_recon_encoder_img_feat] = \
                    sess.run([test2_image, test2_class, test2_image_feat, test2_class_semantic, test2_pretrain_class, test2_recon_encoder_image_feat])
                test2_img_list.append(test2_img)
                test2_cl_list.append(test2_cl)
                test2_img_feat_list.append(test2_img_feat)
                test2_semantic_list.append(test2_semantic)
                test2_pretrain_cl_list.append(test2_pretrain_cl)
                test2_recon_encoder_img_feat_list.append(test2_recon_encoder_img_feat)
            self.vis_img_list = np.array(test2_img_list)
            self.vis_cl_list = np.array(test2_cl_list)
            self.vis_img_feat_list = np.concatenate(test2_img_feat_list, axis=0)
            self.vis_semantic_list = np.array(test2_semantic_list)
            self.vis_pretrain_cl_list = np.array(test2_pretrain_cl_list)
            self.vis_recon_encoder_img_feat_list = np.array(test2_recon_encoder_img_feat_list)

        coord.request_stop()
        coord.join(threads)

        # try:
        #     ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        # except tf.errors.OutOfRangeError as e:
        #     tf.logging.error('Cannot restore checkpoint: %s', e)
        # self.saver.restore(sess, ckpt_state.model_checkpoint_path)

        i = 0
        visual_result = []
        for vis_cl, vis_img_feat, vis_semantic, vis_recon_encoder_img_feat in \
            tqdm(zip(self.vis_cl_list, self.vis_img_feat_list, self.vis_semantic_list, self.vis_recon_encoder_img_feat_list)):

            [_recon_img] = sess.run([self.recon_image],
                                        feed_dict={self.image_feat: vis_img_feat[None, :],
                                                   self.recon_image_feat: vis_recon_encoder_img_feat,
                                                   # self.input_image: train_img1,
                                                   # self.input_real_semantic: train_semantic1[:, None, :],
                                                   # self.input_fake_semantic: self.get_fake_semantic(all_fake_semantic, train_cl1),
                                                   # self.pretrain_label: train_pretrain_cl1,
                                                   self.keep_prob: 1.0})


            visual_result.append(np.concatenate([self.vis_img_list[i], _recon_img[0:1]], axis=0))
            i = i + 1
        visual_result = np.concatenate(visual_result, axis=0)

        collage = patchShow.patchShow((hwc2chw(visual_result[:, :, :, ::-1])), in_range=(-120, 120))
        
        if not os.path.exists(FLAGS.visual_path):
            os.makedirs(FLAGS.visual_path)
        scipy.misc.imsave(os.path.join(FLAGS.visual_path, 'visualization_%s_%s_%s_%d.png'%(FLAGS.dataset, visualization_set, visualize_path, i)), collage)


    def test1(self, sess, seen_testset_path, unseen_testset_path):

        test1_index, test1_class, test1_pretrain_class, test1_positive_norm_class_attr, test1_negative_norm_class_attr, \
            test1_image_feat, test1_comp_image_feat, test1_image = read_data(seen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        test2_index, test2_class, test2_pretrain_class, test2_positive_norm_class_attr, test2_negative_norm_class_attr, \
            test2_image_feat, test2_comp_image_feat, test2_image = read_data(unseen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        self.saver.restore(sess, ckpt_state.model_checkpoint_path)
        print "Restore parameters from checkpoint!"

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                test1_class_semantic = test1_negative_norm_class_attr
                test2_class_semantic = test2_negative_norm_class_attr
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                test1_class_semantic = test1_positive_norm_class_attr
                test2_class_semantic = test2_positive_norm_class_attr
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']

        elif FLAGS.semantic == 'glove':
            test1_class_semantic = test1_class_glove
            test2_class_semantic = test2_class_glove
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            test1_class_semantic = test1_class_word2vec
            test2_class_semantic = test2_class_word2vec
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}
        for test1_iter in tqdm(xrange(self.test1_total_iter)):
            [test1_cl, test1_img_feat, test1_semantic, test1_pretrain_cl] = sess.run([test1_class, test1_image_feat, test1_class_semantic, test1_pretrain_class])

            [_test1_upper_rank_semantic] = sess.run([self.upper_rank_semantic],
                        feed_dict={self.upper_image_feat: test1_img_feat,
                        self.upper_input_real_semantic: test1_semantic[:, None, :],
                        self.keep_prob: 1.0})

            if not int(test1_cl) in gzsl_test_seen_accuracy_list:
                gzsl_test_seen_accuracy_list[int(test1_cl)] = 0
            gzsl_test_seen_accuracy_list[int(test1_cl)] += np.equal(self.get_RANK(_test1_upper_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic), int(test1_cl))

        gzsl_test_seen_class_acc, gzsl_test_seen_total_acc = self.compute_class_accuracy(self.dataset_info['test_seen_class_num'], gzsl_test_seen_accuracy_list)    
        # print "Test1 Rank Loss: %.6f, Total GZSL Accuracy: %.6f" %(np.mean(test1_rank_loss), gzsl_test_seen_total_acc)
        print "Total GZSL Accuracy: %.6f" %(gzsl_test_seen_total_acc)

        for test2_iter in tqdm(xrange(self.test2_total_iter)):
            [test2_cl, test2_img_feat, test2_semantic, test2_pretrain_cl] = sess.run([test2_class, test2_image_feat, test2_class_semantic, test2_pretrain_class])

            [_test2_upper_rank_semantic] = \
                sess.run([self.upper_rank_semantic],
                        feed_dict={self.upper_image_feat: test2_img_feat,
                        self.upper_input_real_semantic: test2_semantic[:, None, :],
                        self.keep_prob: 1.0})

            if FLAGS.zsl_mode in ['zsl', 'both']:
                if not int(test2_cl) in zsl_test_unseen_accuracy_list:
                    zsl_test_unseen_accuracy_list[int(test2_cl)] = 0
                zsl_test_unseen_accuracy_list[int(test2_cl)] += np.equal(self.get_RANK(_test2_upper_rank_semantic, self.unseen_test_mask, FLAGS.semantic), int(test2_cl))

            if FLAGS.zsl_mode in ['gzsl', 'both']:
                if not int(test2_cl) in gzsl_test_unseen_accuracy_list:
                    gzsl_test_unseen_accuracy_list[int(test2_cl)] = 0
                gzsl_test_unseen_accuracy_list[int(test2_cl)] += np.equal(self.get_RANK(_test2_upper_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic), int(test2_cl))

        if FLAGS.zsl_mode in ['zsl', 'both']:
            zsl_test_unseen_class_acc, zsl_test_unseen_total_acc = self.compute_class_accuracy(self.dataset_info['test_unseen_class_num'], zsl_test_unseen_accuracy_list)    

        if FLAGS.zsl_mode in ['gzsl', 'both']:
            gzsl_test_unseen_class_acc, gzsl_test_unseen_total_acc = self.compute_class_accuracy(self.dataset_info['test_unseen_class_num'], gzsl_test_unseen_accuracy_list)    

        gzsl_H = (2*gzsl_test_unseen_total_acc*gzsl_test_seen_total_acc)/(gzsl_test_unseen_total_acc+gzsl_test_seen_total_acc)
        # print "Test2 Rank Loss: %.6f, Total ZSL Accuracy: %.6f, Total GZSL Accuracy: %.6f" %(np.mean(test2_rank_loss), zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc)
        print "Total ZSL Accuracy: %.6f, Total GZSL Accuracy: %.6f, GZSL H: %.6f" %(zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc, gzsl_H)

        coord.request_stop()
        coord.join(threads)

    def test_combine(self, sess):
    
        print "Start loading test to memory..."
        # load test1 and test2 to memory
        test1_index, test1_class, test1_pretrain_class, test1_positive_norm_class_attr, test1_negative_norm_class_attr, \
            test1_image_feat, test1_comp_image_feat, test1_image = read_data(self.seen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        test2_index, test2_class, test2_pretrain_class, test2_positive_norm_class_attr, test2_negative_norm_class_attr, \
            test2_image_feat, test2_comp_image_feat, test2_image = read_data(self.unseen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                test1_class_semantic = test1_negative_norm_class_attr
                test2_class_semantic = test2_negative_norm_class_attr
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                test1_class_semantic = test1_positive_norm_class_attr
                test2_class_semantic = test2_positive_norm_class_attr
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            test1_class_semantic = test1_class_glove
            test2_class_semantic = test2_class_glove
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            test1_class_semantic = test1_class_word2vec
            test2_class_semantic = test2_class_word2vec
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}

        test1_cl_list = []
        test1_img_feat_list = []
        test1_semantic_list = []
        test1_pretrain_cl_list = []
        for test1_iter in tqdm(xrange(self.test1_total_iter)):
            [test1_cl, test1_img_feat, test1_semantic, test1_pretrain_cl] = sess.run([test1_class, test1_image_feat, test1_class_semantic, test1_pretrain_class])
            test1_cl_list.append(test1_cl)
            test1_img_feat_list.append(test1_img_feat)
            test1_semantic_list.append(test1_semantic)
            test1_pretrain_cl_list.append(test1_pretrain_cl)
        self.test1_cl_list = np.array(test1_cl_list)
        self.test1_img_feat_list = np.concatenate(test1_img_feat_list, axis=0)
        self.test1_semantic_list = np.array(test1_semantic_list)
        self.test1_pretrain_cl_list = np.array(test1_pretrain_cl_list)

        test2_cl_list = []
        test2_img_feat_list = []
        test2_semantic_list = []
        test2_pretrain_cl_list = []
        for test2_iter in tqdm(xrange(self.test2_total_iter)):
            [test2_cl, test2_img_feat, test2_semantic, test2_pretrain_cl] = sess.run([test2_class, test2_image_feat, test2_class_semantic, test2_pretrain_class])
            test2_cl_list.append(test2_cl)
            test2_img_feat_list.append(test2_img_feat)
            test2_semantic_list.append(test2_semantic)
            test2_pretrain_cl_list.append(test2_pretrain_cl)
        self.test2_cl_list = np.array(test2_cl_list)
        self.test2_img_feat_list = np.concatenate(test2_img_feat_list, axis=0)
        self.test2_semantic_list = np.array(test2_semantic_list)
        self.test2_pretrain_cl_list = np.array(test2_pretrain_cl_list)

        coord.request_stop()
        coord.join(threads)

        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        self.saver.restore(sess, ckpt_state.model_checkpoint_path)
        print "Restore parameters from checkpoint!"
 
        for cycle_weight in [0.001, 0.01, 0.1, 1, 10, 100]:
            print 'Cycle weight: %f' %(cycle_weight)
            zsl_test_unseen_total_acc = -1
            gzsl_test_unseen_total_acc = -1
            gzsl_test_seen_total_acc = -1

            [_test1_fake_rank_semantic, _test1_rank_semantic] = sess.run([self.fake_rank_semantic, self.rank_semantic],
                        feed_dict={self.image_feat: self.test1_img_feat_list,
                        self.input_real_semantic: self.test1_semantic_list,
                        self.keep_prob: 1.0})


            # cycle_weight = FLAGS.cycle_weight
            # gzsl_test_seen_pred_label= self.get_RANK(_test1_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
            gzsl_test_seen_pred_label= self.get_RANK_combine(_test1_rank_semantic, _test1_fake_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic, cycle_weight)
            gzsl_test_seen_total_acc = self.compute_class_accuracy_total(self.test1_cl_list, gzsl_test_seen_pred_label, self.trainval_mask)

            print "Total GZSL Accuracy: %.6f" %(gzsl_test_seen_total_acc)

            [_test2_fake_rank_semantic, _test2_rank_semantic] = sess.run([self.fake_rank_semantic, self.rank_semantic],
                        feed_dict={self.image_feat: self.test2_img_feat_list,
                        self.input_real_semantic: self.test2_semantic_list,
                        self.keep_prob: 1.0})


            # zsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, self.unseen_test_mask, FLAGS.semantic)
            zsl_test_unseen_pred_label = self.get_RANK_combine(_test2_rank_semantic, _test2_fake_rank_semantic, self.unseen_test_mask, FLAGS.semantic, cycle_weight)
            zsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list, zsl_test_unseen_pred_label, self.unseen_test_mask) 

            # gzsl_test_unseen_pred_label = self.get_RANK(_test2_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
            gzsl_test_unseen_pred_label = self.get_RANK_combine(_test2_rank_semantic, _test2_fake_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic, cycle_weight)
            gzsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list, gzsl_test_unseen_pred_label, self.unseen_test_mask)


            gzsl_H = (2*gzsl_test_unseen_total_acc*gzsl_test_seen_total_acc)/(gzsl_test_unseen_total_acc+gzsl_test_seen_total_acc+1e-8)
            print "Total ZSL Accuracy: %.6f, Total GZSL Accuracy: %.6f, GZSL H: %.6f" %(zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc, gzsl_H)

    def test2(self, sess):
    
        print "Start loading test to memory..."
        # load test1 and test2 to memory
        test1_index, test1_class, test1_pretrain_class, test1_positive_norm_class_attr, test1_negative_norm_class_attr, \
            test1_image_feat, test1_comp_image_feat, test1_image = read_data(self.seen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        test2_index, test2_class, test2_pretrain_class, test2_positive_norm_class_attr, test2_negative_norm_class_attr, \
            test2_image_feat, test2_comp_image_feat, test2_image = read_data(self.unseen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                test1_class_semantic = test1_negative_norm_class_attr
                test2_class_semantic = test2_negative_norm_class_attr
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                test1_class_semantic = test1_positive_norm_class_attr
                test2_class_semantic = test2_positive_norm_class_attr
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            test1_class_semantic = test1_class_glove
            test2_class_semantic = test2_class_glove
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            test1_class_semantic = test1_class_word2vec
            test2_class_semantic = test2_class_word2vec
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}

        test1_cl_list = []
        test1_img_feat_list = []
        test1_semantic_list = []
        test1_pretrain_cl_list = []
        for test1_iter in tqdm(xrange(self.test1_total_iter)):
            [test1_cl, test1_img_feat, test1_semantic, test1_pretrain_cl] = sess.run([test1_class, test1_image_feat, test1_class_semantic, test1_pretrain_class])
            test1_cl_list.append(test1_cl)
            test1_img_feat_list.append(test1_img_feat)
            test1_semantic_list.append(test1_semantic)
            test1_pretrain_cl_list.append(test1_pretrain_cl)
        self.test1_cl_list = np.array(test1_cl_list)
        self.test1_img_feat_list = np.concatenate(test1_img_feat_list, axis=0)
        self.test1_semantic_list = np.array(test1_semantic_list)
        self.test1_pretrain_cl_list = np.array(test1_pretrain_cl_list)

        test2_cl_list = []
        test2_img_feat_list = []
        test2_semantic_list = []
        test2_pretrain_cl_list = []
        for test2_iter in tqdm(xrange(self.test2_total_iter)):
            [test2_cl, test2_img_feat, test2_semantic, test2_pretrain_cl] = sess.run([test2_class, test2_image_feat, test2_class_semantic, test2_pretrain_class])
            test2_cl_list.append(test2_cl)
            test2_img_feat_list.append(test2_img_feat)
            test2_semantic_list.append(test2_semantic)
            test2_pretrain_cl_list.append(test2_pretrain_cl)
        self.test2_cl_list = np.array(test2_cl_list)
        self.test2_img_feat_list = np.concatenate(test2_img_feat_list, axis=0)
        self.test2_semantic_list = np.array(test2_semantic_list)
        self.test2_pretrain_cl_list = np.array(test2_pretrain_cl_list)

        coord.request_stop()
        coord.join(threads)

        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        self.saver.restore(sess, ckpt_state.model_checkpoint_path)
        print "Restore parameters from checkpoint!"

        [_test1_upper_rank_semantic] = sess.run([self.upper_rank_semantic],
                    feed_dict={self.upper_image_feat: self.test1_img_feat_list,
                    self.upper_input_real_semantic: self.test1_semantic_list,
                    self.keep_prob: 1.0})



        gzsl_test_seen_pred_label= self.get_RANK(_test1_upper_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
        gzsl_test_seen_total_acc = self.compute_class_accuracy_total(self.test1_cl_list, gzsl_test_seen_pred_label, self.trainval_mask)

        print "Total GZSL Accuracy: %.6f" %(gzsl_test_seen_total_acc)

        [_test2_upper_rank_semantic] = \
            sess.run([self.upper_rank_semantic],
                    feed_dict={self.upper_image_feat: self.test2_img_feat_list,
                    self.upper_input_real_semantic: self.test2_semantic_list,
                    self.keep_prob: 1.0})

        zsl_test_unseen_pred_label = self.get_RANK(_test2_upper_rank_semantic, self.unseen_test_mask, FLAGS.semantic)
        zsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list, zsl_test_unseen_pred_label, self.unseen_test_mask) 

        gzsl_test_unseen_pred_label = self.get_RANK(_test2_upper_rank_semantic, np.arange(1, self.num_classes+1), FLAGS.semantic)
        gzsl_test_unseen_total_acc = self.compute_class_accuracy_total(self.test2_cl_list, gzsl_test_unseen_pred_label, self.unseen_test_mask)


        gzsl_H = (2*gzsl_test_unseen_total_acc*gzsl_test_seen_total_acc)/(gzsl_test_unseen_total_acc+gzsl_test_seen_total_acc+1e-8)
        print "Total ZSL Accuracy: %.6f, Total GZSL Accuracy: %.6f, GZSL H: %.6f" %(zsl_test_unseen_total_acc, gzsl_test_unseen_total_acc, gzsl_H)



    def visualization(self, sess, train_total_iter, trainset_path, seen_testset_path, unseen_testset_path, visualize_path):
        self.train_total_iter = train_total_iter
        print "Start loading test to memory..."
        # load test1 and test2 to memory
        train_index, train_class, train_pretrain_class, train_positive_norm_class_attr, train_negative_norm_class_attr, \
            train_image_feat, train_comp_image_feat, train_image = read_data(trainset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='train')

        test1_index, test1_class, test1_pretrain_class, test1_positive_norm_class_attr, test1_negative_norm_class_attr, \
            test1_image_feat, test1_comp_image_feat, test1_image = read_data(self.seen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        test2_index, test2_class, test2_pretrain_class, test2_positive_norm_class_attr, test2_negative_norm_class_attr, \
            test2_image_feat, test2_comp_image_feat, test2_image = read_data(self.unseen_testset_path, \
                batch_size=FLAGS.test_batch_size, net_inputsize=self.net_inputsize, mode='test')

        if FLAGS.semantic == 'class_attr':
            if FLAGS.negative_class_attr:
                train_class_semantic = train_negative_norm_class_attr
                test1_class_semantic = test1_negative_norm_class_attr
                test2_class_semantic = test2_negative_norm_class_attr
                all_fake_semantic = self.dataset_info['negative_norm_class_attr']
            else:
                train_class_semantic = train_positive_norm_class_attr
                test1_class_semantic = test1_positive_norm_class_attr
                test2_class_semantic = test2_positive_norm_class_attr
                if FLAGS.dataset == 'cub':
                    all_fake_semantic = self.dataset_info['positive_norm_class_attr']
                else:
                    all_fake_semantic = self.dataset_info['class_attr']
        elif FLAGS.semantic == 'glove':
            train_class_semantic = train_class_glove
            test1_class_semantic = test1_class_glove
            test2_class_semantic = test2_class_glove
            all_fake_semantic = self.dataset_info['class_glove']
        elif FLAGS.semantic == 'word2vec':
            train_class_semantic = train_class_word2vec
            test1_class_semantic = test1_class_word2vec
            test2_class_semantic = test2_class_word2vec
            all_fake_semantic = self.dataset_info['class_word2vec']
        else:
            raise NotImplementedError

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        zsl_test_unseen_accuracy_list = {}
        gzsl_test_seen_accuracy_list = {}
        gzsl_test_unseen_accuracy_list = {}

        train_img_list = []
        train_cl_list = []
        train_img_feat_list = []
        train_semantic_list = []
        train_pretrain_cl_list = []
        for train_iter in tqdm(xrange(self.train_total_iter)):
            [train_img, train_cl, train_img_feat, train_semantic, train_pretrain_cl] = sess.run([train_image, train_class, train_image_feat, train_class_semantic, train_pretrain_class])
            train_img_list.append(train_img)
            train_cl_list.append(train_cl)
            train_img_feat_list.append(train_img_feat)
            train_semantic_list.append(train_semantic)
            train_pretrain_cl_list.append(train_pretrain_cl)
        self.train_cl_list = np.array(train_cl_list)
        self.train_img_feat_list = np.concatenate(train_img_feat_list, axis=0)
        self.train_semantic_list = np.array(train_semantic_list)
        self.train_pretrain_cl_list = np.array(train_pretrain_cl_list)

        # test1_cl_list = []
        # test1_img_feat_list = []
        # test1_semantic_list = []
        # test1_pretrain_cl_list = []
        # for test1_iter in tqdm(xrange(self.test1_total_iter)):
        #     [test1_cl, test1_img_feat, test1_semantic, test1_pretrain_cl] = sess.run([test1_class, test1_image_feat, test1_class_semantic, test1_pretrain_class])
        #     test1_cl_list.append(test1_cl)
        #     test1_img_feat_list.append(test1_img_feat)
        #     test1_semantic_list.append(test1_semantic)
        #     test1_pretrain_cl_list.append(test1_pretrain_cl)
        # self.test1_cl_list = np.array(test1_cl_list)
        # self.test1_img_feat_list = np.concatenate(test1_img_feat_list, axis=0)
        # self.test1_semantic_list = np.array(test1_semantic_list)
        # self.test1_pretrain_cl_list = np.array(test1_pretrain_cl_list)

        # test2_cl_list = []
        # test2_img_feat_list = []
        # test2_semantic_list = []
        # test2_pretrain_cl_list = []
        # for test2_iter in tqdm(xrange(self.test2_total_iter)):
        #     [test2_cl, test2_img_feat, test2_semantic, test2_pretrain_cl] = sess.run([test2_class, test2_image_feat, test2_class_semantic, test2_pretrain_class])
        #     test2_cl_list.append(test2_cl)
        #     test2_img_feat_list.append(test2_img_feat)
        #     test2_semantic_list.append(test2_semantic)
        #     test2_pretrain_cl_list.append(test2_pretrain_cl)
        # self.test2_cl_list = np.array(test2_cl_list)
        # self.test2_img_feat_list = np.concatenate(test2_img_feat_list, axis=0)
        # self.test2_semantic_list = np.array(test2_semantic_list)
        # self.test2_pretrain_cl_list = np.array(test2_pretrain_cl_list)

        coord.request_stop()
        coord.join(threads)

        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.test_checkpoint)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        self.saver.restore(sess, ckpt_state.model_checkpoint_path)
        print "Restore parameters from checkpoint!"

        i = 0
        visual_result = []
        for vis_train_cl, vis_train_img_feat, vis_train_semantic in \
            tqdm(zip(self.train_cl_list, self.train_img_feat_list, self.train_semantic_list)):

            sort_class_by_l2 = np.argsort(np.sum((vis_train_semantic - all_fake_semantic)**2, axis=1)) + 1
            select_five_class = np.concatenate([vis_train_cl, sort_class_by_l2[-4:]])
            five_false_semantic = all_fake_semantic[select_five_class]

            # self.upper_map_semantic = self.upper_map_recon_semantic * self.upper_map_rank_semantic
            # self.below_map_semantic = self.below_map_recon_semantic * self.below_map_rank_semantic

            ## debug
            [_upper_recon_img, _below_recon_img, _upper_map_semantic, _upper_map_recon_semantic, _upper_map_rank_semantic, \
                _below_map_semantic, _below_map_recon_semantic, _below_map_rank_semantic, _upper_recon_img_feat, _below_recon_img_feat, _upper_rank_semantic] \
                    = sess.run([self.upper_recon_image, self.below_recon_image, self.upper_map_semantic, self.upper_map_recon_semantic, \
                        self.upper_map_rank_semantic, self.below_map_semantic, self.below_map_recon_semantic, self.below_map_rank_semantic, \
                            self.upper_recon_image_feat, self.below_recon_image_feat, self.upper_rank_semantic], 
                    feed_dict={self.upper_image_feat: np.tile(vis_train_img_feat[None, :], (5, 1)),
                               self.upper_input_real_semantic: np.tile(vis_train_semantic[None, :], (5, 1, 1)),
                               self.below_image_feat: np.tile(vis_train_img_feat[None, :], (5, 1)),
                               self.below_input_real_semantic: five_false_semantic*1, 
                               self.keep_prob: 1.0})

            print "_upper_rank_semantic", _upper_rank_semantic[0, :10]
            print "_upper_map_rank_semantic", _upper_map_rank_semantic[0, :10]
            print "_upper_map_recon_semantic", _upper_map_recon_semantic[0, :10]
            print "_upper_map_semantic", _upper_map_semantic[0, :10]
            print "_upper_recon_img_feat", _upper_recon_img_feat[0, :10]
            print "_upper_recon_img", _upper_recon_img[0, :5, :5, 0]
            # print "_below_rank_semantic", _below_rank_semantic[0][:10]
            print "_below_map_rank_semantic", _below_map_rank_semantic[0, :10]
            print "_below_map_recon_semantic", _below_map_recon_semantic[0, :10]
            print "_below_map_semantic", _below_map_semantic[0, :10]
            print "_below_recon_img_feat", _below_recon_img_feat[0, :10]
            print "_below_recon_img", _below_recon_img[0, :5, :5, 0]

            # [_upper_recon_img, _below_recon_img] = sess.run([self.upper_recon_image, self.below_recon_image], 
            #         feed_dict={self.upper_image_feat: np.tile(vis_train_img_feat[None, :], (5, 1)),
            #                    self.upper_input_real_semantic: np.tile(vis_train_semantic[None, :], (5, 1, 1)),
            #                    self.below_image_feat: np.tile(vis_train_img_feat[None, :], (5, 1)),
            #                    self.below_input_real_semantic: five_false_semantic, 
            #                    self.keep_prob: 1.0})

            visual_result.append(np.concatenate([train_img_list[i], _upper_recon_img[0:1], _below_recon_img/3.0], axis=0))
            i = i + 1
        visual_result = np.concatenate(visual_result, axis=0)

        collage = patchShow.patchShow((hwc2chw(visual_result[:, :, :, ::-1])), rows=i, cols=7, in_range=(-120, 120))
        
        if not os.path.exists(FLAGS.visual_path):
            os.makedirs(FLAGS.visual_path)
        scipy.misc.imsave(os.path.join(FLAGS.visual_path, 'visualization_%s_%d.png'%(visualize_path, i)), collage)
