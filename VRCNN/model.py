import utils
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import train_data
import struct
import yuv_data
import quantization

class VRCNN(object):
    def __init__(self,
                 QP,
                 blu,
                 sub_image_size=35,
                 sub_label_size=35,
                 batch_size=64,
                 c_dim=1,
                 lr=0.00001,
    ):
        self.image_size = sub_image_size
        self.label_size = sub_label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.lr=lr
        self.build_recon_model(blu,QP)

    def build_recon_model(self,blu,QP):
        self.images = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
        self.images_norm=self.images/255-128/255#tf.divide(self.images,255)
        self.labels_norm=self.labels/255-128/255#tf.divide(self.labels,255)
        self.weights = {
            'w1': tf.get_variable('w1',[5, 5, 1, 64], initializer=variance_scaling_initializer()),#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/25))),
            'w2_1': tf.get_variable('w2_1',[3, 3, 64, 32],initializer=variance_scaling_initializer()),#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64))),
            'w2_2': tf.get_variable('w2_2',[5, 5, 64, 16], initializer=variance_scaling_initializer()),#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/25/64))),
            'w3_1': tf.get_variable('w3_1',[3, 3, 48, 16],initializer=variance_scaling_initializer() ),#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/48))),
            'w3_2': tf.get_variable('w3_2',[1, 1, 48, 32], initializer=variance_scaling_initializer()),#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/1/48))),
            'w4': tf.get_variable('w4',[3, 3, 48, 1], initializer=variance_scaling_initializer())#initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/25/48)))
        }
        self.biases = {
            'b1': tf.get_variable('b1',[64],initializer=tf.constant_initializer(0)),
            'b2_1': tf.get_variable('b2_1',[32], initializer=tf.constant_initializer(0)),
            'b2_2': tf.get_variable('b2_2', [16], initializer=tf.constant_initializer(0)),
            'b3_1': tf.get_variable('b3_1',[16], initializer=tf.constant_initializer(0)),
            'b3_2': tf.get_variable('b3_2', [32], initializer=tf.constant_initializer(0)),
            'b4': tf.get_variable('b3', [1], initializer=tf.constant_initializer(0))
        }
        stepw, self.blu_ub, ratio = quantization.loadQpara(QP)
        self.stepw_list=stepw
        self.stepw = {'w1': stepw[0], 'w2_1': stepw[1], 'w2_2': stepw[2], 'w3_1': stepw[3], 'w3_2': stepw[4],
                      'w4': stepw[5]}
        self.stepw_b = {'b1': stepw[0], 'b2_1': stepw[1], 'b2_2': stepw[2], 'b3_1': stepw[3], 'b3_2': stepw[4],
                        'b4': stepw[5]}
        self.ratio = {'b1': ratio[0], 'b2_1': ratio[1], 'b2_2': ratio[2], 'b3_1': ratio[3], 'b3_2': ratio[4],
                      'b4': ratio[5]}
        if blu:
            self.residual=self.model_blu()
        else:
            self.residual=self.model()
        self.pred = tf.add(self.residual,self.images_norm,name='pred')

        # Loss function (MSE)
        self.loss = tf.nn.l2_loss(tf.subtract(self.labels_norm, self.pred))#tf.reduce_mean(tf.square(self.labels - self.pred))
        self.gloal_step=tf.Variable(0,trainable=False,name='global_step')
        tf.summary.scalar("loss", self.loss)

        mse = tf.reduce_mean(tf.squared_difference(self.labels, self.pred * 255))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * utils.log10(PSNR)
        tf.summary.scalar("PSNR", PSNR)
        tf.summary.image("input_image", tf.cast(self.images * 255, tf.uint8))
        tf.summary.image("target_image", tf.cast(self.labels * 255, tf.uint8))
        tf.summary.image("output_image", tf.cast(self.pred * 255, tf.uint8))
        self.saver=tf.train.Saver(max_to_keep=30)

    def find_sigma(self,fe):
        fe_flat=tf.layers.flatten(fe)
        fe_sort=tf.contrib.framework.sort(fe_flat,direction='ASCENDING')
        shape=tf.shape(fe_sort)
        sigmas=fe_sort[:,258211]
        return tf.reduce_mean(sigmas)
    def model(self):
        print("relu model")
        self.conv1_u=tf.nn.conv2d(self.images_norm, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME')+self.biases['b1']
        self.sigma1=self.find_sigma(self.conv1_u)
        self.conv1 = tf.nn.relu(self.conv1_u,name='conv1')
        self.conv2_1_u=tf.nn.conv2d(self.conv1, self.weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')+self.biases['b2_1']
        self.conv2_1 = tf.nn.relu(self.conv2_1_u,name='conv2_1')
        self.conv2_2_u=tf.nn.conv2d(self.conv1, self.weights['w2_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b2_2']
        self.conv2_2 = tf.nn.relu(self.conv2_2_u,name='conv2_2')
        self.conv2=tf.concat([self.conv2_1,self.conv2_2],3,name='conv2')
        self.conv3_1_u=tf.nn.conv2d(self.conv2, self.weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b3_1']
        self.conv3_1 = tf.nn.relu(self.conv3_1_u,name='conv3_1')
        self.conv3_2_u=tf.nn.conv2d(self.conv2, self.weights['w3_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b3_2']
        self.conv3_2 = tf.nn.relu(self.conv3_2_u,name='conv3_2')
        self.conv3=tf.concat([self.conv3_1,self.conv3_2],3,name='conv3')
        self.conv4 = tf.add(tf.nn.conv2d(self.conv3, self.weights['w4'], strides=[1, 1, 1, 1], padding='SAME'), self.biases['b4'],name='conv4')
        return self.conv4

    def model_blu(self):
        print("blu model")
        self.conv1 = tf.nn.conv2d(self.images_norm, self.weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b1']
        self.blu1 = tf.clip_by_value(self.conv1, -self.blu_ub[0]/2, self.blu_ub[0], name='blu1')
        self.conv2_1 = tf.nn.conv2d(self.blu1, self.weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b2_1']
        self.blu2_1=tf.clip_by_value(self.conv2_1,-self.blu_ub[1]/2,self.blu_ub[1],name='blu2_1')
        self.conv2_2 = tf.nn.conv2d(self.blu1, self.weights['w2_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b2_2']
        self.blu2_2=tf.clip_by_value(self.conv2_2,-self.blu_ub[2]/2,self.blu_ub[2],name='blu2_2')
        self.blu2 = tf.concat([self.blu2_1, self.blu2_2], 3, name='conv2')
        self.conv3_1 = tf.nn.conv2d(self.blu2, self.weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b3_1']
        self.blu3_1=tf.clip_by_value(self.conv3_1,-self.blu_ub[3]/2,self.blu_ub[3],name='blu3_1')
        self.conv3_2 = tf.nn.conv2d(self.blu2, self.weights['w3_2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases['b3_2']
        self.blu3_2=tf.clip_by_value(self.conv3_2,-self.blu_ub[4]/2,self.blu_ub[4],name='blu3_2')
        self.blu3 = tf.concat([self.blu3_1, self.blu3_2], 3, name='blu3')
        self.conv4 = tf.add(tf.nn.conv2d(self.blu3, self.weights['w4'], strides=[1, 1, 1, 1], padding='SAME'),
                            self.biases['b4'], name='conv4')
        return self.conv4

    def train(self, config):
        data = train_data.train_data(config.QP,config.batch_size)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op=optimizer.minimize(self.loss, global_step=self.gloal_step)
        self.merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("log/train", tf.get_default_graph())
        start_time = time.time()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            if config.continue_train:
                if self.load(sess,config.ppro_dir+str(config.QP)):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                    exit(1)
            print("Training...")
            step=0
            #fetch_list=[self.train_op, self.loss,self.gloal_step]
            fetch_list=[self.train_op, self.loss,self.gloal_step,self.conv1,self.conv2_1,self.conv2_2,self.conv3_1,self.conv3_2,self.conv4]
            #fetch_list = [self.train_op, self.loss, self.gloal_step, self.conv4]
            for ep in range(config.epoch):
                for idx in range(0, data.pieces//config.batch_size):
                    batch=data.get_batch(config.batch_size)
                    receive_list = sess.run(fetch_list,feed_dict={self.images: batch[1], self.labels: batch[0]})
                    _, err ,step=receive_list[0:3]
                    del receive_list[0:3]
                    with open("feature_map.data","wb") as f:
                        for item in receive_list:
                            f.write(item)
                    exit(0)
                    if step % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], lr:[%.6f],train_loss: [%.8f]"
                          % ((ep + 1), step, time.time() - start_time, self.lr,err))
                        summary_info=sess.run(self.merged_summary,feed_dict={self.images: batch[1], self.labels: batch[0]})
                        train_writer.add_summary(summary_info, step)
                #self.quantize(sess,config.QP)
                self.saver.save(sess,config.ppro_dir+str(config.QP)+'/VRCNN.model',global_step=step)
    def quantize(self,sess,QP):
        wf=[]
        for w in self.weights:
            wf.append(sess.run(self.weights[w]))
        quant_params = quantization.quantNsave(wf,QP)
        return quant_params
    def quantize1(self,config):
        with tf.Session() as sess:
            quantization.quant_blu_save(config.QP)
            #self.load(sess,config.ppro_dir+str(config.QP))
            #self.quantize(sess,config.QP)
    def quant_w(self,sess):
        wf={}
        wq={}
        update=[]
        for w in self.weights:
            wf[w] = sess.run(self.weights[w])
            wq[w] = np.clip(np.around(wf[w] / self.stepw[w]), -128, 127) * self.stepw[w]
            update.append(tf.assign(self.weights[w], wq[w]))
        sess.run(update)
    def quant_finetune(self,config):
        wf={}#training weights
        wq={}#quantized weights
        we={}#weights gradients
        wn={}#updated weights
        bf={}#training biases
        bq={}#quantized biases
        be={}#biases gradients
        bn={}#updated biases
        feeddict={}
        update=[]
        start_time = time.time()
        with tf.Session() as sess:
            data = train_data.train_data(config.QP, config.batch_size)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op=optimizer.minimize(self.loss, global_step=self.gloal_step)
            tf.global_variables_initializer().run()
            # load ppro model
            self.load(sess, config.ppro_dir+str(config.QP))
            load_step=sess.run(self.gloal_step)
            #initialize quant weights
            for w in self.weights:
                wf[w] = sess.run(self.weights[w])
                wq[w]=np.clip(np.around(wf[w]/self.stepw[w]),-128,127)*self.stepw[w]
                update.append(tf.assign(self.weights[w],wq[w]))
            #for b in self.biases:
            #    bf[b] = sess.run(self.biases[b])
            #    bq[b] = np.around(bf[b]*self.ratio[b]/self.stepw_b[b])*self.stepw_b[b]/self.ratio[b]
            #    update.append(tf.assign(self.biases[b],bq[b]))
            # finetune
            #fetch_list = [self.train_op, self.loss, self.gloal_step, self.sigma1]
            fetch_list = [self.train_op, self.loss, self.gloal_step]
            #sigma_history = []
            print(data.pieces//config.batch_size,'iters in total')
            for i in range(0,data.pieces//config.batch_size):
                batch = data.get_batch(config.batch_size)
                feeddict[self.labels]=batch[0]
                feeddict[self.images]=batch[1]
                sess.run(update)
                receive_list = sess.run(fetch_list,feed_dict=feeddict)
                _, err, step = receive_list[0:3]
                #del receive_list[0:3]
                #sigma_history.append(receive_list)
                #print(receive_list)
                if step%10==0:
                    print("Finetune: step: [%2d], local_step: [%2d], time: [%4.4f], train_loss: [%.8f]"%(step,step-load_step,time.time()-start_time, err))
                for w in wq:
                    wn[w] = sess.run(self.weights[w])
                    we[w]=np.subtract(wn[w],wq[w])
                    wf[w]=np.clip(np.add(wf[w],we[w]),-128*self.stepw[w],127*self.stepw[w])
                    wq[w]=np.around(wf[w]/self.stepw[w])*self.stepw[w]
                #for b in bq:
                #    bn[b] = sess.run(self.biases[b])
                #    be[b]=np.subtract(bn[b],bq[b])
                #    bf[b]=np.add(bf[b],be[b])
                #    bq[b]=np.around(bf[b]*self.ratio[b]/self.stepw_b[b])*self.stepw_b[b]/self.ratio[b]
            sess.run(update)
            self.saver.save(sess, config.blu_dir + str(config.QP) + '/VRCNN_dblu.model', global_step=step)
            #with open("feature_sigma%d.data" % config.QP, "wb") as f:
            #    for item in sigma_history:
            #        f.write(struct.pack('%sf' % len(item), *item))
        return 0

    def divided_run(self,sess,input,output):
        #输入是一张图片
        # 默认参数是最大卷积核尺寸，这里取21
        pad=10
        #3000以下尺寸都可以分为四块处理
        height=input.shape[1]
        width=input.shape[2]
        subheight=height//2
        subwidth=width//2
        subinput1 = input[:, 0:subheight + pad, 0:subwidth + pad, :]
        subinput2 = input[:, subheight - pad:height, 0:subwidth + pad, :]
        subinput3 = input[:, 0:subheight + pad, subwidth - pad:width, :]
        subinput4 = input[:, subheight - pad:height, subwidth - pad:width, :]
        #print(subinput1.shape,subinput2.shape,subinput3.shape,subinput4.shape)
        suboutput1=sess.run(output,feed_dict={self.images:subinput1})
        suboutput2=sess.run(output,feed_dict={self.images:subinput2})
        suboutput3=sess.run(output,feed_dict={self.images:subinput3})
        suboutput4=sess.run(output,feed_dict={self.images:subinput4})
        output_row1=np.concatenate((suboutput1[:,0:subheight,0:subwidth,:],suboutput2[:,pad:subheight+pad,0:subwidth,:]),1)
        output_row2=np.concatenate((suboutput3[:,0:subheight,pad:subwidth+pad,:],suboutput4[:,pad:subheight+pad,pad:subwidth+pad,:]),1)
        return np.concatenate((output_row1,output_row2),2)

    def test(self, config):
        test_data = yuv_data.YuvData(config.QP)
        with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
            if config.blu:
                if self.load(sess, config.blu_dir+str(config.QP)):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                    exit(1)
            else:
                if self.load(sess, config.ppro_dir+str(config.QP)):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed...")
                    exit(1)
            #self.quant_w(sess)
            for j in range(len(test_data.testData)):
                ori = test_data.testData[j][1]
                rec = test_data.testData[j][2]
                output = np.empty((1, ori.shape[1], ori.shape[2]))
                start_time = time.time()
                for i in range(ori.shape[0]):
                    input=rec[i].reshape(1,ori.shape[1],ori.shape[2],1)
                    if rec[i].shape[0]>1500 or rec[i].shape[1]>1500:
                        #print("memory limited, divided running")
                        pred_image = self.divided_run(sess,input,self.pred)
                    else:
                        pred_image = sess.run(self.pred, feed_dict={self.images: input})
                    pred_image = pred_image * 255+128
                    output=np.concatenate((output,pred_image.reshape(1,ori.shape[1],ori.shape[2])),0)
                run_time=time.time() - start_time
                output = np.delete(output, 0, 0)
                result = np.clip(output, 0, 255)
                psnr_rec = utils.psnr(rec, ori)
                psnr_net = utils.psnr(result, ori)
                with open("psnr.data","ab") as f:
                    f.write(struct.pack("<d",psnr_net))
                print(test_data.testData[j][0],test_data.testData[j][1].shape,'time:%.3f' % run_time)
                print("PSNR: before net %.3f\tafter net %.3f" % (psnr_rec, psnr_net))

    def save(self, sess, checkpoint_dir, step):
        model_name = "VRCNN.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print('ckpt'+ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    
    def dump(self,config):
        with tf.Session() as sess:
            self.load(sess, config.ppro_dir + str(config.QP))
            w1, w2_1, w2_2, w3_1, w3_2, w4 = sess.run([self.weights['w1'], self.weights['w2_1'], self.weights['w2_2'], self.weights['w3_1'], self.weights['w3_2'], self.weights['w4']])
            b1, b2_1, b2_2, b3_1, b3_2, b4 = sess.run([self.biases['b1'], self.biases['b2_1'], self.biases['b2_2'], self.biases['b3_1'], self.biases['b3_2'], self.biases['b4']])
        with open("vrcnn_16.0_qfp_%d.data"%config.QP, "wb") as f:
            f.write(w1)
            f.write(b1)
            f.write(w2_1)
            f.write(b2_1)
            f.write(w2_2)
            f.write(b2_2)
            f.write(w3_1)
            f.write(b3_1)
            f.write(w3_2)
            f.write(b3_2)
            f.write(w4)
            f.write(b4)
            f.write(struct.pack('6d',*self.stepw_list))
            f.write(struct.pack('6d',*self.blu_ub))

    def dump_feature(self,config):
        test_data = yuv_data.YuvData(config.QP)
        anchor = test_data.testData[0][2]
        input = anchor.reshape(1, anchor.shape[1], anchor.shape[2], 1)
        with tf.Session() as sess:
            self.load(sess, config.blu_dir+str(config.QP))
            [c1] = sess.run([self.conv4],
                                  feed_dict={self.images: input})
            with open("feature_map.data", "wb") as f:
                f.write(c1)

    def conv_validation(self,config):
        test_data = yuv_data.YuvData(config.QP)
        with tf.Session() as sess:
            self.load(sess,config.ppro_dir+str(config.QP))
            #self.quantize(sess,config.QP)
            for j in range(len(test_data.testData)):
                anchor = test_data.testData[j][2]
                input = anchor.reshape(1, anchor.shape[1], anchor.shape[2], 1)
                if input.shape[0] > 1500 or input.shape[1] > 1500:
                    print("memory limited, divided running")
                    c = self.divided_run(sess, input, self.conv2_1)
                else:
                    c = sess.run(self.conv2_1, feed_dict={self.images: input})
                #[x, w, b, c] = sess.run([self.images_norm, self.weights['w1'], self.biases['b1'], self.conv1],feed_dict={self.images: input})
                #[x,w,b,c] = sess.run([self.blu1,self.weights['w2_1'],self.biases['b2_1'],self.conv2_1], feed_dict={self.images: input})
                #[x, w, b, c] = sess.run([self.blu1, self.weights['w2_2'], self.biases['b2_2'], self.conv2_2],feed_dict={self.images: input})
                #[x, w, b, c] = sess.run([self.blu2, self.weights['w3_1'], self.biases['b3_1'], self.conv3_1],feed_dict={self.images: input})
                #[x, w, b, c] = sess.run([self.blu2, self.weights['w3_2'], self.biases['b3_2'], self.conv3_2],feed_dict={self.images: input})
                #[x, w, b, c] = sess.run([self.blu3, self.weights['w4'], self.biases['b4'], self.conv4],feed_dict={self.images: input})
                #print('x:',x[0,0:5,0:5,0])
                #print('weights:',w[0:5,0:5,0,0])
                #print('biases:',b[0])
                #print('conv:',np.around(c[0,0:5,0:5,0]*255))
                with open("feature_map.data","wb") as f:
                    f.write(c)
                print('max:',c.max(),'mean:',np.mean(np.abs(c)))
                pass
                #print('mean absolute residual:',np.mean(np.absolute(c*255)))
