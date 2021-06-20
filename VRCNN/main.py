from model import VRCNN
import tensorflow as tf
import pprint

flags = tf.app.flags
flags.DEFINE_boolean("blu", False, "True for blu, False for relu")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing")
flags.DEFINE_boolean("continue_train",False,"")
flags.DEFINE_boolean("finetune",False,"")
flags.DEFINE_string("ppro_dir","checkpoint_16.0_ppro_QP","Running platform")
flags.DEFINE_string("blu_dir","checkpoint_16.0_blu_QP","Running platform")
flags.DEFINE_integer("QP", 37, "quantize parameter")
flags.DEFINE_integer("epoch", 30, "Number of epoch")
flags.DEFINE_integer("batch_size", 64, "The size of batch images")
flags.DEFINE_integer("sub_image_size", 64, "The size of image to use")
flags.DEFINE_integer("sub_label_size", 64, "The size of label to produce")
flags.DEFINE_integer("stride", 35, "The size of stride to apply input image")#24
flags.DEFINE_integer("padding", 1, "for test")#24
flags.DEFINE_float("learning_rate", 0.00001, "The learning rate of gradient descent algorithm")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color.")
FLAGS = tf.flags.FLAGS
pp = pprint.PrettyPrinter()

def main(_):
    #pp.pprint(flags.FLAGS.__flags)

    model = VRCNN(
                FLAGS.QP,
                blu=FLAGS.blu,
                sub_image_size=FLAGS.sub_image_size,
                sub_label_size=FLAGS.sub_label_size,
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                lr=FLAGS.learning_rate
                )
    if FLAGS.is_train:
        if FLAGS.finetune:
            model.quant_finetune(FLAGS)
        else:
            model.train(FLAGS)
    else:
        model.quantize1(FLAGS)
        #model.dump_feature(FLAGS)
        #model.conv_validation(FLAGS)
        #model.test(FLAGS)
        #model.dump(FLAGS)
            #model.freeze_model(FLAGS.checkpoint_dir,output_node,FLAGS.checkpoint_dir+'model.pb')

if __name__ == '__main__':
  tf.app.run()
