from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def pcnn(is_training):
    if is_training:
        framework = Framework(is_training=True, use_bag=False)
    else:
        framework = Framework(is_training=False, use_bag=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.pcnn(embedding, FLAGS.hidden_size, framework.mask, activation=tf.nn.relu)
    logit, repre = framework.selector.no_bag(x)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(logit))
        framework.load_test_data()
        framework.test()
