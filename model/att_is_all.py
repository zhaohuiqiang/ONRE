from framework import Framework
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def att_is_all(is_training):
    FLAGS.hidden_size = 320

    if is_training:
        framework = Framework(is_training=True)
    else:
        framework = Framework(is_training=False)

    word_embedding = framework.embedding.word_embedding()
    pos_embedding = framework.embedding.pos_embedding()
    embedding = framework.embedding.concat_embedding(word_embedding, pos_embedding)
    x = framework.encoder.attention_is_all_you_need(embedding, FLAGS.hidden_size, framework.mask)
    logit, repre = framework.selector.attention(x, framework.scope, framework.label_for_select)

    if is_training:
        loss = framework.classifier.softmax_cross_entropy(logit)
        output = framework.classifier.output(logit)
        framework.init_train_model(loss, output, optimizer=tf.train.GradientDescentOptimizer)
        framework.load_train_data()
        framework.train()
    else:
        framework.init_test_model(tf.nn.softmax(logit))
        framework.load_test_data()
        framework.test()
