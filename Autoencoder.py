import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

learning_rate = 0.001
n_hidden1 = 512
n_hidden2 = 4
n_input = 784
training_epochs = 100
batch_size = 200
display_step = 25

x = tf.placeholder(tf.float32,[None,n_input])
y = x


def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

def batch_normal(x):
    return x/255.

def encoder(x):
    layer_1 = tf.nn.relu(tf.matmul(x,W['encoder_h1'])+b['encoder_b1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1,W['encoder_h2'])+b['encoder_b2'])
    return layer_2

def decoder(x):
    layer_1 = tf.nn.relu(tf.matmul(x,W['decoder_h1'])+b['decoder_b1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1,W['decoder_h2'])+b['decoder_b2'])
    return layer_2

W = {
    'encoder_h1':weight_variable([n_input,n_hidden1]),
    'encoder_h2':weight_variable([n_hidden1,n_hidden2]),
    'decoder_h1':weight_variable([n_hidden2,n_hidden1]),
    'decoder_h2':weight_variable([n_hidden1,n_input]),
}

b = {
    'encoder_b1':bias_variable([n_hidden1]),
    'encoder_b2':bias_variable([n_hidden2]),
    'decoder_b1':bias_variable([n_hidden1]),
    'decoder_b2':bias_variable([n_input]),
}

encoder_out = encoder(batch_normal(x))
encoder_image = tf.reshape(encoder_out,(-1,2,2))
pred = decoder(batch_normal(encoder_out))

loss = tf.reduce_mean((y-pred)**2)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer() ) 
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练 
    total_batch = int(mnist.train.num_examples/batch_size) #总批数 
    for epoch in range(training_epochs): 
        for i in range(total_batch): 
            batch_xs,_= mnist.train.next_batch(batch_size)
            _, c = sess.run([train, loss], feed_dict={x: batch_xs}) 
        if (epoch+1) % display_step == 0: 
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            test_image,_ = mnist.test.next_batch(8)            
            encoder_img,encode_decode = sess.run([encoder_image,pred], feed_dict={x:test_image})
            fig, ax = plt.subplots(3, 8, figsize=(8, 3)) 
            for i in range(8): 
                ax[0][i].imshow(np.reshape(test_image[i], (28, 28))) 
                ax[1][i].imshow(np.reshape(encoder_img[i], (2, 2))) 
                ax[2][i].imshow(np.reshape(encode_decode[i], (28, 28))) 
                ax[0][i].get_xaxis().set_visible(False)
                ax[0][i].get_yaxis().set_visible(False)
                ax[1][i].get_xaxis().set_visible(False)
                ax[1][i].get_yaxis().set_visible(False)
                ax[2][i].get_xaxis().set_visible(False)
                ax[2][i].get_yaxis().set_visible(False)
            plt.show() 
    print("OVER") 

