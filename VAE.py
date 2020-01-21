import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)


#画图方法
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

#输入维度
input_dim = 784
#隐层编码维度
hidden_encoder_dim = 400
#隐层解码维度
hidden_decoder_dim = 400
#隐变量z的维度
z_dim = 20
#学习率
lr = 0.001
#minibatch的大小
batch_size = 250


#定义权重方法：
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)
#定义偏置方法：
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# =============================== 编码部分 ======================================

X = tf.placeholder(tf.float32, shape=[None, input_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

#编码器隐藏层权重及偏置
Q_W1 = tf.Variable(weight_variable([input_dim, hidden_encoder_dim])) #[784,400]
Q_b1 = tf.Variable(bias_variable(shape=[hidden_encoder_dim])) #[400]
#参数mu权重及偏置
Q_W2_mu = tf.Variable(weight_variable([hidden_encoder_dim, z_dim])) #[400,20]
Q_b2_mu = tf.Variable(bias_variable(shape=[z_dim])) #[20]
#参数log(sigma)权重及偏置
Q_W2_logsigma = tf.Variable(weight_variable([hidden_encoder_dim, z_dim])) #[400,20]
Q_b2_logsigma = tf.Variable(bias_variable(shape=[z_dim]))   #[20]


#编码器方法
def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1) #[None,400]
    #正态分布的均值：
    mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu   #[None,20]
    #正太分布方差的对数，，这里直接计算成方差的对数
    log_var = tf.matmul(h, Q_W2_logsigma) + Q_b2_logsigma #[None,20]
    return mu,log_var


#计算Z
def sample_z(mu,log_var):
    #标准正态分布
    eps = tf.random_normal(shape=tf.shape(mu))
    #返回Z的分布
    return mu + tf.exp(log_var / 2) * eps




# =============================== 解码部分 ======================================
#解码器隐藏层权重及偏置
P_W1 = tf.Variable(weight_variable([z_dim, hidden_decoder_dim])) #[20,400]
P_b1 = tf.Variable(bias_variable(shape=[hidden_decoder_dim]))   #[400]
#解码器最后一层权重及偏置
P_W2 = tf.Variable(weight_variable([hidden_decoder_dim, input_dim])) #[400,784]
P_b2 = tf.Variable(bias_variable(shape=[input_dim])) #[784]

#解码器方法
def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob,logits


#获取mu，以及方差的对数
mu,logvar = Q(X)
#获取Z的分布
z = sample_z(mu,logvar)
#通过z生成图像
X_samples,logits = P(z)

#重构损失,比较还原的图像与原图像的损失
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X),1)
#KL散度
KLD = 0.5*tf.reduce_sum(tf.pow(mu, 2) + tf.exp(logvar) - 1 - logvar,1)
# VAE loss
vae_loss = tf.reduce_mean(KLD + BCE)

train = tf.train.AdamOptimizer(lr).minimize(vae_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(100001):
        
        batch_xs, _ = mnist.train.next_batch(batch_size) 
        _, loss = sess.run([train, vae_loss], feed_dict={X: batch_xs})

        if it % 2500 == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'. format(loss))
            print('-----------------------')
            samples = sess.run(X_samples, feed_dict={z:np.random.randn(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

