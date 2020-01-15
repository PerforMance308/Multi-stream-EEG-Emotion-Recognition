import tensorflow as tf

class F3CNN:
    def __init__(self, input, labels, device):
        self.device = device
        self.build_net(input, labels)
   
    def conv(self, input, filter_size, out_channels, name, activation=True):
        with tf.device(self.device):
            with tf.name_scope(name):
                _,_,_,c = input.get_shape().as_list()
                filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, c, out_channels], dtype=tf.float32, stddev = 0.01), name='weight', trainable=True)
                biases = tf.Variable(tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=True, name='biases')
                conv = tf.nn.conv2d(input, filter, [1,1,1,1], padding='SAME')
                conv = tf.nn.bias_add(conv, biases)

                if activation:
                    conv = tf.nn.relu(conv)
                return conv

    def max_pool(self, input, name):
        with tf.device(self.device):
            with tf.name_scope(name):
                return tf.nn.max_pool(input, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

    def dense(self, input, units, name, activation=True):
        with tf.device(self.device):
            with tf.name_scope(name):
                _,C = input.get_shape().as_list()
                weight = tf.Variable(tf.random_normal((C, units)))
                biases = tf.Variable(tf.random_normal((units,)))
                dense = tf.matmul(input, weight) + biases
                if activation:
                    return tf.nn.relu(dense)
                return dense
            
    def ResBlock(self, inputs, out_channels, ksize, scope):
        with tf.device(self.device):
            with tf.name_scope(scope):
                net = self.conv(inputs, ksize, out_channels, 'conv1')
                net = self.conv(net, ksize, out_channels, 'conv2', activation=None)
                return net + inputs
    
    def build_net(self, data, label):
        with tf.device(self.device):
            net = self.conv(data, 5, 32, 'enc1_1')
            net = self.ResBlock(net, 32, 5, scope='enc1_2')
            net = self.ResBlock(net, 32, 5, scope='enc1_3')
            net = self.ResBlock(net, 32, 5, scope='enc1_4')
            net = self.conv(data, 5, 64, 'enc2_1')
            net = self.ResBlock(net, 64, 5, scope='enc2_2')
            net = self.ResBlock(net, 64, 5, scope='enc2_3')
            net = self.ResBlock(net, 64, 5, scope='enc2_4')
            net = self.conv(data, 5, 128, 'enc3_1')
            net = self.ResBlock(net, 128, 5, scope='enc3_2')
            net = self.ResBlock(net, 128, 5, scope='enc3_3')
            net = self.ResBlock(net, 128, 5, scope='enc3_4')

            net = tf.reshape(net, [-1, 62*2*128])
            
            net = self.dense(input=net, units=1024, name='dense1')

            net = self.dense(input=net, units=500, name='dense2')
            
            self.encoder = net
            
            net = self.dense(input=net, units=62*2*128, name='dense2')
            
            net = tf.reshape(net, [-1, 62, 2, 128])
            
            net = self.ResBlock(net, 128, 5, scope='dec3_3')
            net = self.ResBlock(net, 128, 5, scope='dec3_2')
            net = self.ResBlock(net, 128, 5, scope='dec3_1')
            net = self.conv(data, 5, 64, 'enc2_4')
            net = self.ResBlock(net, 64, 5, scope='dec2_3')
            net = self.ResBlock(net, 64, 5, scope='dec2_2')
            net = self.ResBlock(net, 64, 5, scope='dec2_1')
            net = self.conv(data, 5, 32, 'enc1_4')
            net = self.ResBlock(net, 32, 5, scope='dec1_3')
            net = self.ResBlock(net, 32, 5, scope='dec1_2')
            net = self.ResBlock(net, 32, 5, scope='dec1_1')
            net = self.conv(data, 5, 5, 'enc0', activation = False)
            
            self.logits = tf.nn.sigmoid(net)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=data, logits=net)
            self.loss = tf.reduce_mean(loss)
