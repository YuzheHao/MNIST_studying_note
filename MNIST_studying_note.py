import input_data
import tensorflow as tf

# read the training data from MNIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create a session, it's the very first step in TF
sess=tf.InteractiveSession()

# Set placeholders
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_img=tf.reshape(x,[-1,28,28,1])
'''
there are three kind of basic data in TF
(1) constant: it is XXX initiately, and it can't change later.
(2) variable: it is XXX initaitely, but it can change later.
(3) placeholder: it don't have a value now, but I will tell you later, and it can change.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

the reshape function is shown as below:
> tf.reshape(input,shape,name=None)

it has 5 parameters:

1、input：the Tensor need to be reshaped

2、shape：the size after reshaping
> [batch, height, width, channels]

3、name：

! the "-1" in reshape means this parameter will be utomatically calculated. Because when all other parameters are settled, the last paremeter will be decided.
'''

# replace 2D-convolution function in TF with our self-defined expression, just for convenience
def conv2d(x,w):
return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') 
'''
after "define", it's our self-defined expression
after "return", it's the formal expression in TF

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

the standard 2d-convolution function is shown as below:
> tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
its output is a Tensor[batch, height, width, channels]

it has 6 parameters:

1、input： a Tensor as the input of the convolution 
> the four deminsion of this Tensor: [number of images in a batch, height, width, number of channels]

2、filter： filtting mask（A.K.A. convolutional kernel），it is a Tensor
> the four deminsion: [height, width, channels，number of masks] 
!Watch Out! these four parameters are not at the same positions with input Tensor
the [number of masks] can be seen as the number of neurons of this layer. Each neuron means a output image of this layer

3、strides： how long the mask moves in each direction
> the four deminsion: [up，down，left，right]
in fact, only the "down" and "left" are used, the others are 1 in most occasions

4、padding： how the mask slide on the image
there are two methods: 
‘VALID’: the whole mask will never get out of the image
‘SAME’: only will the center of mask never get out, which means these center can be located at the edge even though part of mask will get out of image

5、use_cudnn_on_gpu: it is a bool deciding whether using cudnn acceleration，its default is true

6、name: to be honest, I don't know what it is used for, but it is always blank in akmost all codes
'''


# replace max pooling function in TF with our self-defined expression
def max_pool_2x2(x):
return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

''''
the standard max pooling function is shown as below:
> tf.nn.max_pool(value, ksize, strides, padding, name=None)
its output is a Tensor[batch, height, width, channels]

it has 5 parameters:

1、value： the input Tensor 
> [batch, height, width, channels]

2、ksize： the size of pooling mask，a Tensor 
> [1, height, width, 1]
cause we don't want "batch" and "channels" get pooled, these two dimension are set to 1

3、strides： just like in 2D-conv, how long the mask moves in each direction
[1, down, left, 1]

4、padding： 'VALID' or 'SAME'

5、the same as in 2D-Convolution
'''

# 1st layer: Convolution1 and Pooling1
w_conv1=tf.Variable(tf.truncated_normal([3,3,1,32],stddev=0.1)) 
# Initialize weight of 1st layer's filter masks
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
# Initialize bias of 1st layer's filter masks, each filter has a bias
h_conv1=tf.nn.relu(conv2d(x_img,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
'''
function that outputs random value from cut truncated normal distribution
> tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

it has 6 parameters (mainly 4)

1、shape: shape of output > [batch, height, width, channels]

2、mean: the mean of normal normal distribution

3、stddev: the standard deviation of normal normal distribution

4、dtype: type of data

5、always default

6、always default

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

Nonlinear activation function
> h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

there is always a such function in each layer and it shows on two positions:
(1) After convolution
(2) At last of FC layer
'''

# 2nd layer: Convolution2 and Pooling2
w_conv2=tf.Variable(tf.truncated_normal([3,3,32,50],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[50]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

# 3rd layer: FullyConnected1
w_fc1=tf.Variable(tf.truncated_normal([7*7*50,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*50]) # reshape the output of layer pooling2 to the size fitted with FC layer
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
'''
matrix multiplication function
> tf.matmul(A,B)
'''

# dropout
keep_prob=tf.placeholder(tf.float32) # we need a placeholder before dropout
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
'''
Dropout function — for preventing overfitting
> x_drop=tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)

always use the former 2 parameters:

1、x: input tensor

2、keep_prob: a float，The probability that each element can be preserved
'''


# 4th layer: FullyConnected2
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_out=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
'''
softmax function: transfer the result to a mapping of probavility
> x_drop=tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
in the last FC layer, we use softmax rather than relu

always use the former 2 parameters:

1、x: input tensor

2、keep_prob: a float，The probability that each element can be preserved
'''

#building loss function，we use cross entroy here.
loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_out),reduction_indices=[1]))
'''
calculate max value
> tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)

calculate mean value
> tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)

they have 4 parameters:

1、input_tensor：just as its name —— input Tensor

2、reduction_indices：which specific dimension are we going to calculate

3、4、usually default

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

calculate natural logarithm
> tf.log(input_tensor, name=None)
'''

# set Adam optimizer to minimize loss function, studying rate is set to 1e-4
training_step=tf.train.AdamOptimizer(1e-4).minimize(loss)

# build the expression of accuracy
bingo=tf.equal(tf.argmax(y_out,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(bingo,tf.float32))
'''
> tf.argmax(A,1)
when A is a vector, return the column of max value in the vector
when A is a matrix, return a vector, whose number of column is equal to number of matrix' row,
each value means the column of max value in each matrix' row

eg: 
A = [[2,4,2],[4,2,5]]
>>> tf.argmax(A,1) = [1,2]
which means: the max value in 1st row of A is located in 2nd column, and the max value in 2st row of A is located in 3rd column

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

> tf.equal(A,B)
compare the element at the same position in A and in B
if they are the same, output "True" at the same position in output matrix, oppesitely, output "False"

eg:
A = [[2,4,2],[4,2,5]]
B = [[2,3,6],[4,2,5]]
>>> tf.equal(A,B) = [[True,False,False],[True,True,True]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

> tf.cast(a,dtype)
transfer the data type of a to the type of "dtype"
'''

# start training
tf.global_variables_initializer().run() # standard form of variable initialization
for i in range(25000): # iterating for 25000 times
    batch=mnist.train.next_batch(50)
    if i%100==0: # testing for once after training for 100 times
        training_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
        print "step %d,training_accuracy= %g"%(i,training_accuracy)
    training_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

# testing after training
print "testing_accuracy= %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1})