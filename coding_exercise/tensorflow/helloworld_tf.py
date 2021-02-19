import tensorflow as tf

# Create Tensorflow object called tensor
hello_constant = tf.constant('Hello World!')


''' TF v1.x
with tf.Session() as sess:
	# Run the tf.constant operation in the session
	output = sess.run(hello_constant)
	print(output)
'''

''' TF v2.x 
tf.print(hello_constant)
'''

with tf.compat.v1.Session() as sess:
	hello_constant = tf.constant('Hello World!')
	print(sess.run(hello_constant))