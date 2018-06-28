
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names
import cv2
import matplotlib.pyplot as plt

#%%
# Load image
img = cv2.imread('/home/alvinsun/Documents/Code/PhD/tf_image_recongnition_tutorial/images/dog.jpeg', 3)
img = cv2.resize(img, (224, 224))
plt.imshow(img)
img = img.reshape([1,224,224,3])

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


#%%
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
vgg = tl.models.VGG16(x)

# restore pre-trained VGG parameters
sess = tf.InteractiveSession()

vgg.restore_params(sess)

probs = tf.nn.softmax(vgg.outputs)

vgg.print_params(False)

vgg.print_layers()

#%%

_ = sess.run(probs, feed_dict={x: img})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: img})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])