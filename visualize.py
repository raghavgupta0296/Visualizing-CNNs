import cv2
import numpy as  np
import tensorflow as tf

class visualize_cnns:

    def __init__(self, batch_size, tensors):
        self.image_list = ["Image%d"%x for x in range(batch_size)]
        self.tensor_list = [x.name for x in tensors]
        self.current_im_name = self.image_list[0]
        self.current_tensor_name = self.tensor_list[0]
        cv2.namedWindow("Input")

        self.gui()

    def add_tensor(self, tensor):
        self.tensor_list.append(tensor.name)
        self.gui()

    def gui(self):
        cv2.createTrackbar("Image","Input",0,len(self.image_list)-1,self.selectedImBatch)
        cv2.createTrackbar("Tensor","Input",0,len(self.tensor_list)-1,self.selectedTensor)

    def selectedTensor(self,chosen_tensor_name):
        self.current_tensor_name = self.tensor_list[chosen_tensor_name]

    def selectedImBatch(self,chosen_imBatch_name):
        self.current_im_name = self.image_list[chosen_imBatch_name]

    def update_visuals(self, ims, tensor_vals):
        cv2.imshow("Input", ims[int(self.current_im_name.lstrip("Image"))])
        tensor_val = tensor_vals[self.tensor_list.index(self.current_tensor_name)]
        tensor_val = tensor_val[int(self.current_im_name.lstrip("Image"))]
        del tensor_vals
        wid, ht, channels = tensor_val.shape[0], tensor_val.shape[1], tensor_val.shape[2]
        columns = rows = int(np.ceil(np.sqrt(channels)))
        tensor_val = np.reshape(tensor_val, (wid, ht, channels))
        big_im = np.zeros(shape=(ht * rows, wid * columns))

        channel_i = 0
        for i in range(0, ht * rows, ht):
            for j in range(0, wid * columns, wid):
                if channel_i < channels:
                    big_im[i:i + wid, j:j + ht] = tensor_val[:, :, channel_i]
                channel_i += 1
        cv2.putText(big_im, self.current_tensor_name, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Visualizations", big_im)
        cv2.waitKey(1)

if __name__ == "__main__":
    X = tf.placeholder(tf.float32,shape=(None,200,200,1))

    def ini_wt(shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return initial

    def ini_bias(shape):
        initial = tf.zeros(shape)
        return initial

    W1 = ini_wt([3,3,1,32])
    b1 = tf.Variable(ini_bias([32]))
    layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.bias_add(layer1, b1)
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.layers.max_pooling2d(layer1, pool_size=[2, 2], strides=2)
    W2 = ini_wt([10, 10, 32, 64])
    b2 = tf.Variable(ini_bias([64]), name='b2')
    layer3 = tf.nn.conv2d(layer2, W2, strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.bias_add(layer3, b2)
    layer3 = tf.nn.relu(layer3)
    layer4 = tf.layers.max_pooling2d(layer3, pool_size=[2, 2], strides=2)

    c = visualize_cnns(batch_size=2, tensors=[layer1,layer2,layer3, W2])
    c.add_tensor(layer4)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ims = []
        for f in ["./1/81777.png","./1/81781.png"]:
            im = cv2.imread(f)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            im = np.expand_dims(im, axis=-1)
            ims.append(im)
        ims = np.array(ims)
        while True:
            layer1_, layer2_, layer3_, layer4_, W2_ = sess.run([layer1, layer2, layer3, layer4, W2],feed_dict={X:ims})
            c.update_visuals(ims,[layer1_,layer2_,layer3_,layer4_,
                                  np.reshape(W2_,(W2_.shape[2],W2_.shape[0],W2_.shape[1],W2_.shape[3]))])
