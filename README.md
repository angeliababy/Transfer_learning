# 迁移学习、预训练模型、图片预测

本项目完整源码地址：[https://github.com/angeliababy/Transfer_learning](https://github.com/angeliababy/Transfer_learning)

项目博客地址: [https://blog.csdn.net/qq_29153321/article/details/103973211](https://blog.csdn.net/qq_29153321/article/details/103973211)

    识花，花朵数据集

    选用数据为同一大类的样本，5个类别的花种图像，共3000多张，同一大类别且样本数较少，故识别难度更大。
    使用预训练模型inception-v3进行迁移学习，这个本来就是做图像识别的模型，只不过最后的类别数不同。使用预训练模型（模型及权重）进行最后一层输出层的调整，能够加快收敛速度和效果。

## 各目录情况如下：

### datas：

    存储训练（含测试）图片集、预测图片集、
    tmp存储图片数据经过预训练模型后得到的特征向量（倒数第二层，称为瓶颈层，模型inception-v3输出节点数为2048）
    
### pre_model

    准备好的预训练模型文件

### models:

    存放生成的模型文件及日志

### train_models:

    模型训练过程，该模型没有数据预处理过程，直接将数据随机分为训练、测试样本后，每张丢进预训练模型得到2048维特征向量作为一个样本，
	并为它打上one-hot标签进行训练，选用的类别数为5类，直接构造一个全连接层进行训练便可。
   
### predicts:

    模型的预测过程
    
## 运行过程示例

    1. 第一步，模型训练及保存模型，运行train_nodel/train.py
    2. 第二步，模型恢复及预测过程，运行predicts/predict.py
    
 ## 训练
 1.模型训练、测试数据
 ```
 # 读取所有的图片,并按训练集、测试集分开
 image_lists = create_image_lists(TEST_PERCENTAGE)
 ```
2.定义模型结构

第一步，定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的结点取值
第二步，加入新的全连接层进行分类预测
```
    with tf.Graph().as_default() as graph:
        # 读取训练好的inception-v3模型
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # 加载inception-v3模型，并返回瓶颈层输出张量和数据输入张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

        # 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的结点取值。
        # 可以将这个过程类似的理解为一种特征提取tensor
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')

        # 定义新的标准答案类别输入tensor
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

        # 定义一层全连接层解决新的图片分类问题
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.1)) # 权重
            biases = tf.Variable(tf.zeros([n_classes])) # 偏置
            logits = tf.matmul(bottleneck_input, weights) + biases # 全连接w*x+b
            final_tensor = tf.nn.softmax(logits)

        global_step = tf.Variable(0, trainable=False)  # 训练步数
        # 定义交叉熵损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean, global_step = global_step)

        # 计算正确率
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
3.训练
恢复模型
```
# 检查点存在否
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 检查点
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 加载已存在的模型，从断点开始训练
        else:
            tf.global_variables_initializer().run()  # 初始化所有变量
```

将处理好的每张图片的向量（经过Inception-v3模型前向传播到达瓶颈层时的模型输出向量）喂入新的模型
```
# 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training',
                jpeg_data_tensor, bottleneck_tensor)
            # 开始训练
            _ ,step = sess.run(
                [train_step, global_step],
                feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
```

4.单元预测
调用unittest包

```
import unittest
import sys
sys.path.append('../validation')
import test_predict

class MyTestCase(unittest.TestCase):

    #测试未知图片
    def test_unkonw_img(self):
        retcode = test_predict.unknow_img_test()
        self.assertEqual(1, retcode)

if __name__ == '__main__':
    unittest.main()
```

最后，模型能达到90%左右的准确率，之前用神经网络只有30%左右。
