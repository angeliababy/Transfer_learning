#!/usr/bin/env python
# encoding: utf-8


'''
图像识别：
该部分是实现基于预训练模型inception-v3的训练部分（第一步）
'''


import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 数据参数
MODEL_DIR = '../pre_model/'  # inception-v3模型的文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'  # inception-v3模型文件名
CACHE_DIR = '../datas/tmp/bottleneck'  # 图像的特征向量保存地址
INPUT_DATA = '../datas/train'  # 图片数据文件夹
TEST_PERCENTAGE = 10  # 验证数据的百分比

# inception-v3模型参数
BOTTLENECK_TENSOR_SIZE = 2048  # inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 模型超参数及存储路径
# 神经网络的训练参数
LEARNING_RATE = 0.01
# 迭代次数
STEPS = 8000
# 样本批次大小
BATCH = 100
# 每隔多少步保存模型
CHECKPOINT_EVERY = 100
MODEL_SAVE_PATH = '../models/checkpoints/'
MODEL_NAME = 'model'

'''
功能：从数据文件夹中读取所有的图片列表并按训练、验证、测试分开
输入：测试集百分占比
输出：训练集、测试集的字典
'''
def create_image_lists(test_percentage):
    result = {}  # 保存所有图像,key为类别名称,value也是字典，存储了所有的图片名称
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录（包括当前目录）
    is_root_dir = True  # 第一个目录为当前目录，需要忽略

    # 分别对每个子目录进行操作
    for sub_dir in sub_dirs:
        if is_root_dir:  # 忽略当前目录
            is_root_dir = False
            continue

        # 获取当前子目录下的所有有效图片
        extensions = {'jpg', 'jpeg', 'JPG', 'JPEG'}
        file_list = []  # 存储所有图像
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名字，即类别名
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))  # 使用glob.glob获得文件路径，并加入file_list
        if not file_list:
            continue

        # 将当前类别的图片随机分为训练数据集、测试数据集
        label_name = dir_name.lower()  # 通过目录名获取类别的名称（小写）
        training_images = []
        test_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 获取该图片的名称
            chance = np.random.randint(100)  # 随机产生100个数代表百分比
            if chance < test_percentage:
                test_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据集放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'test': test_images
        }
    # 返回整理好的所有数据
    return result

'''
功能：通过类别名称、所属数据集、图片编号获取一张图片的地址
输入：数据字典、数据特征向量地址、类别名、索引号、训练/测试
输出：一张图片的地址
'''
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取给定类别中的所有图片
    category_list = label_lists[category]  # 根据所属数据集的名称获取该集合中的全部图片
    mod_index = index % len(category_list)  # 规范图片的索引
    base_name = category_list[mod_index]  # 获取图片的文件名
    sub_dir = label_lists['dir']  # 获取当前类别的目录名
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 图片的绝对路径
    return full_path

'''
功能：使用inception-v3处理图片获取特征向量
输入：sess,数据字典,数据输入张量和瓶颈层输出张量
输出：一张图片的特征向量
'''
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)  # 压缩成一维数组
    return bottleneck_values

'''
功能：获取一张图片经过inception-v3模型处理后的特征向量
输入：sess,数据字典，类别名，第几个，训练/测试，数据输入张量和瓶颈层输出张量
输出：一张图片经过inception-v3模型处理后的特征向量
'''
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    # 存储图片特征向量的地址
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    # 通过类别名称、所属数据集、图片编号获取特征向量值的地址
    bottleneck_path = get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

    # 如果该特征向量文件不存在，则通过inception-v3模型计算并保存
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)  # 获取图片原始路径
        image_data = gfile.FastGFile(image_path, 'rb').read()  # 获取图片内容
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)  # 通过inception-v3计算特征向量

        # 将特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 否则直接从文件中获取图片的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # 返回得到的特征向量
    return bottleneck_values

'''
功能：随机获取一个batch图片作为训练数据
输入：sess，类别，数据字典、每批次数，训练/测试、数据输入张量和瓶颈层输出张量
输出：训练数据及标签
'''
def get_random_cached_bottlenecks(sess, n_classes, image_lists, batch,
                                  category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(batch):
        # 随机一个类别和图片编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        # 获取一张图片经过inception - v3模型处理后的特征向量
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        # 制造标签
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有的图片,并按训练集、测试集分开
    image_lists = create_image_lists(TEST_PERCENTAGE)
    # 类别数
    n_classes = len(image_lists.keys())

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

    # 训练过程
    with tf.Session(graph=graph) as sess:

        # 检查点存在否
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 检查点
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 加载已存在的模型，从断点开始训练
        else:
            tf.global_variables_initializer().run()  # 初始化所有变量

        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training',
                jpeg_data_tensor, bottleneck_tensor)
            # 开始训练
            _ ,step = sess.run(
                [train_step, global_step],
                feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # 在验证集上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                # 每次获取一个batch的测试数据
                test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'test',
                    jpeg_data_tensor, bottleneck_tensor)
                test_accuracy = sess.run(
                    evaluation_step,
                    feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
                print('Step %d : test accuracy on random sampled %d examples = %.1f%%' % (step, BATCH, test_accuracy * 100))

            # 每隔checkpoint_every保存一次模型
            if step % CHECKPOINT_EVERY == 0:
                path = saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)  # 保存模型
                print('Saved model checkpoint to {}\n'.format(path))


if __name__ == '__main__':
    tf.app.run()