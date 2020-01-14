#!/usr/bin/env python
# encoding: utf-8


'''
图像识别：
该部分是实现预测（第二步）
'''

import tensorflow as tf
import numpy as np

# 模型目录
CHECKPOINT_DIR = '../models/checkpoints'
INCEPTION_MODEL_FILE = '../pre_model/tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 测试数据及标签
image_path = '../datas/predict/0_1.jpg'
y_test = [0]

'''
功能：预测部分
输入：图片路径
输出：预测结果
'''
def evaluate(image_path):
    # 读取数据
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # 评估
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
            # 读取训练好的inception-v3模型
            with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

            # 使用inception-v3处理图片获取特征向量
            bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
            # 压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
            bottleneck_values = [np.squeeze(bottleneck_values)]

            # 加载元图和变量
            checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

            # 预测值占位符
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

            # 预测值
            all_predictions = sess.run(predictions, {input_x: bottleneck_values})

    # 如果提供了标签，则打印正确率
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print('\nTotal number of test examples: {}'.format(len(y_test)))
        print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))

    return all_predictions


def main(argv=None):
    # 打印单张图片预测结果
    evaluate(image_path)

if __name__ == '__main__':
    tf.app.run()