#!/usr/bin/env python
# encoding: utf-8


'''
单元测试
'''

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
