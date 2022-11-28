# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmdeploy.utils.timer import TimeCounter


def test_count_time():

    class test:

        @TimeCounter.count_time('fun1')
        def fun1(self):
            time.sleep(0.01)

    t = test()
    with TimeCounter.activate('fun1', warmup=10, log_interval=10):
        for i in range(50):
            t.fun1()

    for i in range(50):
        t.fun1()

    TimeCounter.print_stats('fun1')
