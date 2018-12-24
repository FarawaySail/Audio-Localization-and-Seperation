from __future__ import print_function
import time

class Logger(object):
    def __init__(self, fp):
        self.fp = fp
    def __call__(self, string, end='\n'):
        new_string = '[%s] ' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))
