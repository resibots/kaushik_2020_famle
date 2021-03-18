# -*- coding: utf-8 -*-
import sys, os
from pyprind import ProgBar as Pb

class stream_writer:
    STDOUT = 1
    STDERR = 2
    def __init__(self, stream=STDOUT):
        self.print_func = sys.stdout.write if stream==self.STDOUT else sys.stderr.write
        self.flush_func = sys.stdout.flush if stream==self.STDOUT else sys.stderr.flush

    def write(self,string):
        if not 'DISPLAY' in os.environ: 
            result = string.find('ETA:')
            self.print_func(string[result::])
        else: 
            self.print_func(string)

    def flush(self):
        self.flush_func()

class ProgBar(Pb):
    def __init__(self, iterations, track_time=True, width=30, bar_char='#', stream=stream_writer(), 
                title='', monitor=False, update_interval=None):
        # super(ProgBar, self).__init__(iterations=iterations, track_time=track_time, width=width, bar_char=bar_char, 
        #                                 stream=stream, title=title, monitor=monitor, update_interval=update_interval)
        Pb.__init__(self, iterations=iterations, track_time=track_time, width=width, bar_char=bar_char, 
                                        stream=stream, title=title, monitor=monitor, update_interval=update_interval)

if __name__=='__main__':
    import time
    bar = ProgBar(100, width=100, track_time=True, title='\nExecuting....', bar_char='▒')
    for i in range(100):
        time.sleep(0.1)
        bar.update(item_id= " Step " + str(i) + " ")