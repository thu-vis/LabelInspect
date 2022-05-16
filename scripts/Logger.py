import sys
from os.path import exists
from os import remove

class Logger(object):
    def __init__(self, path):
        if exists(path):
            remove(path)
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()
