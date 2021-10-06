import os
import signal
import time


def uniquify(path):
    # creates a unique file name by adding an incremented number
    # to the existing filename
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + '_' + str(counter) + extension
        counter += 1
    return path

########################
# graceful exit
##################
class GracefulExiter():
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print('exit flag set to True (repeat to exit now)')
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state
