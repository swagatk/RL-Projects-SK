import os


def uniquify(path):
    # creates a unique file name by adding an incremented number
    # to the existing filename
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + '_' + str(counter) + extension
        counter += 1
    return path
