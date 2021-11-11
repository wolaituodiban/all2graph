import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def diff(self):
        return time.time() - self.start_time

    def __enter__(self):
        print('"{}" start'.format(self.name))
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('"{}" used {} seconds'.format(self.name, self.diff()))
