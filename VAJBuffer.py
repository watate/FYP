#note that VAJBuffer is now also a buffer for the other input: "Target in the local frame"

from collections import deque
import random

class VAJBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, linear_vel, angular_vel, linear_accel, angular_accel, linear_jerk, angular_jerk, local_x, local_y):
        experience = (linear_vel, angular_vel, linear_accel, angular_accel, linear_jerk, angular_jerk, local_x, local_y)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def pop1(self):
        motion1 = self.buffer.pop()
        self.buffer.append(motion1)
        return motion1

    def pop2(self):
        motion1 = self.buffer.pop()
        motion2 = self.buffer.pop()
        self.buffer.append(motion2)
        self.buffer.append(motion1)
        return motion1, motion2