# Thrid party
import math


class SUM:
    def __init__(self, period):
        self._period = period
        self._value = 0
        self._timestamps = list()
        self._values = list()
        
    def update(self, value, timestamp):
        while len(self._timestamps) > 0 and timestamp - self._timestamps[0] > self._period:
            self._value -= self._values[0]
            self._timestamps.pop(0)
            self._values.pop(0)
        self._value += value
        self._values.append(value)
        self._timestamps.append(timestamp)
            
    def get_signal(self):
        return self._value


class EMA:
    def __init__(self, period):
        self._tau = 1. / period
        self._value = 0
        self._timestamp = 0
        self._start = False
        
    def update(self, value, timestamp):
        if self._start:
            delta = timestamp - self._timestamp
            alpha = math.exp(-delta  * self._tau)
            self._value = (1 - alpha) * value + alpha * self._value
        else:
            self._value = value
            self._start = True
        self._timestamp = timestamp
            
    def get_signal(self):
        return self._value