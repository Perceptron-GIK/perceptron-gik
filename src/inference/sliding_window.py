from collections import deque
import numpy as np

class SlidingWindow:
    '''
    Wrapper class for data received from one hand
    '''
    def __init__(self, maxlen=None):
        if maxlen:
            self.data = deque(maxlen=maxlen)
        else:
            self.data = deque()

    def append(self, new_data):
        '''
        new_data: A single row of data received from the Arduino
        '''
        self.data.append(new_data)

    def pop_chunk(self, n):
        '''
        Given an index that corresponds to a row of data, extract all data in the queue up to the index
        '''
        return [self.data.popleft() for _ in range(min(n, len(self.data)))]
    
    def fsr_detected(self, fsr_indices):
        '''
        Checks for FSR falling edge

        If FSR falling edge found:
            Return the row index where FSR was last active
        Else:
            Return None
        '''
        indices = np.array(fsr_indices)
        
        for i in range(len(self.data) - 1):
            row = self.data[i]
            next_row = self.data[i+1]

            fsr_active = indices[row[indices] == 1]
            if len(fsr_active) == 0:
                continue

            if np.any(next_row[fsr_active] == 0):
                return i
        
        return None
    
    def timestamp_matched(self, timestamp):
        '''
        Returns the row index with corresponding timestamp just before the given one.
        Used to identify which samples to extract from the opposite hand where FSR was not detected.
        '''
        for i in range(1, len(self.data)):
            if self.data[i][-1] > timestamp:
                return i - 1
        return None
