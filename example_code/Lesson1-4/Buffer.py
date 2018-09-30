#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 07:06:22 2018

@author: raffa
"""

import numpy as np

class Buffer():
        
    def __init__(self, buff_len):
        self.buff_len = buff_len
        self.buffer = []
        
    def push(self,element):
        self.buffer.append(element)
        if len(self.buffer) > self.buff_len:
            del(self.buffer[0])
        
    def is_full(self):
        return len(self.buffer) == self.buff_len
    
    def is_empty(self):
        return len(self.buffer) == 0
    
    def length(self):
        return len(self.buffer)
    
    def get_mean(self):
        if not self.is_empty():
            return np.mean(self.buffer)
        else:
            return np.NaN
        
    def get_median(self):
        if not self.is_empty():
            return np.median(self.buffer)
        else:
            return np.NaN
            
    def get_buffer(self):
        return self.buffer
    
    def buff_length(self):
        return self.buff_len
    
    def sort(self):
        self.buffer.sort()
        
    def get_sorted(self):
        return(sorted(self.buffer))
        
    def clear_buf(self):
        self.buffer.clear()
        
    def print(self):
        print(self.buffer)
        

if __name__ == "__main__":
    
    buf = Buffer(7)
    
    rand_vec = [i*np.random.random() for i in np.random.randint(0,30,20)]
    print(rand_vec)
    for i in list(rand_vec):
        print(i)        
        buf.push(i)
        print('mean:', buf.get_mean())
        print('median:',buf.get_median())
        print('buf len:', buf.length())
        buf.print()
        print(buf.get_sorted())
        print()
    print(buf.get_buffer())
    buf.clear_buf()
    print(buf.is_empty())
        