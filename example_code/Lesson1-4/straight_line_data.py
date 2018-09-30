#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 07:56:03 2018

@author: raffa
"""

def get_straight_line_data(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    gradient = (y2 - y1) / (x2 - x1)
    intercept = (x2*y1 - x1*y2) / (x2 - x1)        
    return (gradient,intercept)

p1 = (80,-530)
p2 = (120,-350)

g,i = get_straight_line_data(p1,p2)
print((g,i))
#print(g*)