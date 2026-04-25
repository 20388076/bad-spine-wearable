# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:33:45 2025

@author: AXILLIOS
"""
with open("x_3step_per_min_w_anomaly.txt", "r", encoding="latin1") as fin, \
     open("x_3step_per_min_w_anomaly.csv", "w", encoding="latin1") as fout:
    for line in fin:
        fout.write(line)
