#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 02:03:54 2021

@author: anelmusic
"""

from sklearn.ensemble import RandomForestClassifier

# Rf classifier
rf_classifier = RandomForestClassifier(random_state=0, n_estimators=500, max_depth=5)
