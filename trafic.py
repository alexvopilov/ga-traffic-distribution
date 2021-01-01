#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
class BKM321:
    def __init__(self):
        self.seat_places = 26
        self.capacity = 115
        self.count = 101

class Trolleybus:
    def __init__(self,id,cars=0,people=0,model=BKM321):
        self.id = id
        self.car = model()
        self.cars = cars
        self.people = people

