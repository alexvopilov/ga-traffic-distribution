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

class Traffic:
    def __init__(self, tbuses, days = 5, cars = 1, people = 1):
        self.tbuses = []
        self.days = days
        
        for t in tbuses:
            self.tbuses.append(Trolleybus(id=t,cars=cars,people=people))
            
    def tolist(self):
        return [[[t.id,t.cars,t.people] for t in self.tbuses] for _ in range(self.days)]
    
    def toarray(self):
        return np.array(self.tolist())
            
    def __str__(self):
        r = []
        for d in range(self.days):
            r.append(f"Day {d+1}: ")
            for t in self.tbuses:
                r.append(f"{t.id} {t.cars} {t.people} | ")
            r.append("\n")
        return "".join(r)
    
    def __repr__(self):
        return self.__str__()
