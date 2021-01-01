#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd


# In[75]:


class BKM321:
    def __init__(self):
        self.seat_places = 26
        self.capacity = 115
        self.count = 101


# In[76]:


class Trolleybus:
    def __init__(self,id,cars=0,people=0,model=BKM321):
        self.id = id
        self.car = model()
        self.cars = cars
        self.people = people


# In[84]:


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


# In[93]:


def cost(traffic, traffic_goal):
    errors = traffic - traffic_goal
    overtr = abs(errors[errors > 0].sum())
    undertr = abs(errors[errors < 0].sum())
    
    overtr_c = 1
    undertr_c = 1
    
    cost = overtr_c * overtr + undertr_c * undertr
    return cost


# In[96]:


def generate_random_traffic(tbuses, days):
    route_nums = (7,10,11,12,16,24,25,29,32)
    cars = np.random.randint(1,)
    traffic = Traffic(tbuses = route_nums, days = days, cars = cars, people = people)
    return traffic


# In[73]:





# In[95]:





# In[ ]:


goal_cars = 1
goal_people = 1500
goal = np.array([[[r,goal_cars,goal_people] for r in route_nums] for _ in range(days)])
goal[0]

