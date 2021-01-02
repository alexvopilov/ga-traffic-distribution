#!/usr/bin/env python
# coding: utf-8

# Популяция состоит из трафиков.  
# Трафик состоит из периодов.  
# Периоды включают в себя троллейбусы.  
# Троллейбусы говорят о кол-ве машин и людях.  

# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# In[2]:


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
def mutation(individual, mutations_count):
    size1 = individual.shape[0]
    size2 = individual.shape[1]
    for i in range(mutations_count):
        day = np.random.randint(0, size1)
        tbus = np.random.randint(0, size2)
        
        #Cars
        d = np.random.choice((-1,1))
        individual[day,tbus,1] += d
        
        #People
        d = np.random.choice(np.arange(-5,6))
        individual[day,tbus,2] += d
        
    return individual

def mutate(offspring):
    for i in range(0,len(offspring),2):
        offspring[i] = mutation(offspring[i].toarray(), 2)
        offspring[i] = Traffic.create(offspring[i],DAYS)
    return offspring

# In[73]:


def set_goal(days, goal_cars = 1, goal_people = 1500):
    return np.array([[[r,goal_cars,goal_people] for r in ROUTES] for _ in range(days)])


# In[17]:


THE_BEST = select_best(generation,goal)
pd.DataFrame(THE_BEST[0].toarray().reshape(-1,3)[:DAYS], columns=("ID","Cars","People"))


# In[19]:


plt.plot(acc_list)


# In[ ]:




