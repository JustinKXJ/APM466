#!/usr/bin/env python
# coding: utf-8

# In[579]:


import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
from datetime import datetime
from dateutil.relativedelta import relativedelta


# In[580]:


bonds = pd.read_csv('11_bonds.csv')


# 4.(a)

# In[583]:


def ytm(clean_price, coupon, initial_date, maturity_date):
    initial_time = datetime.strptime(initial_date,"%Y/%m/%d")
    maturity_time = datetime.strptime(maturity_date,"%Y/%m/%d")
    time_interval = (maturity_time - initial_time).days

    # Separate the time to maturity to the small time interval.
    p,q = int(time_interval/182), time_interval %182
    initial = q/365
    last_payment_time = initial_time + relativedelta(days=+q) - relativedelta(months=+6)
    
    # Convert the clean, close price to dirty price.
    dirty_price = clean_price + (initial_time - last_payment_time).days/365 * coupon *100
    times = np.asarray([2* initial + n for n in range(0,p+1)])
    
    # Make each payment in different periods as an array.
    payment_different_periods = np.asarray([coupon /2 * 100] * p + [coupon /2 * 100 + 100])
    
    # Use Newtonian optimization to solve the yield to maturity.
    F = lambda x: np.dot(payment_different_periods, (1+x/2)**(-times)) - dirty_price
    return scipy.optimize.newton(F,0.05),dirty_price


# In[591]:


# Calculate 10 selected bonds' yields and plot the corresponding yield curve

dates = ['2020/01/02','2020/01/03','2020/01/06','2020/01/07','2020/01/08','2020/01/09','2020/01/10',
         '2020/01/13','2020/01/14','2020/01/15']

dirty_prices = np.empty([10,11])

yields_to_maturities = np.empty([10,11])

for i in range(0,10):
    maturities = []
    for j in range(0,11):
        all_yields = ytm(float(bonds.iloc[j,i+4]), float(bonds.iloc[j,0]),dates[i],bonds.iloc[j,3])
        yields_to_maturities[i][j], dirty_prices[i][j] = all_yields[0],all_yields[1]
        maturities.append((datetime.strptime(bonds.iloc[j,3],"%Y/%m/%d")- datetime.strptime(dates[i],"%Y/%m/%d")).days/365)
    plt.plot(maturities, yields[i], label = dates[i])   

    
plt.xticks(np.arange(0,5.1))
plt.xlabel('maturity (year)')
plt.ylabel('yield')
plt.legend()
plt.show()


# In[544]:


print("Yield of 10 bonds is {}".format(yields_to_maturities))


# 4.(b)

# In[545]:


# Define two empty matrices to store values later

spot_rates = np.empty([10,11])
maturities = np.empty([10,11])

for i in range(0,10):
    for j in range(0,11):
        maturities[i][j] = (datetime.strptime(df.iloc[j,3],"%Y/%m/%d") - datetime.strptime(dates[i],"%Y/%m/%d")).days/365
        # First calculate the initial spot rate.
        if j == 0:
            spot_rates[i][j] = -np.log(dirty_prices[i][j]/(100+ 100*float(df.iloc[j,0])/2)) /(maturities[i][j])
        else:
        # Considering the cases of the 6th and the 8th bond where we have irregular months of bonds, which divide the time span of bonds into two parts.
            if j!= 5 and j != 7:
                payments = np.asarray([100 *float(bonds.iloc[j,0])/2] * j + [100 + 100*float(bonds.iloc[j,0])/2])
            else:
                payments = np.asarray([100 *float(bonds.iloc[j,0])/2] * (j-1) + [100 + 100*float(bonds.iloc[j,0])/2])
        # Considering the spot rates of the first five bonds
            if j >= 1 and j <= 4:
                rate = spot_rates[i,0:j]
                time = maturities[i,0:j]
        # Considering the spot rates of the rest of bonds (i.e. the 6th,7th,8th,9th,10th,11th bonds)
            elif j == 5:
                time = np.asarray([maturities[i][n] + 1/4 for n in range(0,4)])
                rate = np.asarray([(spot_rates[i][n] + spot_rates[i][n+1])/2 for n in range(0,4)])
            elif j == 6:
                time = np.asarray([maturities[i][n] for n in range(0,5)] + [maturities[i][5] + 1/4])
                rate = np.asarray([spot_rates[i][n] for n in range(0,5)] + [2* spot_rates[i][5] - spot_rates[i][4]])    
            elif j == 7:
                time = np.asarray([maturities[i][n] + 1/4 for n in range(0,5)] +[maturities[i][5] + 1/2])
                rate = np.asarray([(spot_rates[i][n] + spot_rates[i][n+1])/2 for n in range(0,4)] + [spot_rates[i][5]] +                       [3*spot_rates[i][5] - 2*spot_rates[i][4]])
            elif j == 8:
                time = np.asarray([maturities[i][n] for n in range(0,5)] +[maturities[i][5] +1/4] + [maturities[i][6]] +                                 [maturities[i][7] +1/4])
                rate = np.asarray([spot_rates[i][n] for n in range(0,5)] + [2* spot_rates[i][5] - spot_rates[i][4]] +                                 [spot_rates[i][6]] +[2* spot_rates[i][7] - spot_rates[i][6]])
            elif j == 9:
                time = np.asarray([maturities[i][n] for n in range(0,5)] +[maturities[i][5] +1/4] + [maturities[i][6]] +                                 [maturities[i][7] +1/4] +[maturities[i][8]])
                rate = np.asarray([spot_rates[i][n] for n in range(0,5)] + [2* spot_rates[i][5] - spot_rates[i][4]] +                                 [spot_rates[i][6]] +[2* spot_rates[i][7] - spot_rates[i][6]] + [spot_rates[i][8]])
            elif j == 10:
                time = np.asarray([maturities[i][n] for n in range(0,5)] +[maturities[i][5] +1/4] + [maturities[i][6]] +                                 [maturities[i][7] +1/4] +[maturities[i][8],maturities[i][9]])
                rate = np.asarray([spot_rates[i][n] for n in range(0,5)] + [2* spot_rates[i][5] - spot_rates[i][4]] +                                 [spot_rates[i][6]] +[2* spot_rates[i][7] - spot_rates[i][6]] + [spot_rates[i][8]] +                                 [spot_rates[i][9]])
            forward_raw = lambda x: np.dot(payments[0:-1], np.exp(-(np.multiply(rate,time)))) + payments[-1] * np.exp(-x* maturities[i][j]) - dirty_prices[i][j]
            # Solve the spot rate using optimization
            spot_rates[i][j] = scipy.optimize.newton(forward_raw,0.05)
    plt.plot(maturities[i],spot_rates[i],label = dates[i])
plt.xticks(np.arange(0,6))
plt.xlabel('maturities')
plt.ylabel('spot rate')
plt.legend()
plt.show()


# 4.(c)

# In[546]:


forward_rates = np.empty([10,4])
m = np.empty([10,5])
for i in range(0,10):
    for j in range(0,5):
        a = maturities[i, (2*j+1):(2*j+3)]
        b = spot_rates[i, (2*j+1):(2*j+3)]
        c = np.polyfit(a,b,1)
        m[i][j] = np.polyval(c,j+1)
for i in range(0,10):
    for j in range(0,4):
        forward_rates[i][j] = ((1 + m[i][j+1])**(j+2)/(1 + m[i][0]))** (1/(j+1)) -1
    plt.plot([1,2,3,4], forward_rates[i], label = dates[i])
plt.xlabel('time')
plt.ylabel('forward rates')
plt.xticks(ticks = [1,2,3,4],labels = ['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'])
plt.legend(loc = "upper right", prop={"size" :8})
plt.show()


# 5.

# In[566]:


ytm_matrix = np.empty([10,5])
for i in range(0,10):
    for j in range(0,5):
        a = maturities[i, (2*j+1):(2*j+3)]
        b = yields_to_maturities[i, (2*j+1):(2*j+3)]
        c = np.polyfit(a,b,1)
        yield_matrix[i][j] = np.polyval(c,j+1)
        
forward_matrix = np.empty([5,9])
for i in range(0,5):
    x = yield_matrix[1:10,i] 
    y = yield_matrix[0:9,i]
    forward_matrix[i,:] = np.log(x / y)
cov1 = np.cov(forward_matrix)
cov2 = np.cov(forward_rates.T)


# In[567]:


print("covariance of YTM is {}".format(cov1))


# In[568]:


print("covariance of forward rate is {}".format(cov2))


# 6.

# In[557]:


# eigenvalues and eigenvectors of covariance matrix of log returns of yield:

w,v = np.linalg.eig(log_return_covariance)
print("eigen 1 is {}\n{}".format(w,v))


# In[558]:


# eigenvalues and eigenvectors of covariance matrix of forward rates:

x,y = np.linalg.eig(forward_covariance)
print("eigen 2 is {}\n{}".format(x, y))


# In[ ]:


# Calculate the weights:


# In[588]:


weight1 = 8.48617389e-04/(8.48617389e-04+6.41662387e-05+4.33060806e-05+2.92760812e-06+1.94221422e-05)
print(weight1)


# In[590]:


weight2 = 2.46520906e-07 / (2.46520906e-07 + 2.64908883e-08 + 3.34826113e-09 + 1.15893562e-08)
print(weight2)


# In[ ]:




