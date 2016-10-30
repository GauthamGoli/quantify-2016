import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ambhas.copula import Copula
from scipy import stats

input1 = pd.read_excel("Input_Data.xlsx", sheetname="Oil Call Option Prices")
input2 = pd.read_excel("Input_Data.xlsx", sheetname="FX Call Option Prices")
input3 = pd.read_excel("Input_Data.xlsx", sheetname="Joint_FX_Put")
input3 = pd.read_excel("Input_Data.xlsx", sheetname="Joint_Oil_Call")

x1=input1["Strike"].as_matrix()
y1=input1["Price"].as_matrix()

x2=input2["Strike"].as_matrix()
y2=input2["Price"].as_matrix()


fd1=np.gradient(y1)
fd2=np.gradient(y2)


sd1=np.gradient(fd1)
sd2=np.gradient(fd2)


# Figure 1
plt.plot(x1,sd1)
plt.xlabel('Price of Oil')
plt.ylabel('f($X_{Oil}$)')
plt.show()
#Figure 2
plt.plot(x2,sd2)
plt.xlabel('Price of FX')
plt.ylabel('f($X_{FX}$)')
plt.show()

# For Oil Digital Options
price = []
for K in range(30,71):
	temp = 0
	for i in np.nditer(x1):
		if i > K:
			index = np.where(x1==i)
			temp = temp + sd1[index]
	price.append(temp)
np.savetxt('Q1_1.csv',np.array(price))
temp = range(30,71)
# plt.plot(temp,price)
plt.show()

price = []
for K in range(20,106):
	temp = 0
	for i in np.nditer(x2):
		if i > K:
			index = np.where(x2==i)
			temp = temp + sd2[index]
	price.append(temp)
np.savetxt('Q1_2.csv',np.array(price))
temp = range(20,106)
plt.plot(temp, price)
plt.show()

# Oil Exotic Options
price = []
for K in range(30,71):
	temp = 0
	for i in np.nditer(x1):
		if i > K:
			index = np.where(x1==i)
			temp = temp + ((i-K)**2)*sd1[index]
	price.append(temp)
np.savetxt('Q2_1.csv',np.array(price))
temp = range(30,71)
plt.plot(temp, price)
plt.show()

# FX Exotic Options
price = []
for K in range(20,106):
	temp = 0
	for i in np.nditer(x2):
		if i > K:
			index = np.where(x2==i)
			temp = temp + ((i-K)**2)*sd2[index]
	price.append(temp)
np.savetxt('Q2_2.csv',np.array(price))
plt.plot(range(20,106),price)
plt.show()


xk1 = np.arange(len(list(sd1)))
pk1 = sd1
# Generating a random number distribution for Oil
custm1 = stats.rv_discrete(name='custm1', values=(xk1, pk1))

xk2 = np.arange(len(list(sd2)))
pk2 = sd2
# Generating a random number distribution for FX
custm2 = stats.rv_discrete(name='custm2', values=(xk2, pk2))

# Generating Random Numbers from the distributions
R1 = custm1.rvs(size=10000)
R2 = custm2.rvs(size=10000)

# function to generate copula from two sets of random numbers which follow the given marginal probability distribution
def genCopulas():
    fig = plt.figure()

    frank = Copula(R1,R2,family='frank')
    xf,yf = frank.generate_xy(500000)

    clayton = Copula(R1,R2,family='clayton')
    xc,yc = clayton.generate_xy(500000)

#   to return the random number pairs from frank copula
    return xf, yf
#   to return the random number pairs from clayton copula
#    return xc, yc

# Create a grid to calculate the joint distribution from generated random number pairs
m1, m2 = genCopulas()
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
# Using Gaussian Kernel Density Estimator
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

# Verifying that the obtained joint distribution is adequate
# Comparing with Actual Marginal obtained from Question1 
fd1=np.gradient(X.T[0])
fd2=np.gradient(Y[0])
x_list = []
y_list = []
for i in range(100):
    temp_x = 25 + X[i][0]
    temp_y =sum(Z[i])*fd2[0]
    x_list.append(temp_x)
    y_list.append(temp_y)
plt.plot(x_list,y_list, label = 'Estimated Marginal')
plt.plot(x1,sd1, label = 'Actual Marginal')
plt.ylabel("f($X_1$)")
plt.xlabel("Price of Oil ($X_1$)")
plt.legend()
plt.show()

fd1=np.gradient(X.T[0])
fd2=np.gradient(Y[0])
x_list = []
y_list = []
for i in range(100):
    temp_x = 15 + Y[0][i]
    temp_y =sum(Z.T[i])*fd1[0]
    x_list.append(temp_x)
    y_list.append(temp_y)
plt.plot(x_list,y_list, label = 'Estimated Marginal')
plt.plot(x2,sd2, label = 'Actual Marginal')
plt.ylabel("f($X_2$)")
plt.xlabel("Price of FX ($X_2$)")
plt.legend()
plt.show()

# for 'Q2'
B1 = [35, 41, 47, 53, 59, 65]
pred = []
for k in B1:
    sum2 = 0
    for j in range(100):
        sum1 = 0
        for i in range(100):
            if (25+X[i][0]) > k:
                sum1 = sum1 + (25+X[i][0]-k)*Z[i][j]
        sum2 = sum2 + (15+Y[0][j])*sum1
    sum3 = sum2*fd1[0]*fd2[0]
    pred.append(sum3)

actual = [912.104648, 591.928507, 309.753731, 115.46706, 27.091061, 3.655863]
plt.plot(B1,actual, label = 'Actual Joint_Oil_Call')
plt.plot(B1,pred, label = 'Estimated Joint_Oil_Call')
plt.legend()
plt.show()

# for 'Q1'
B2 = [30, 40, 50, 60, 70, 80]
pred = []
for k in B2:
    sum2 = 0
    for j in range(100):
        sum1 = 0
        for i in range(100):
            if (15+Y[0][i]) < k:
                sum1 = sum1 + (k-(15+Y[0][i]))*Z[j][i]
        sum2 = sum2 + (25+X[j][0])*sum1
    sum3 = sum2*fd1[0]*fd2[0]
    pred.append(sum3)
actual = [4.640858, 59.718679, 235.426702, 493.174062, 814.620805, 1214.109622]
plt.plot(B2,actual,  label = 'Actual Joint_FX_Put')
plt.plot(B2,pred,  label = 'Estimated Joint_FX_Put')
plt.legend()
plt.show()

# Final Estimation of OilCall_FXPut
B1 = [35, 39, 43, 47, 51, 55, 59, 63, 67]
B2 = 90
fname = 'temp_90.txt'
pred = []
for k in B1:
    sum2 = 0
    for j in range(100):
        if (15+Y[0][j])<B2:
            sum1 = 0
            for i in range(100):
                if (25+X[i][0]) > k:
                    sum1 = sum1 + (25+X[i][0]-k)*Z[i][j]
            sum2 = sum2 + (B2 - (15+Y[0][j]))*sum1
    sum3 = sum2*fd1[0]*fd2[0]
    pred.append(sum3)
np.savetxt(fname,pred)
# plt.plot(pred)
# plt.show()
