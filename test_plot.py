import numpy as np
from test_linreg_univariate import plotData1D
from test_linreg_univariate import plotRegLine1D
from linreg import LinearRegression
filePath = "data/univariateData.dat"
file = open(filePath, 'r')
allData = np.loadtxt(file, delimiter=',')
X = np.matrix(allData[:,:-1])
y = np.matrix((allData[:,-1])).T
# g e t t h e number o f i n s t a n c e s ( n ) and number o f f e a t u r e s ( d )
n,d = X.shape
X = np.c_[np.ones((n,1)), X]
lr_model = LinearRegression(alpha = 0.01, n_iter = 5)
lr_model.fit(X,y)
plotRegLine1D(lr_model, X, y)