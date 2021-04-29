import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


df = pd.read_csv('./ECData.csv')
df = df[df.country == "United States"]
df = df[df.year > 1953]

# CORRELATION MATRIX
# corr_matrix = df.corr()
# sn.heatmap(corr_matrix)
# plt.show()


features = [
            'emp', 'avh','ccon', 'cda', 'cgdpe', 'cgdpo', 'cn', 
            'rgdpna', 'rconna', 'rdana', 'rnna', 'rkna', 'rtfpna',
            'rwtfpna', 'labsh', 'delta', 'pl_con', 'pl_da', 'pl_gdpo',
            'csh_c', 'csh_x', 'csh_m', 'csh_r', 'pl_c', 'pl_i',
            'pl_g', 'pl_x', 'pl_m', 'pl_n'
            ]

target = 'rgdpo'
# target = 'rgdpe'

x = df.loc[:, features]
x = np.array(x)
y = df.loc[:, [target]]
y = np.array(y)


train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size = 0.25)


rf = RandomForestRegressor(n_estimators = 1000)
# rf = LinearRegression()
rf.fit(train_features, train_labels.ravel())


predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2))

mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# pred = rf.predict([[300, 1765.34639, 1.68262360e+07,  2.13835520e+07,  2.07913640e+07,  2.05660340e+07,
#   6.90590960e+07,  2.05635920e+07,  1.68031520e+07,  2.14365900e+07,
#   6.90590720e+07,  1.04588222e+00,  1.01686203e+00,  1.01968193e+00,
#   5.97091138e-01,  4.59649940e-02,  1.04239917e+00,  1.03087151e+00,
#   1.04216623e+00,  7.03201294e-01,  1.11489668e-01, -1.65831953e-01,
#   1.45913630e-02,  1.00570726e+00,  9.88309801e-01,  1.26685011e+00,
#   7.17117965e-01,  7.52817988e-01,  1.06946027e+00]])
# pred = rf.predict([[329.064917, 1000, 1.68262360e+07,  2.13835520e+07,  2.07913640e+07,  2.05660340e+07,
#   6.90590960e+07,  2.05635920e+07,  1.68031520e+07,  2.14365900e+07,
#   6.90590720e+07,  1.04588222e+00,  1.01686203e+00,  1.01968193e+00,
#   5.97091138e-01,  4.59649940e-02,  1.04239917e+00,  1.03087151e+00,
#   1.04216623e+00,  7.03201294e-01,  1.11489668e-01, -1.65831953e-01,
#   1.45913630e-02,  1.00570726e+00,  9.88309801e-01,  1.26685011e+00,
#   7.17117965e-01,  7.52817988e-01,  1.06946027e+00]])

print(pred)

