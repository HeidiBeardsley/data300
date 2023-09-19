1. In your own words, describe:

a. Bias and variance in machine learning.

Bias is the difference between expected and predicted y's. Bias creates consistent error in the model. Variance refers to the amount of change observed in the predicted values when we change the training set, or the
variability of the predictions. 

b. Describe the bias-variance tradeoff.

If the model has high variance, it tends to have low bias, which refers to overfitting the model. If the model has low variance, it tends to have high bias, which refers to underfitting the model. 

c. In 3 or less sentences, describe bias and variance in relation to complex and simple models. When is variance high? When is it low? How about bias?

If the model is simple and less flexible, the variance tends to be low and bias is high. If the model is flexible and complicated, the bias tends to be low and variance is high. 

2. List the assumptions of linear regression.

- There is a linear relationship between the predictor and predicted variables.
- The observations are independent.
- The variance of the errors is consistent across all levels of the dependent variable.
- The errors are normally distributed.
- There is no multicollinearity.

3. The following regression equation shows the effects of biking and smoking on heart disease. Interpret the coefficients in the model. 

For every 1 unit change in biking, there is a -0.2 unit change in heart disease, meaning they are negatively correlated. For every 1 unit change in smoking, there is a 0.178 unit change in heart disease, meaning they are positively correlated.

4. The following is the R output of a multiple linear regression model. We want to estimate sales as a function of advertising budget invested in different media: YouTube, Facebook, and Newspapers.

a. Write down the equation for the above regression.

y = 3.52667 + 0.04567youtube + 0.18853facebook - 0.00104newspaper

b. Interpret the above output focusing on the residuals, the coefficients, the standard error, the t value, the p value, R squared and the F statistic.

residuals- Residuals are the difference between the observed and predicted y, so the min, 1Q, median, 3Q, and max show the spread of all the residuals in the model. Since the min is -10.59 and the max is 3.40, this could suggest skewed data, and that the mean is not equal to the median.

coefficients- For every 1 unit increase in youtube advertising budget, there is a 0.04567 unit increase in sales. For every 1 unit increase in facebook advertising budget, there is a 0.18853 unit increase in sales. For every 1 unit increase in newspaper advertising budget, there is a 0.00104 unit decrease in sales. 

standard error- Standard error measures the average amount an estimate deviates from the actual value. The standard error values are relatively small, so that means the values are not very spread out and/or there are a lot of sample units for each x value.

t value- The t-value helps us determine if the data are statistically significant, which gives us the value for Pr(>|t|). Since Pr(>|t|)<0.05 for all values except newpaper, those values are significant. However, the value for newpaper is insignificant. 

p value- P-value is the probability of observing the evidence under the null assumption, and since the p-value is small, the data is statistically significant enough to reject the null hypothesis. At least one of the variables has a statistically significant relationship with sales.

R squared- R squared is the portion of the y explained by the x's, so 89.7% of variation in sales can be explained by the linear regression model.

F statistic- The F statistic shows the global fit of the model. This value shows that all of the x variables can explain the y variable.


```python
# Python version 3.9.12, conda version 22.9.0
```


```python
import numpy as np # v 1.21.5
import sklearn # v 0.0
import pandas as pd # v 1.4.2
import ydata_profiling as pp # v 4.5.1
import matplotlib.pyplot as plt # v 3.5.1
import statsmodels.api as sm # v.0.13.2

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


%matplotlib inline
```


```python

```

EDA


```python
ca_houses = pd.read_csv("/Users/heidibeardsley/Downloads/California_Houses.csv")
```


```python
ca_houses.shape, ca_houses.columns, ca_houses.dtypes
```




    ((20640, 14),
     Index(['Median_House_Value', 'Median_Income', 'Median_Age', 'Tot_Rooms',
            'Tot_Bedrooms', 'Population', 'Households', 'Latitude', 'Longitude',
            'Distance_to_coast', 'Distance_to_LA', 'Distance_to_SanDiego',
            'Distance_to_SanJose', 'Distance_to_SanFrancisco'],
           dtype='object'),
     Median_House_Value          float64
     Median_Income               float64
     Median_Age                    int64
     Tot_Rooms                     int64
     Tot_Bedrooms                  int64
     Population                    int64
     Households                    int64
     Latitude                    float64
     Longitude                   float64
     Distance_to_coast           float64
     Distance_to_LA              float64
     Distance_to_SanDiego        float64
     Distance_to_SanJose         float64
     Distance_to_SanFrancisco    float64
     dtype: object)




```python
ca_houses.isna().sum()

# no missing values
```




    Median_House_Value          0
    Median_Income               0
    Median_Age                  0
    Tot_Rooms                   0
    Tot_Bedrooms                0
    Population                  0
    Households                  0
    Latitude                    0
    Longitude                   0
    Distance_to_coast           0
    Distance_to_LA              0
    Distance_to_SanDiego        0
    Distance_to_SanJose         0
    Distance_to_SanFrancisco    0
    dtype: int64




```python
ca_houses.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Median_House_Value</th>
      <th>Median_Income</th>
      <th>Median_Age</th>
      <th>Tot_Rooms</th>
      <th>Tot_Bedrooms</th>
      <th>Population</th>
      <th>Households</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Distance_to_coast</th>
      <th>Distance_to_LA</th>
      <th>Distance_to_SanDiego</th>
      <th>Distance_to_SanJose</th>
      <th>Distance_to_SanFrancisco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>452600.0</td>
      <td>8.3252</td>
      <td>41</td>
      <td>880</td>
      <td>129</td>
      <td>322</td>
      <td>126</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>9263.040773</td>
      <td>556529.158342</td>
      <td>735501.806984</td>
      <td>67432.517001</td>
      <td>21250.213767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>358500.0</td>
      <td>8.3014</td>
      <td>21</td>
      <td>7099</td>
      <td>1106</td>
      <td>2401</td>
      <td>1138</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>10225.733072</td>
      <td>554279.850069</td>
      <td>733236.884360</td>
      <td>65049.908574</td>
      <td>20880.600400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>352100.0</td>
      <td>7.2574</td>
      <td>52</td>
      <td>1467</td>
      <td>190</td>
      <td>496</td>
      <td>177</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>8259.085109</td>
      <td>554610.717069</td>
      <td>733525.682937</td>
      <td>64867.289833</td>
      <td>18811.487450</td>
    </tr>
    <tr>
      <th>3</th>
      <td>341300.0</td>
      <td>5.6431</td>
      <td>52</td>
      <td>1274</td>
      <td>235</td>
      <td>558</td>
      <td>219</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>7768.086571</td>
      <td>555194.266086</td>
      <td>734095.290744</td>
      <td>65287.138412</td>
      <td>18031.047568</td>
    </tr>
    <tr>
      <th>4</th>
      <td>342200.0</td>
      <td>3.8462</td>
      <td>52</td>
      <td>1627</td>
      <td>280</td>
      <td>565</td>
      <td>259</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>7768.086571</td>
      <td>555194.266086</td>
      <td>734095.290744</td>
      <td>65287.138412</td>
      <td>18031.047568</td>
    </tr>
  </tbody>
</table>
</div>




```python
ca_houses.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Median_House_Value</th>
      <th>Median_Income</th>
      <th>Median_Age</th>
      <th>Tot_Rooms</th>
      <th>Tot_Bedrooms</th>
      <th>Population</th>
      <th>Households</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Distance_to_coast</th>
      <th>Distance_to_LA</th>
      <th>Distance_to_SanDiego</th>
      <th>Distance_to_SanJose</th>
      <th>Distance_to_SanFrancisco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20635</th>
      <td>78100.0</td>
      <td>1.5603</td>
      <td>25</td>
      <td>1665</td>
      <td>374</td>
      <td>845</td>
      <td>330</td>
      <td>39.48</td>
      <td>-121.09</td>
      <td>162031.481121</td>
      <td>654530.186299</td>
      <td>830631.543047</td>
      <td>248510.058162</td>
      <td>222619.890417</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>77100.0</td>
      <td>2.5568</td>
      <td>18</td>
      <td>697</td>
      <td>150</td>
      <td>356</td>
      <td>114</td>
      <td>39.49</td>
      <td>-121.21</td>
      <td>160445.433537</td>
      <td>659747.068444</td>
      <td>836245.915229</td>
      <td>246849.888948</td>
      <td>218314.424634</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>92300.0</td>
      <td>1.7000</td>
      <td>17</td>
      <td>2254</td>
      <td>485</td>
      <td>1007</td>
      <td>433</td>
      <td>39.43</td>
      <td>-121.22</td>
      <td>153754.341182</td>
      <td>654042.214020</td>
      <td>830699.573163</td>
      <td>240172.220489</td>
      <td>212097.936232</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>84700.0</td>
      <td>1.8672</td>
      <td>18</td>
      <td>1860</td>
      <td>409</td>
      <td>741</td>
      <td>349</td>
      <td>39.43</td>
      <td>-121.32</td>
      <td>152005.022239</td>
      <td>657698.007703</td>
      <td>834672.461887</td>
      <td>238193.865909</td>
      <td>207923.199166</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>89400.0</td>
      <td>2.3886</td>
      <td>16</td>
      <td>2785</td>
      <td>616</td>
      <td>1387</td>
      <td>530</td>
      <td>39.37</td>
      <td>-121.24</td>
      <td>146866.196892</td>
      <td>648723.337126</td>
      <td>825569.179028</td>
      <td>233282.769063</td>
      <td>205473.376575</td>
    </tr>
  </tbody>
</table>
</div>




```python
ca_houses.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Median_House_Value</th>
      <th>Median_Income</th>
      <th>Median_Age</th>
      <th>Tot_Rooms</th>
      <th>Tot_Bedrooms</th>
      <th>Population</th>
      <th>Households</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Distance_to_coast</th>
      <th>Distance_to_LA</th>
      <th>Distance_to_SanDiego</th>
      <th>Distance_to_SanJose</th>
      <th>Distance_to_SanFrancisco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>2.064000e+04</td>
      <td>2.064000e+04</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>206855.816909</td>
      <td>3.870671</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.898014</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>35.631861</td>
      <td>-119.569704</td>
      <td>40509.264883</td>
      <td>2.694220e+05</td>
      <td>3.981649e+05</td>
      <td>349187.551219</td>
      <td>386688.422291</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115395.615874</td>
      <td>1.899822</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.247906</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>2.135952</td>
      <td>2.003532</td>
      <td>49140.039160</td>
      <td>2.477324e+05</td>
      <td>2.894006e+05</td>
      <td>217149.875026</td>
      <td>250122.192316</td>
    </tr>
    <tr>
      <th>min</th>
      <td>14999.000000</td>
      <td>0.499900</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>32.540000</td>
      <td>-124.350000</td>
      <td>120.676447</td>
      <td>4.205891e+02</td>
      <td>4.849180e+02</td>
      <td>569.448118</td>
      <td>456.141313</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>119600.000000</td>
      <td>2.563400</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>295.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>33.930000</td>
      <td>-121.800000</td>
      <td>9079.756762</td>
      <td>3.211125e+04</td>
      <td>1.594264e+05</td>
      <td>113119.928682</td>
      <td>117395.477505</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>179700.000000</td>
      <td>3.534800</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>34.260000</td>
      <td>-118.490000</td>
      <td>20522.019101</td>
      <td>1.736675e+05</td>
      <td>2.147398e+05</td>
      <td>459758.877000</td>
      <td>526546.661701</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>264725.000000</td>
      <td>4.743250</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>37.710000</td>
      <td>-118.010000</td>
      <td>49830.414479</td>
      <td>5.271562e+05</td>
      <td>7.057954e+05</td>
      <td>516946.490963</td>
      <td>584552.007907</td>
    </tr>
    <tr>
      <th>max</th>
      <td>500001.000000</td>
      <td>15.000100</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>41.950000</td>
      <td>-114.310000</td>
      <td>333804.686371</td>
      <td>1.018260e+06</td>
      <td>1.196919e+06</td>
      <td>836762.678210</td>
      <td>903627.663298</td>
    </tr>
  </tbody>
</table>
</div>




```python
ca_houses.plot(subplots=True)
plt.show()

# Many variables in the dataset seem to be correlated with one another.
```


    
![png](output_14_0.png)
    



    <Figure size 259200x181440 with 0 Axes>



```python
pp.ProfileReport(ca_houses)

# always crashes the kernel, unsure why.
```


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]


Machine Learning


```python
y = ca_houses.Median_House_Value
```


```python
x = ca_houses.loc[:, ca_houses.columns != "Median_House_Value"]
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
```


```python
model = LinearRegression()
```


```python
model.fit(x_train, y_train)
```




    LinearRegression()




```python
model.coef_
```




    array([ 3.92350456e+04,  8.57173901e+02, -6.27518202e+00,  9.89201998e+01,
           -3.98576147e+01,  5.54067767e+01, -4.56185190e+04, -2.70359518e+04,
           -2.25575153e-01, -1.45160265e-01,  2.50801184e-01,  1.66833378e-01,
           -1.41222782e-01])




```python
print(model.intercept_, model.coef_,model.score(x_test, y_test))
```

    -1639814.9702004073 [ 3.92350456e+04  8.57173901e+02 -6.27518202e+00  9.89201998e+01
     -3.98576147e+01  5.54067767e+01 -4.56185190e+04 -2.70359518e+04
     -2.25575153e-01 -1.45160265e-01  2.50801184e-01  1.66833378e-01
     -1.41222782e-01] 0.6540306373366478


R^2 = 0.654. This is an acceptable R squared value and is generally a good fit. 65.4% of the variability is explained by the model.


```python
y_pred = model.predict(x_test)
```


```python
y_pred
```




    array([138347.62072032, 210482.3339375 , 119767.88728811, ...,
           224302.620426  , 209270.37819201,  95918.22986394])




```python
MAE = mean_absolute_error(y_test,y_pred)
```


```python
MSE = mean_squared_error(y_test,y_pred)
```


```python
MAPE =  mean_absolute_percentage_error(y_test,y_pred)
```


```python
MSE, MAE,MAPE
```




    (4773284091.24259, 50439.7858327504, 0.29162671574513394)


