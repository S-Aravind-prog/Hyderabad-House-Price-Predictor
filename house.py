import pandas as pd
import numpy as np

df = pd.read_csv('prices.csv')


# CLEANING THE DATA


location_counts = df['location'].value_counts()

valid_locations = location_counts[location_counts >= 10].index

df['location'] = df['location'].apply(
    lambda x : x if x in valid_locations else 'others')

#df['location']

df['location'] = df['location'].str.strip()
df['location'] = df['location'].str.title()




import numpy as np

# Extract BHK number
df['BHK'] = df['title'].str.extract(r'(\d+)')

# Convert to numeric
df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')

# Fill NaN (for Residential Plot etc.)
df['BHK'] = df['BHK'].fillna(0)

# Extract Property Type
df['PropertyType'] = df['title'].str.replace(r'^\d+\s*BHK\s*', '', regex=True)

# Handle plots separately
df.loc[df['title'] == 'Residential Plot', 'PropertyType'] = 'Plot'

df = df[df['PropertyType']!='Plot']

# Check result
print(df[['title', 'BHK', 'PropertyType']].head(20))

df['AreaPerRoom'] = np.where(df['BHK']>0,df['area_insqft'] / df['BHK'], 0)
print(df[['title','BHK','PropertyType']].head(20))

# HANDLING TO MANY OUTLIERS INOUR PROJECT


Q1 = df['price(L)'].quantile(0.25)
Q3 = df['price(L)'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Lower:", lower)
print("Upper:", upper)

outliers = df[df['price(L)'] > upper]
print("Outliers:", outliers.shape[0])

df_clean = df[df['price(L)'] <= upper]

X = df_clean[['location','PropertyType','building_status','BHK','area_insqft','AreaPerRoom']]
y = df_clean['price(L)']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

category_col = ['location','PropertyType','building_status']
num_col = ['BHK','area_insqft','AreaPerRoo']
trf1 = ColumnTransformer(
    transformers = [('trf1',
        OneHotEncoder(handle_unknown = 'ignore'),category_col),('num',StandardScaler(), num_col)])

pipe = Pipeline([
    ('trf1',trf1),
    ('model',LinearRegression())
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('MAE : ',mae)
print('MSE : ',mse)
print('RMSE : ',rmse)
print('R2 SCORE : ',r2)

compare = pd.DataFrame({"Actual Price":y_test,"Predicted Price":y_pred})
compare.head()

import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("HOUSE PRICES")
plt.show()

# EXPORT

import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))
model = pickle.load(open('pipe.pkl','rb'))