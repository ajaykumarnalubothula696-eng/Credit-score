import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib

np.random.seed(42)

# Create dataset
data = pd.DataFrame({

'Income':np.random.randint(20000,150000,500),

'Monthly_Balance':np.random.randint(1000,80000,500),

'Credit_Cards':np.random.randint(1,10,500),

'Loan_Amount':np.random.randint(0,500000,500),

'Late_Payments':np.random.randint(0,15,500),

'Credit_Utilization':np.random.randint(10,100,500)

})

# Create score
data['Score'] = (

data['Income']*0.0001 +

data['Monthly_Balance']*0.0005 -

data['Loan_Amount']*0.00001 -

data['Late_Payments']*2 -

data['Credit_Utilization']*0.05 +

data['Credit_Cards']*2

)

# Credit band function
def credit_band(score):

    if score < 20:
        return 0

    elif score < 40:
        return 1

    elif score < 60:
        return 2

    else:
        return 3

data['Credit_Band'] = data['Score'].apply(credit_band)

# Features
X = data[['Income',
'Monthly_Balance',
'Credit_Cards',
'Loan_Amount',
'Late_Payments',
'Credit_Utilization']]

# Target
y = data['Credit_Band']

# Split
X_train,X_test,y_train,y_test = train_test_split(
X,y,
test_size=0.2,
random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# CREATE MODEL (THIS WAS YOUR ERROR)
model = RandomForestClassifier(n_estimators=200)

# TRAIN MODEL
model.fit(X_train,y_train)

# PREDICT
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)

print(classification_report(y_test,y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

plt.title("Confusion Matrix")

#plt.show()

# Test customer
new_customer = scaler.transform([[75000,25000,4,100000,2,40]])

prediction = model.predict(new_customer)

labels = ['Poor','Average','Good','Excellent']

print("Credit Band:",labels[prediction[0]])

# Save model
import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model,"credit_model.pkl")