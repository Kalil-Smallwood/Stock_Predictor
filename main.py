import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period='max')#Gets history on sp500

del sp500['Dividends']#Not used for general, used for specific stocks
del sp500['Stock Splits']

#Setting the target

sp500['Tmrw'] = sp500['Close'].shift(-1)
sp500['Target'] = (sp500['Tmrw'] > sp500['Close']).astype(int)


#Getting rid of old data

sp500 =sp500.loc['1990-01-01':].copy()

#Using Random Forest Model

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
#n_estimators = number of decision trees
#mine_sample_split = protect against overfitting

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])#Using predictor columns to try to predict Target

#Checking accuracy of model

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

precision_score(test['Target'], preds)#Gives accuracy

#Plot predictions

combined =pd.concat([test['Target'], preds], axis=1)
combined.plot()