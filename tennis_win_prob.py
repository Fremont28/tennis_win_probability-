#4/2/18 tennis win probability model 
import pandas as pd 
import numpy as np 
pune=pd.read_csv("atp_18x.csv")
pune.info() 
pune.head(4)

#hot-encode variables
pune['surface_code'].value_counts() 
from sklearn.preprocessing import LabelEncoder
label_enc=LabelEncoder() 
pune['surface_code']=label_enc.fit_transform(pune["surface"])
pune['winner_hand']=label_enc.fit_transform(pune["winner_hand"])
#fill in missing values with mean 
pune['winner_ht']=pune['winner_ht'].fillna(pune['winner_ht'].mean())
pune['winner_age']=pune['winner_age'].fillna(pune['winner_age'].mean())
pune['loser_ht']=pune['loser_ht'].fillna(pune['loser_ht'].mean())
pune['winner_rank']=pune['winner_rank'].fillna(pune['winner_rank'].mean())
pune['loser_rank']=pune['loser_rank'].fillna(pune['loser_rank'].mean())
pune['loser_age']=pune['loser_age'].fillna(pune['loser_age'].mean())
pune['winner_hand']=pune['winner_hand'].fillna(pune['winner_hand'].mean())

#subset data
Y=pune["Win"].values
X=pune[["winner_ht","winner_age","winner_rank","loser_ht","loser_age","loser_rank","surface_code"]].values 
#split data for testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.28,random_state=90)

#random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=1000,random_state=0,max_depth=1,max_features="auto") 
#train the model on training data
rf_model.fit(X_train,Y_train)
predictions=rf_model.predict(X_test)
#absolute errors
errors=abs(predictions-Y_test)
errors 
round(np.mean(errors),2) #0.26 
#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(predictions,Y_test) #74.4% accuracy
rf_model.feature_importances_
rf_model.get_params(deep=True) 
features = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(train_x.columns, rf_model.feature_importances_):
    features[feature] = importance #add the name/value pair 
importances = pd.DataFrame.from_dict(features, orient='index').rename(columns={0: 'Gini-importance'})
importances 

#visualizations 
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns 

pune_viz=pd.read_csv("atp_matches_2018.csv")
pune.info()
#winner/loser age?
pune_viz.winner_age.mean() #27.17
pune_viz.loser_age.mean() #27.04
pune_viz['loser_age']=pune['loser_age'].fillna(pune['loser_age'].mean())
sns.distplot(pune_viz["winner_age"])
plt.show() 
sns.distplot(pune_viz["loser_age"])
plt.show()

#winner rank-loser rank diff 
pune_viz['age_diff']=pune_viz['winner_age']-pune_viz['loser_age']
pune_viz.age_diff.mean() #-0.0145 (young outplay the old)
sub_x=pune_viz[["tourney_name","age_diff"]]
age_diff=sub_x.groupby(['tourney_name']).mean() 
age_diff.to_csv("age_diff.csv")

# tournaments with the best compeition
statsx=pune_viz.groupby(['tourney_name']).mean() 
statsx.shape
statsx.columns.get_loc("winner_age")
statsx.reset_index(level=['tourney_name'])
winner_age_t=statsx[["winner_age"]]
test_sort=statsx.sort_values(['winner_rank'],ascending=[0])
test_sort1=statsx.sort_values(['loser_rank'],ascending=[0])


