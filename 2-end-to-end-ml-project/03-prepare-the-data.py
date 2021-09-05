#%%
#fresh copy of training dataset
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# %%
#handling missing values in total_bedrooms
#to drop all observations with missing values
#housing.dropna(subset=["total_bedrooms"]) 
#to drop the feature entirely
#housing.drop("total_bedrooms", axis=1)
#to impute missing values with median
#median = housing["total_bedrooms"].median()
#housing["total_bedrooms"].fillna(median, inplace=True)

#or using simpleimputer from sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

#this can only be applied to numerical attributes
housing_num = housing.drop("ocean_proximity", axis=1)

#%%
#now to fit the imputer to the training data
imputer.fit(housing_num)
#only the total_bedrooms varialbe had missing values but we can't be sure that 
#   will be true for new data so it's safer to fit all features
imputer.statistics_

# %%
#now to transform the numeric values
X = imputer.transform(housing_num)
#and if I want to put it back into a df
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# %%

