# %%
#create a copy of training set to explore
housing = strat_train_set.copy()

# %%
#take a look at the data geographically 
housing.plot(kind="scatter", x="longitude", y="latitude")

# %%
#all I see is Ca, setting transparency lower lets me see density
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# %%
#now adding some more variables to the plot
### point size (s) is population
### point color (c) is house value, target variable
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()
#looks like housing prices are very much related to:
### location, e.g. close to ocean
### population density,
###     a clustering algorithm could be used for detecting main clusters and 
###     defining a new feature of distance to main clusters

# %%
#running a correlation test between all variables
corr_matrix = housing.corr()
#and checking correlation results for price
corr_matrix["median_house_value"].sort_values(ascending=False)

# %%
#now plotting scatter plot for a few promising features
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# %%
#taking a closer look at median_income correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#the price cap at 500,000 is clearly visible here
#there are a couple other horizontal lines at 450,000, 350,000 etc
#these districts may need to be removed before training a model

# %%
#now trying a couple feature combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
#nice!  bedrooms_per_room is much more correlated with median house value than total number of rooms or bedrooms
#rooms per household is also more informative than total number of rooms
