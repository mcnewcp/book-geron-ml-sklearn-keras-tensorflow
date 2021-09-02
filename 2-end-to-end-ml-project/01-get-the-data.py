# %%
import os
import tarfile
import urllib

#%%
#some paths to use in downloading dataset
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#function for downloading data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#%%
#does the data need to be downloaded?
download_data = False

#download data if needed
if download_data:
    fetch_housing_data()

#%%
#load data csv
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# %%
#take a look at the df
housing.head()
housing.info()
#all features are floats except for ocean_proximity
#total_bedrooms has some missing values

# %%
#take a look at ocean_proximity feature
housing["ocean_proximity"].value_counts()

#%%
#use describe to take a look at all numerical features
housing.describe()

# %%
#use matplotlib to plot histograms
# only use this in ipynb: %matplotlib inline 
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

### observations from histograms
###     median income is not in USD, ~10s of thousands
###     housing_median_age & median_house_value were capped
###     features have very different scales
###     many histograms are tail-heavy

# %%
#now to split into training and test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# %%
import numpy as np
#or if we want to do stratified splitting instead
#for example median_income is likely a very important feature
#looking at the histogram, most values are 1.5 - 6.0, 
#so defining income categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

#and taking a quick look at the bins
housing["income_cat"].hist()

# %%
#now using stratified sampling from sklearn
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# %%
#quick check to see if it worked
print("test ratios:", strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print("train ratios:", strat_train_set["income_cat"].value_counts()/len(strat_train_set))

# %%
#dropping income_cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
