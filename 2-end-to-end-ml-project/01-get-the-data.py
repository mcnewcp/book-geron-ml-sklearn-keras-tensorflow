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