#%%
import pandas as pd
from ast import literal_eval
import csv

def remove_rows_with_missing_ratings(df):
    df.dropna(axis='index', subset=['Cleanliness_rating', 'Accuracy_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)
    return df

def literal_eval_function(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

def combine_description_strings(df):
    df.dropna(subset='Description', inplace=True)
    df['Description'] = df['Description'].apply(lambda x: literal_eval_function(x))
    df['Description'] = [x for x in df['Description'] if x != ' ']
    df['Description'] = df['Description'].str.join(', ').str.replace('About this space, ', '')
    return df

def set_default_feature_values(df):
    df[['guests', 'beds', 'bathrooms', 'bedrooms']] =  df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(value=1).replace(' ', 1)
    return df

def clean_tabular_data(df):
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)
    return df

def load_airbnb(file):
   df=pd.read_csv(file, usecols=['beds', 'bathrooms', 'Price_Night', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count' ], delimiter=',')
   df.dropna(inplace=True)
   features=df.loc[:, df.columns != 'Price_Night']
   labels=df['Price_Night']
   return (features, labels)

load_airbnb('clean_tabular_data.csv')

# if __name__ == "__main__":
#     df = pd.read_csv('listing.csv')
#     df2=clean_tabular_data(df)
#     df2.to_csv('clean_tabular_data.csv')
    

# %%
