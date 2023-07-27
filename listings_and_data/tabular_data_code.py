## The function of this file is to clean the data before it is transferred over
## to be used in the prediction modelling. Here, the basic cleaning methods will be used, 
## such as removing particular rows and combining lists into strings.

## The two imports used are Pandas- for the dataframe, and ast.literal_eval to skip over data 
## that is "invalid" for the cleaning but still needed in the dataframe
import pandas as pd
from ast import literal_eval

## This function is required to remove all the rows that have null value data in the specified 
## columns, and then return the updated dataframe.
def remove_rows_with_missing_ratings(df):
    df.dropna(axis='index', subset=['Cleanliness_rating', 'Accuracy_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace=True)
    return df

## This function will be later called in the combine description strings function. Essentially,
## the majority of the description columns are the string data type containing lists and we need 
# to convert these over to just strings. The literal_eval function will recognise if they are strings 
## and if not, raise an error. If an error is raised, we know the data is valid and we can just return 
## the original value and continue on.
def literal_eval_function(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

##The description column in the current dataframe is a list of strings. We need to combine these lists 
## into a singular string. To do so, we first drop null values, call the apply function for the 
## literal_eval function and then remove the rows with empty descriptions. After this we remove the 
## "about this space" section because it is not necessary for us to use, especially when every
## column has this prefix.
def combine_description_strings(df):
    df.dropna(subset='Description', inplace=True)
    df['Description'] = df['Description'].apply(lambda x: literal_eval_function(x))
    df['Description'] = [x for x in df['Description'] if x != ' ']
    df['Description'] = df['Description'].str.join(', ').str.replace('About this space, ', '')
    return df

## In some of the numeric columns, many rows have missing data. Instead of removing the data we just set the 
## values to 1. So the function, takes the two types of empty values we have- null values and empty columns(just space),
## and sets them to 1.
def set_default_feature_values(df):
    df[['guests', 'beds', 'bathrooms', 'bedrooms']] =  df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(value=1).replace(' ', 1)
    df['Category'] = df['Category'].astype('category')
    return df

##Rather than calling all the functions seperately, it makes sense to do so in one main function.
## So we place the functions defined earlier, and create a single argument which can be filtered through into each
## following function.
def clean_tabular_data(df):
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)
    return df

## The load airbnb function is to be used in the other file to load in the cleaned csv file. It will take in
## the file name, the column you wish to use as the label column and then the columns you wish to use for the 
## features columns. In our case, we will only be using the numerical columns in our dataframe for the features
## across all our models.
def load_airbnb(file, label_column, columns):
   df=pd.read_csv(file, usecols=columns)
   df.dropna(inplace=True)
   features=df.drop(label_column, axis = 1)
   labels=df[label_column]
   return (features, labels)
    

# %%
