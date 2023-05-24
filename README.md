# airbnb-property-listing-model

Milestone 3 consists of cleaning and formatting the data correctly ready to use in the models and networks. The first step was creating three functions: one which removed the rows in the dataframe with missing values, one combined the description column to one string per column, rather than multiple lists, and the final one added features in to the null values in some of the columns, e.g NaN values in beds were made to equal 1.
![image](https://user-images.githubusercontent.com/116648304/231770992-735e4fa1-6f0c-4e99-98fc-63d969aec74e.png)
The literal_eval function was designed to skip past rows in the description column that already met the condtion I required. 
The final function I designed placed all these functions into one to make the code easier to run, these then ran sequentially to provide a final cleaned dataframe that I would upload as a csv. 
![image](https://user-images.githubusercontent.com/116648304/231771695-57cc8e4f-3c05-4b03-8749-46ae57d399b4.png)
I then created a dataframe out of the given listings.csv, cleaned it and assigned it to my chosen variable, then saved it as a cleaned csv file on my computer. All this was contained in the if __name__ = __main__ function.
![image](https://user-images.githubusercontent.com/116648304/231772186-b2986b5c-9524-41e4-9115-37a4211e4093.png)

The final task in this milestone was to create a (features, labels) tuple. I used the Price_Night column in my dataframe as the label, so I assigned this to my Label variable, and the rest of the dataframe excluding the Price_Night column to my features variable. The dataframe aforementioned was a different dataframe consisting of just the numerical data in my listings.csv file.
![image](https://user-images.githubusercontent.com/116648304/231773140-27f87258-89f8-4c0b-bf1f-f915bd03b694.png)

Milestone 4 was my first introduction to models. The models I would be testing will be predicting the price per night of an airbnb holiday let. The first task was the implementation of a basic SGDRregressor model. The code I originally used was mostly deleted but the basis can still be seen. I started off by importing a tuple containing the data frame and the original 'price per night' column. These were then set as my (features, labels). I set the (features, labels) to (x, y) which were then scaled. The next step was to assign a train-test-val split. The training set would be used to set the hyperparameters in the future and train the model. The test set would then be used to test the models capabilities and make sure there isn't any misfitting. After this I predicted the y values using X_train and then found the r2_score and the rmse of the baseline model.
![2023-05-10 (12)](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/5d0e9b79-4983-41b5-bcaa-47d1f836b2f4)

The next task would be to begin tuning the SGDRegressor's hyperparameter. The code in this photo will have been extended for further in the project when I was testing multiple models but the basis is the same. I started by creating a dictionary for the hyperparameters and inside a nested dictionary of all the values I would iterate through. I then created a function which would fit the model, predict the y values and then find the score. At the bottom of the function, multiple variables were introduced for the best model and its metrics\performances. 
![2023-05-10 (8)](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/697b4c17-de7c-492c-a01b-27a8c0f4484e)

The next task would be to run this model and its parameter grid on sklearn's GridSearchCV. This would automatically test all the combinations of parameters and return the models best metrics and parameters. I would use this to compare between my own results the computers. 
![2023-05-10 (9)](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/1f3c8bba-1443-476c-99e0-23e8fcf37313)

The next task after this was to save the model as a joblib file, and its performance metrics and parameters as separate JSON files. This was fairly straightforward and in future I would use this function to create folders named after each model I was testing, all of which contain the thre files menioned beforehand.
![2023-05-10 (10)](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/b94cd2a3-ccb7-4290-b37d-b951a04ab313)

Using this infrastructure I was able to extend on the hyperparameter dictionary and create iterations inside each function to test 3 new models-randomforestregressor, gradientboostingregressor and decisiontreeeregressor. 

The last task was to create a function to decide the best model. This function would load in each model, create predictions and then test its scores. The function would then compare model against model and return the best model and its metrics\parameters.
![2023-05-10 (11)](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/482a0096-18d3-4fa2-90d5-fa1c48664dab)

All of this was enclosed in another function called evaluate_all_models which would then be placed in an if __name__=='__main__' condition. This is so if in future of this project I am required to import this file, the functions won't run multiple times.

The best model was the RandomForestRegressor. I believe this to be accurate in my situation as the SGDRegressor can be difficult to make efficient if the various hyperparameters aren't tuned properly. At the time of doing this section in my project I was inexperienced with tuning so I would like to see if I can improve upon this. The other two models-gradient boosting and decision tree are both prone to overfitting. This was an issue for me with all my models due to my tuning being wide and inefficient but particularly so in these two models. 

One thing I would like to go back and alter would be the hyperparameter dictionary. I believe it is rather remedial at this point and I could drasticlly improve on this. Furhter on into this project I will revisit this when i have more experience and understanding. Although I am tweaking it constantly, the changes are currently small and I believe a complete redo could be effective.

Progressing into milestone 5, the main task was to repeat all of the steps of the regression milestone but with classification models instead. Here the first decision was how to go bout this, create a new file and load it into the original or repeat the code in the original file. However, I completely revamped the original file so all functions would intake the model type (classification or regresssion) and produce an output based off the result. This allowed me to save a lot of time and space copying code just for the same output. 

![1-Imports](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/8c205899-4889-417e-bcfd-893814b65b8e)
These are all the imports used for both the regression and classification models.

![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/96e894b8-e332-47b0-a307-971ffec40541)
The function created in my tabular_data.py file to load the dataset in with 'category' as the label

![2-dataset imports](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/bc7e6e21-c7d9-450b-89c4-3730515a62a3)
This shows the imports of both datasets, one to create a datatset with the 'price_night' as the label and one to include 'price_night in the features and 'category' as the label.

![3- X,y](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/15eb5af6-2ac5-42a4-ac38-d1743a31397f)
This shows creating X, y for each dataset and creating a train-test split twice to create train/test/validation sets. These are then scaled. I used StandardScaler for both datasets as research suggested this was the most commonly used and best for removing outliers that would negatively affect scores

![5- base model and score](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/a670be0c-0794-4e33-b6e6-f9291c9a3030)
This shows the the function I used for creating a baseline model for both the regression model- SGDRegressor and classification model- Logistic regression. The function would take the model type as an argument and output the trained model. The next function would then take the returned model in and create the appropriate scores depending on the model type, on both the train and test sets.

![6- custom tune regression](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/9e571915-ad59-4496-af21-65cb13033d13)
This file shows the updated custom tune function, which now only takes in the original SGDRegressor model and trains it. It is tidied up from the first model in this file as that iterated through all the models, which I believe was not the task. It this returns the best model, its parameters and performance metrics.

![7- gridsearch both models](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/d8618875-be7b-4233-b690-4adc483f52ca)
These two functions are both models. They iterate over the hyperparameter dictionary I created for eeach model type, perform a gridsearch and create a list of the best model, its parameters and a dictionary of its performance metrics. For both instances, the model's performance is decided by the highest validation score. I then save the models in a folder named after the model type, with each model class having its own folder containing separate files for the model (saved as a joblib file), and JSON files for the hyperparameters and performance metrics. The code to save the model is shown in the picture below.
![8-save model](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/ec229402-0c8a-4645-bdc1-5fdeb779b8fd)
![9- find best model](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/55dd0b4a-4dbe-4d7e-b272-56ab00ef1f67)
The photo above is a function that again can take either model type in as an argument and return the best model class across the board. This is also based off the validation set score.
![10-eval models if name=main](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/4d42ae48-50fc-4ab3-8b5a-327717afe70f)
All tuning functions are called in another function called evaluate_all_models before being passed into my if name == main, where the best model function is carried out afterwards.

![4- model dicts](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/8679aa61-0567-4c0a-898e-ba74eaa76c93)
This picture shows the dictionaries used to tune the models. The choice of these hyperparameters were due to effectiveness these had on the model. From research I was led to believe these are the 'key' hyperparameters and testing more would just lead to overfitting/ not having an effect at all really. Testing these hyperparameters, I didn't experience any 'mis-fitting issues on my models and all seemed to return an appropriate score. Had this not been the case, I would have experimented more with hyperparameters such as early stopping and 'pruning' on certain model classes.

![2-class](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/0c296451-0f0a-4778-87f8-64cdec49829c)
This is my first Neural Network class. ALthough basic it demonstrates two hidden layers amongst ReLU activation functions. The widths I used went from 9 features to 27, as this seemed a good staring point being 3x as much. This was then placed down to 18 as a middle point and finally my one output layer. 

![3-train](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/335eb98e-63e5-4897-b0c1-63f56fa375a8)
My main function is the training function. Although this function was primarily used to train the model, as I progressed I called many other functions inside including my save model and scoring of the models. For the hyperparameter testing, we were told to only use 16 parameters. This led me to choose just using learning rate and momentum as my hyperparameters to choose, with 4 values for each key. After training I saved the model in a folder using an isinstance function to decide if it was an nn.Module or not, with the adjooining hyperparameters and metrics in the same file.

This was called in another function that was used to find the best NN model.

![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/c353d224-2b5b-48d2-9f95-8db7bdeff834)
This graph shows all the models being trained in comparison to their loss. It shows a precise similarity between all the models meaning they follow the same trend. This could be due to the limited hyperparameter tuning. In future it would be interesting to see what the comparison is when further tuning is carried out.

![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/8666eb31-adb3-4be5-af3c-ee7a18443ff0)
The loss of the best model.
![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/ef79fb47-ec74-482a-a7a1-66a4bb48821b)
The r2 score of the best model on the training set.
![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/6abff748-021d-47e6-941b-5ad0877db1e5)
The diagram of my networks layers.

![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/f9b3ec2c-e00c-4cad-bdec-159c46e91ca5)
This image shows the introduction of my new dataset used to test my 3 models. As shown, there was a categorical variable now being used which meant introducing OneHotEncoder into my code so the data could be used on the models without outputting null metrics.

![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/7a425af4-1c6c-425b-8f7d-515fa17b0f51)
![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/f230b7ea-101e-488c-889c-f952d5ffadd4)
Example predictions for number of beds in a house, for linear regression style model (top), and classification style model (bottom)
Compared to the prediction output of custom Neural Network (below).
![image](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/5a8284ef-1ca3-4f8a-a7b8-613ed7ab6449)

![ac vs pred lr](https://github.com/OllieMountier/airbnb-property-listing-model/assets/116648304/a2a4ef69-c8ad-4d9b-a3b8-dfb2e2c1de8e)
For the regression models, an easy graph to show the quality of it was the actual vs prediction scatter graph. A good quality graph is supposed in look at a 45 degree up slope. Just looking at it I believe mine is a bit lower but follows that trend quite closely.
