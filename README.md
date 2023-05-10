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

All of this was enclosed in another function called evaluate_all_models which would then be placed in an if __name=='__main' condition. This is so if in future of this project I am required to import this file, the functions won't run multiple times.

The best model was the RandomForestRegressor. I believe this to be accurate in my situation as the SGDRegressor can be difficult to make efficient if the various hyperparameters aren't tuned properly. At the time of doing this section in my project I was inexperienced with tuning so I would like to see if I can improve upon this. The other two models-gradient boosting and decision tree are both prone to overfitting. This was an issue for me with all my models due to my tuning being wide and inefficient but particularly so in these two models. 

One thing I would like to go back and alter would be the hyperparameter dictionary. I believe it is rather remedial at this point and I could drasticlly improve on this. Furhter on into this project I will revisit this when i have more experience and understanding. Although I am tweaking it constantly, the changes are currently small and I believe a complete redo could be effective.

