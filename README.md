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
