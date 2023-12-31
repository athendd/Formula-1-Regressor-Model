# Formula-1-Regressor-Model

The objective of the project was to build a random forest regressor model that could accurately predict a given Formula 1 team's point total for a single season. I was curious to figure out what features correlate strongly with points.  

I chose to do the Formula 1 for my individual project because I'm an avid Formula 1 fan. I had become interested in the idea of building a machine learning model to predict Formula1 results after my gorup for the group project chose a Formula 1 dataset as one of the potential three options for the project. 

# Selection of Data

I used the Formula 1 World Championship (1950-2023) datset from kaggle. The dataset is made up of 14 contected datasets. 

**The 14 Datasets:**

![there](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/31a549be-a78a-4035-b5ca-af4a1f36be9a)


I chose to only use 5 of the 14 datasets because my computer lacks the computational abilities to handle the entire dataset. The 5 datasets I chose were the results, races, constructors, constructors_standings, and constructors results. I chose these datasets because I beleived the dataset's feature could have a strong correlation with points or the datasets contained a lot of features that could privde insightful data and be connected to one of the other datasets. 

**Chosen Datasets And Their Respective Columns:**

    results: resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId
    races: raceId, year, round, circuitId, name, date, time, url, fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time, quali_date, quali_time, sprint_date, sprint_time
    constructors: constructorId, constructorRef, name, nationality, url
    constructor_results: constructorResultsId, raceId, constructorId, points, status
    constructor_standings: constructorStandingsId, raceId, constructorId, points, position, positionText, wins

I merged these datasets using pd.merge because many of these datasets contained the same columns. I also needed to merge these datasets in order to have a singular dataset to train the model on. I then checked the head of the merged dataset using df.head() to ensure the merging had been performed properly. I then looked at the descriptive statistics for each column using df.describe() to get an understanding of the characteristics of each column and to learn which columns were numerical and which ones weren't.

**Code To Merge The Datasets:**

![Model4](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/0a503a29-207e-4af7-b179-9ae15b9232ac)

**Head Of The Merged Dataset:**

![Model5](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/c7bed1de-3a36-4710-a580-a18c84ebca62)

**Descriptive Statistics For Each Columns In The Dataset:**

![model13](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/10b8886d-65cc-432c-94c1-2402db651a3a)

The dataset does contain outliers. For example, in the year 1988 Mclaren won the constructor's championship with a total 199 points which was 134 more points than Ferrari who finished in second that season. I chose not to get rid of outliers since they represent true values. If I got rid of the outliers then the dataset would become inacurate due to the loss of these true values. 

A histogram of the target variable (points) revealed that its distributions of values was right skewed. I need to get rid of this skewness because I want the model to learn relationships and patterns accross the entire range of values so it will be more robust. I chose to perform a log transformation on the target variable in order to give it a more even distribution of values. 

**Histogram For Target Variable Before Log Transformation:**

![model9](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/35024b9d-5fdd-4281-a7ed-cbd6e8c96a6d)

**PHistogram for Predictor Variable After Log Transformation:**

![model8](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/05f7ba2e-8003-4636-bc51-1402d3b0f716)

I chose not to scale the predictor variable or normalize the distribution of values for the predictor variables because a random forest regressor model is inherently robust to the scale of the features. The decision trees that make up a random forest regressor model have a decision making algorithm that isn't affected by the scale of the features because it focuses on the ordr of values.

I used df.isnull().sum().sum() to get the total number of null values in the dataset which ended up being 0. Since there were no null values in the dataset there is no need for me to drop or deal with null values. 

I did have to convert all my categorical data columns to numerical ones because a regessor model requires that all of its features be numerical data. I accomplished this through the use of one hot encoding with pd.get_dummies because it converts each category in the column into a dummy variable. 

**Dataset's Columns After One Hot Encoding:**

![Model_Columns](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/4796f141-ce4d-43be-8a59-c61343492267)

# Methods

**Libraries Used:** 

    numpy: peform log transformation on distribution of target variable and get original value scale for target variable on scatterplot
    pandas: to obtain the datasets, merge the datasets, modify the datasets, and turn the datasets into a dataframe
    matplotlib: for data visualizations and graphs
    scikit learn: to create and build the random forest regressor machine learning model 

I chose to use a regressor random forest model because I needed to use a regressor model in order to predict the points for a team and I chose random forest because of its ability to avoid overfitting the data. I also chose the random forest regressor become it runs effecentially on large datasets such as Formula 1 World Championship(1950-2023). 

**Picture Of Code To Build The Random Forest Regressor Model And Fit It To The Dataset:**

![Model7](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/ded2f822-144a-4559-973a-0bfc30ff1ec1)

I used the random forest regressor model's feature_importances_ to get the importance of each feature with respect to the model. I did this to not only figure out which features best contributed to the dataset but also how well they contributed to the dataset. I then used this information to get rid of features that had a very weak correlation because they provided little to no useful data to the model. 

**Picture Of Code To Get Importance Of Each Feature:**

![Model6](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/76211d94-c85a-4c15-9f69-9b394c60e10f)

The model had an initially high mean squared error of 77 so I also built a scatterplot of the model's predicted vs actual values to see what was causing my it. After looking at this model I noticed that the model had trouble predicting outliers because they are so rare but I also noticed that there was a vast difference in the accumalation of points overtime. Upon further research, I discovered that this was due to the fact that Formula 1 has changed its points system over time. From the years 1961 to 1990 the point system was 9, 6, 4, 3, 2, 1 points but in the year 1991 to 2002 that point system was changed to 10, 8, 6, 5, 4, 3, 2, 1 points and was then changed again in the year 2003 to 25, 18, 15, 12, 10, 8, 6, 4, 2, 1. 

**Picture of Scatterplot For Model:**

![model1](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/455f2a1e-4b01-4532-807d-4bff515b36c3)


The reason for my model's high mean sqaured error was because of the fact that I was training the model on multiple different point systems which made it diffcult for the model to accuratly predict the points for a constructor in a season because it didn't have a singular system to go off of. I decided to fix this error by only using data between the years 1961 and 1990 since the point system remained the same during that time period which gives the model a singular point system to use. I aslo chose that time period since there was very little divitation between the values of points in the points system which makes it easier for the model to predict values due to the lack of variation of point's values. I found that I can predict a constructor's points in a season based off my chosen features because the model only had a mean squared error of 0.0387. I used a scatter plot to compare the model's predicted values to its actual values. 

**Picture Of Updated Scatterplot For Model:**

![model3](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/6b213098-aacb-4a9d-960b-c55311618fb8)

I realized that in order to maximize the efficiency of my model, I needed to use best the best possible hyperparameters, which are parameters whose values control the learning process of the model, for my random forest regressor model. I decided to use the hyprparameter tuning method known as grid search CV. I needed to  define lists for all potential values of each hyperparameters then import those lists into a grid search CV object along with the random forest regressor model. The grid search CV would find the best hyperparameters for the models by testing all possibe combinations of each value from the hypterparameters values' lists. The mean squared error of the model was decreased to 0.0379. 

**Picture Of Code To Use GridSearchCV On Model:**

![model10](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/928c0da5-d752-4377-a412-d9ca0bb4e46f)

# Results

The low mean sqaured error of the model shows that its features are effective predictors of the points a Formula 1 team will score in a given season. This suggests that the chosen features such as grid, laps, constructor position, etc have a signifigant impact on a Formula 1 team's performance. 

# Discussion

Formula 1 team, analysts, and investors can use this insights to make informed decisions about strategies, resurce allocation, and team management to improve their odds of scoring more points. The model's results are similar to what other researchers have found like the fact that better starting grid position correlate with more points accumalted. This model can be used as a base model for future research as more features can be added on such as pit stop time, qualifying time, etc in order to find other features that have a significant impact on a Formula 1 team's season point total. There can also be external factors such as advancements in technology that could impact a Formula 1 team's season point total that could be incorporated into future models. The model could also be improved to incorporate data from other time periods or base the model on more current time periods, like building a model for the modern Formula 1 points system so it can predict future results. 

# Coding

**Linked To Main ipynb File:** content/codes/Final_Project_Model_GridSearchCV.ipynb
