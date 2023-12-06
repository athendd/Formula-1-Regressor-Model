# Formula-1-Regressor-Model

I built a random forest regressor model with the purpose of predicting a Formula 1 team's point total for a single season. I was curious to figure out what features correlate strongly with points in order to figure out a Formula 1 team's season point total.  

I chose to do the Formula 1 for my individual project because I'm an avid Formula 1 fan and my group for the group project chose a Formula 1 dataset as one of our potential three options. 
The Formula 1 World Championship (1950-2023) datset comes from kaggle. The dataset is  made up of fourteen contected datasets. Here is a picture of all 14 datasets:

![Modelanother](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/2aaf7b3a-264f-47f2-bc1a-f0b22ea32d99)

# Selection of Data

I chose to only do five of those fourteen datasets because my computer doesn't have the computational capability to handle the entire dataset. The five datasets I chose were the results, races, constructors, constructors_standings, and constructors results. (picture of the datasets and their respective columns)

Chosen Datasets and their Respective Columns:

results: resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId
races: raceId, year, round, circuitId, name, date, time, url, fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time, quali_date, quali_time, sprint_date, sprint_time
constructors: constructorId, constructorRef, name, nationality, url
constructor_results: constructorResultsId, raceId, constructorId, points, status
constructor_standings: constructorStandingsId, raceId, constructorId, points, position, positionText, wins

I merged these datasets using pd.merge because these datasets were connected by key columns. I also needed to merge these datasets in order to have a singular dataset to train the model on. Here are pictures of the code used to merge the datasets and the head of the merged dataset:

Picture of Code to Merge the Datasets:
![Model4](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/0a503a29-207e-4af7-b179-9ae15b9232ac)

Picture of the Head of the Merged Dataset:
![Model5](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/c7bed1de-3a36-4710-a580-a18c84ebca62)



The dataset is prone to outliers. For example, in the year 1988 Mclaren won the constructor's championship with a total 199 points which was 134 more points than Ferrari who finished in second that season. I chose not to get rid of outliers since they are true values. If I got rid of the outliers then the dataset would become inacurate due to the loss of true values. 

I did make a histogram of the target variable points to see if it was skewed which ended up being the case as the target variable was right skewed. I need to get rid of the skewness in the target variable because I want the model to learn relationships and aptterns accross the entire range of values so it will be more robust and less sensitive to variations in the inputted data. I  chose to perform a log transformation on the target variable in order to give it a more even distribution. 

I chose not to scale or normalize the distribution of values for the predictor variables because I'm using a random forest regressor model which inherently robust to the scale of the features. Random forests models are a hierarchy of decision trees whose decision making algorithm isn't affected by the scale of the features. This is due to the fact the the decision making algorithm focuses on the order of values and not their scale. 

Picture of Histogram for Predictor Variable:


I used df.isnull().sum().sum() to get the total number of null values in the dataset which ended up being 0. Since there were no null values in the dataset there is no need for me to drop or deal with null values. 

I did have to convert all my categorical data columns to numerical ones because a regessor model requires that all of its features be numerical data. I accomplished this through the use of one hot encoding with pd.get_dummies because it converts each category in the column into a dummy variable. 
![model2](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/c3cde784-69ec-4d85-bd46-300519306d95)

# Methods

Libraries Used: numpy, matplotlib, scikit learn, and pandas

I chose to use a regressor random forest model because I needed to use a regressor model in order to predict the points for a team and I chose random forest because of its ability to avoid overfitting the data. I also chose the random forest regressor become it runs effecentially on large datasets such as Formula 1 World Championship(1950-2023). I used the random forest regressor model's feature_importances_ to get the importance of each feature with respect to the model. I did this to not only figure out which features best contributed to the dataset but also how well they contributed to the dataset. I then used this information to get rid of features that had a very weak correlation because they provided little to no useful data to the model. 

Picture of Code to get Importance of each Feature:
![Model6](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/76211d94-c85a-4c15-9f69-9b394c60e10f)

The model had an initially high mean squared error of 77 so I also built a scatterplot of the model's predicted vs actual values to see what was causing my it. After looking at this model I noticed that the model had trouble predicting outliers because they are so rare but I also noticed that there was a vast difference in the accumalation of points overtime. Upon further research, I discovered that this was due to the fact that Formula 1 has changed its points system over time. From the years 1962 to 1990 the point system was 9, 6, 4, 3, 2, 1 points but in the year 1991 to 2002 that point system was changed to 10, 8, 6, 5, 4, 3, 2, 1 points and was then changed again in the year 2003 to 25, 18, 15, 12, 10, 8, 6, 4, 2, 1. 

Picture of Scatterplot for Model:
![model1](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/455f2a1e-4b01-4532-807d-4bff515b36c3)


The reason for my model's high mean sqaured error was because of the fact that I was training the model on multiple different point systems which made it diffcult for the model to accuratly predict the points for a constructor in a season because it didn't have a singular system to go off of. I decided to fix this error by only using data between the years 1961 and 1990 since the point system remained the same during that time period which gives the model a singular point system to use. I aslo chose that time period since there was very little divitation between the values of points in the points system which makes it easier for the model to predict values due to the lakc of variation in points. I found that I can predict a constructor's points in a season based off my chosen features because the model only had a mean squared error of 0.003. I used a scatter plot to compare the model's predicted values to its actual values. 

Picture of Updated Scatterplot for Model:
![model3](https://github.com/athendd/Formula-1-Regressor-Model/assets/141829395/6b213098-aacb-4a9d-960b-c55311618fb8)

# Results

The low mean sqaured error of the model shows that its features are effective predictors of the points a Formula 1 team will score in a given season. This suggests that the chosen features such as grid, laps, constructor position, etc have a signifigant impact on a Formula 1 team's performance. 

# Discussion

Formula 1 team, analysts, and investors can use this insights to make informed decisions about strategies, resurce allocation, and team management to improve their odds of scoring more points. The model's results are similar to what other researchers have found like the fact that better starting grid position correlate with more points accumalted. This model can be used as a base model for future research as more features can be added on such as pit stop time, qualifying time, etc in order to find other features that have a significant impact on a Formula 1 team's season point total. There can also be external factors such as advancements in technology that could impact a Formula 1 team's season point total that could be incorporated into future models. The model could also be improved to incorporate data from other time periods or base the model on more current time periods, like building a model for the modern Formula 1 points system so it can predict future results. 

