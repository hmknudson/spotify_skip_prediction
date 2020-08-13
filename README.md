# 1. Introduction

The dataset in this project is one of Spotify’s publicly-released skip-prediction datasets. It was originally released for Spotify’s 2018 skip prediction challenge (https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge/dataset_files). The object of the original challenge (which is now over) was to use a track’s features (such as danceability rating, track length, etc.) to predict whether users would skip that specific track. Therefore, Spotify released two separate datasets - one containing only track features and one containing only user behavior features.

This particular project makes use of the user behavior dataset instead of the track features dataset to attempt to predict track skips. Since many competitors have long since submitted their solutions to the original challenge, I wanted to attempt something different that would address the same issue from a new angle. Therefore, this project posed the challenge of determining whether user-related features alone would be enough to predict track skips. Additionally, because the dataset I used was quite large - approximately 2.9 million observations - the project presented a great opportunity to work with Dask to deal with big data in an efficient manner.

The skip-prediction tool that resulted from this project ultimately would help Spotify determine whether users are likely to skip tracks based solely on their behaviors, without overcomplicating the model by utilizing the separately collected track features. It also would allow Spotify to pinpoint user behaviors that are more associated with skipping tracks with a high degree of specificity. Thus, this solution is a real-world, workable solution that contributes to Spotify’s understanding of what makes users choose to skip a song, which may lead to better song recommendations.

# 2. Data

## About the Data

The dataset comes from AICrowd.com, which is where Spotify held their skip prediction challenge, and the only place they released the dataset. Due to the subprime computing power available, I downloaded one segment of the 130 million-observation dataset and used this segment as a sample, which proved adequate, as the sample alone contains almost 3 million observations. The dataset was read in with Dask, and all 21 columns that describe user behavior were present in the sample dataset. All columns contained categorical data.

## Cleaning the Data

Spotify included no null values in the dataset, and all variables were categorical, so outliers were not an issue. 

All objects were coded into numeric values, and month was changed from a regular object to a DateTime object, and month and year were also separated out into their own variables, since the entire date was not likely to go into the models. Some variables were originally in what I would consider backwards order (where 0 means ‘yes’ and 1 means ‘no’), so I reversed that order when they were re-coded, so later interpretation would be simpler.

Finally, in order to limit the scope of the project and make sure consistency, only listening sessions that contained a total of 20 tracks were included in the final dataset, which still had approximately 1.8 million observations. Then, it was decided that the project would examine the user behaviors for just the first, tenth, and twentieth tracks, so these were each put into their own datasets so they could be analyzed separately. This configuration tells us how Spotify users behave differently during the first track in their listening session, vs. their middle track, vs. their last track. This created three different datasets, each with a length of 89,672 and 23 columns (due to adding the extra date columns).

# 3. Methods

The following is a breakdown of the methods used in this project.

## Descriptive Statistics & Unsupervised Learning

Various descriptive statistics were examined to get a better sense of exactly what’s going on in each dataset. 

Unsupervised learning - specifically, clustering - was utilized as a part of the descriptive statistics. Understanding how the data was clustered by HDBSCAN allowed us to learn a great deal about Spotify users and what makes them different from each other. Because the datasets were separated by track number (1, 10, and 20), I was able to get a more granular view of how user behavior was segmented within each specific track.

Before clustering, Spearman correlations were run between all the features to determine where multicollinearity could be a problem. In addition to checking the correlation matrix, a Variance Inflation Factor (VIF) was run to determine the magnitude of the multicollinearity for each variable. Thus, the combination of the correlation matrix and the VIF informed the removal of variables that were multicollinear. A total of 10 features were removed during this part of feature selection, which left us with 13 features. 

Then, a few features that had more granularity than necessary were re-coded into simpler categorical buckets, with the goal of retaining as much information as was necessary, but also forgoing the temptation of excessive granularity and overcomplication. For example, ‘hour of day,’ which contained all 24 hours as separate values, was re-coded into ‘early morning,’ ‘morning,’ ‘afternoon,’ and ‘night.’ At this point, one-hot encoding was performed on all of the remaining features, which expanded the number of features into 42.

HDBSCAN was then used as the clustering algorithm, and after optimizing it for some time, I was able to produce a 12-cluster solution for all 3 tracks. A statistical logistic regression was run for each of the track’s clusters, and from this, coefficients and odds ratios for each cluster were obtained. This information tell us what each cluster is primarily about, and each cluster was also plotted alongside against each feature in the datasets, to gain more knowledge about how Spotify users could be segmented.

## Supervised Learning

Fortunately, feature selection was already done earlier, prior to clustering.

### Model Selection

Based on the specifications of the three datasets (i.e., not normally distributed, all categorical, require a classification task, and are a modest size), three different types of models were chosen - a Logistic Regression, a Random Forest, and XGBoost. 

### Model Optimization

Parfit was used to thoroughly hyper-parameterize each chosen model in a step-by-step manner. One hyperparameter was run through Parfit, with an empty model object. Then, when the best value for that particular hyperparameter was found, it was applied to the model. Then, a second Parfit was run to find the next hyper-parameter’s optimal value, given the current model with the hyperparameter that was already selected, etc., etc.  

A train/validation/test set split was used, rather than just train/test or cross-validation, because I wanted to approximate a real-world situation where you would want to leave out test data until the very end, because you’d need to test the model with unseen data. In cross-validation, all the data has been seen by the model, which doesn’t realistically tell us what the model is capable of when incoming data is unknown. Thus, I saved the test until the end and only retrieved test set scores when I had chosen the best model based on training and validation set scores.

### Model Evaluation

Once all 3 models were optimized for all 3 tracks (a total of 9 rounds of optimization/hyper-parameterization), it was time to evaluate the models and choose the best-performing one. 

The Logistic Regression was evaluated first in terms of its log loss score and area under the ROC curve (AUC score), while F1 score and accuracy score were secondary & tertiary metrics. Both the Random Forest and XGBoost were evaluated first in terms of their AUC scores and F1 scores, with accuracy score as a secondary metric. After checking model performance across all three datasets, XGBoost was chosen as the best model.

### Model Interpretation

Once XGBoost was chosen, the feature importance for each feature in the model was calculated and plotted, for all three datasets’ models. Then, a statistical logistic regression was run (again, for each dataset) with only those features that were determined to be most important to the model. The statistical logistic regression showed which features were statistically significant to the model and which were not, which in turn allowed me to remove those that were not and see whether that made the model better as a whole (utilizing such metrics as AIC score, BIC score, Log-Likelihood, etc.). For models where these scores improved upon removing certain features, those features were kept out. Then, each of the three XGBoost models was run again, this time with just the most important features, ultimately leaving us with more stable models. Finally, odds ratios or each of the three tracks’ features were calculated, allowing us to interpret what the models mean in more of a business case setting.

# 4. Conclusion

## Recapping the Project

In sum, 3 different supervised learning algorithms were used to try to build the best possible models for the Spotify user behavior dataset, looking specifically at users’ first tracks, their tenth tracks, and their last tracks. Prior to running these algorithms, clustering with HDBSCAN was also done, which led to a greater understanding of the datasets and also later proved to be a highly important feature in the supervised learning models. The model that consistently scored the best across model-based and performance-based metrics was XGBoost. After choosing this algorithm, feature importance analyses were run, and statistical logistic regressions were run with only the most important features to try to arrive at more stable models. Finally, odds ratios were calculated for each of the important features, to explain what each feature means in terms of track-skipping behavior and how this might affect Spotify overall.

## Limitations

Were I to do this again, there are a few things I might change. First, if I had the luxury of more time, it would have been nice to attempt to optimize the track 1 models a bit more. While it makes sense that the first track of a user’s listening session would be the most difficult to predict, it may have been possible to increase the models’ performance a bit more. Additionally, simply having chosen a dataset with only categorical data imposed that limitation on the models, which often perform better with more continuous data. However, this was mitigated as much as possible by ensuring all variables were numeric, using one-hot encoding, and working to not overly condense the information within the variables. Finally, the relatively low amount of computing power available to me was also a limitation. Using a subsample of the 130 million observation dataset, using Dask at the outset of the project, and creating an analysis plan that allowed me to split the data up into three datasets vastly mitigated this issue. The only moment when processing took quite a while was during the clustering portion of the project. In an ideal world, it would have been fascinating to see how the entire dataset would function in the models, but the amount of data that was used was at least satisfactory.

## Conclusions & Implications

Ultimately, after obtaining the feature importance and odds ratios for the XGBoost models, a few conclusions about Spotify users’ skipping behavior can be made. First, whether users paused before playing their current track was one of the most important determinants of whether they would skip that track. Those who did pause first were 12 times more likely to skip. Second, Spotify users who were listening to either a personal playlist or to Spotify’s top charts feature were much more likely to skip their current track than those who were not. Finally, the majority of Spotify users did skip their current track, regardless of whether it was their first, tenth, or twentieth. This implies that most people who listen to Spotify don’t listen to every song that gets played; rather, most of these people seem to skip around quite a bit. More research would be needed to pinpoint exactly what motivates this skipping, but at a higher level, it seems a user’s level of engagement with Spotify is not the major determining factor in whether or not they’ll skip their current track. Rather, skipping is more probably down to the user’s context at the time - i.e, the time of day and the type of playlist or feature (like Spotify radio) they’re listening to, etc.
 
## Future Research Possibilities

Now that this project on Spotify user behavior and skipping has been completed, I've thought of a few different projects that might be worthwhile to do in the future, to further investigate Spotify users' skipping behavior. First, it could be interesting to compare this project to a project that does not use one-hot encoding, but rather just the variables in their numeric form. Having both of these frameworks would hopefully give us not only the current level of granularity that we've achieved with one-hot encoding, but also a higher level overview of how each feature as a whole contributes to predictive models on track skipping. Additionally, it would be fascinating to look into how specific track features (e.g., danceability, length, acousticness, etc.) affect skipping, instead of user behavior. Of course, the Spotify skip prediction challenge was largely about this topic, so Spotify may have enough information on the track features as it stands now.
