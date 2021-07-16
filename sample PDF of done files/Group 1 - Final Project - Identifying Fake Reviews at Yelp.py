# Databricks notebook source
# MAGIC  %md
# MAGIC  # Group 1 Project - Identifying Fake Yelp Restaurant Reviews

# COMMAND ----------

# MAGIC %md
# MAGIC Business Analytics Capstone (BA 5651) - Spring 2021

# COMMAND ----------

# MAGIC %md
# MAGIC Group Members:- Neil D Souza, Ajin Thomas, Debanjana Dey, Yash Gaur

# COMMAND ----------

# MAGIC %md
# MAGIC #Business Problem

# COMMAND ----------

# MAGIC %md
# MAGIC The success or failure of a business can highly depend on its online reviews. From choosing the best hotels in town to finding the best restaurants, most people these days tend to rely on online reviews. Reviews are quickly replacing friend recommendations to become the main way we vet businesses against one another. We tend to rely on online reviews so much that a single bad review can completely ruin the online reputation of a company.
# MAGIC Yelp is one of the largest online reviews’ platform for restaurants and other businesses, where customer reviews are utilized to make the necessary changes to their products and services. Our project is based on yelp restaurant reviews. The business problem is that a lot of fake restaurant reviews exist, and these reviews affect the business’s overall ratings, and diminishes their ability to achieve the goals these restaurants have in mind, as customer perceptions of their products and services diminish. Without being able to identify which of its restaurant reviews are fake or suspicious, Yelp will not be able to make the necessary changes to maximize platform brand loyalty, satisfaction, and profits. So, It is crucial for Yelp to be able to identify such fake or suspicious reviews immediately, and take corrective action, in order for it as a platform to stay relevant. 
# MAGIC 
# MAGIC This Business Problem is a very relevant problem, and is not just one that Yelp is facing, but is also common to almost all other online review platforms. Our end goal is to create models that can not only help address this issue at Yelp, but that can be utilized across other similar platforms, and companies. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Overview of the Analysis Conducted

# COMMAND ----------

# MAGIC %md
# MAGIC The following are some of the main technical aspects of our project.
# MAGIC 
# MAGIC •	Initially, we start with performing an EDA on the data, to identify some key characteristics of the data for further analysis.
# MAGIC 
# MAGIC •	Through Feature Engineering we added new/additional features to our dataset that enabled us to develop a more accurate and robust analytical model to differentiate Fake and True reviews.
# MAGIC 
# MAGIC •	TFIDF. TF-IDF stands for Term Frequency — Inverse Document Frequency and is a statistic that aims to better define how important a word is. We analyzed the customer reviews to find whether they are positive, negative, or neutral for the respective Yelp restaurants as per the dataset. 
# MAGIC 
# MAGIC 
# MAGIC •	Dimensional Reduction. We made a correlation matrix and removed variables that were highly correlated. 
# MAGIC 
# MAGIC •	Model Building. This was the final step in our analysis
# MAGIC 
# MAGIC •   Adjusting for Imbalanced Data - Modified Algorithm Approach
# MAGIC 
# MAGIC •   Model Selection

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# As mentioned above, EDA is the first step of our analysis. Here, we are importing all the necessary libraries and packages needed for our analysis:-

import os
import numpy as np 
import pandas as pd

# COMMAND ----------

# Creating a Spark DataFrame:-

spark_df = spark.read.format("csv").load("dbfs:/FileStore/shared_uploads/tul56215@temple.edu/Restaurant_data.csv")

# COMMAND ----------

# Converting the Spark DataFrame to a Pandas DataFrame:-

pandas_df = spark_df.toPandas()

# COMMAND ----------

# Screenshot of the first few rows of the DataFrame:-

pandas_df.head()

# COMMAND ----------

# Screenshot of the last few rows of the DataFrame:-

pandas_df.tail()

# COMMAND ----------

# Dropping the index column:-

df = pandas_df.rename(columns=pandas_df.iloc[0]).drop(pandas_df.index[0])
print(df)

# COMMAND ----------

# Importing all the necessary libraries and packages:-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# A look at the Feature Data Types:-

print(df.dtypes)

# COMMAND ----------

# Converting the data types to the appropriate form:-

df["reviewCount"] = pd.to_numeric(df["reviewCount"])
df["rating"] = pd.to_numeric(df["rating"])
df["restaurantRating"] = pd.to_numeric(df["restaurantRating"])
df["reviewUsefulCount"] = pd.to_numeric(df["reviewUsefulCount"])
df["friendCount"] = pd.to_numeric(df["friendCount"])
df["firstCount"] = pd.to_numeric(df["firstCount"])
df["usefulCount"] = pd.to_numeric(df["usefulCount"])
df["coolCount"] = pd.to_numeric(df["coolCount"])
df["funnyCount"] = pd.to_numeric(df["funnyCount"])
df["complimentCount"] = pd.to_numeric(df["complimentCount"])
df["tipCount"] = pd.to_numeric(df["tipCount"])
df["fanCount"] = pd.to_numeric(df["fanCount"])
df["mnr"] = pd.to_numeric(df["mnr"])
df["rl"] = pd.to_numeric(df["rl"])
df["rd"] = pd.to_numeric(df["rd"])
df["Maximum Content Similarity"] = pd.to_numeric(df["Maximum Content Similarity"])
df['date'] = pd.to_datetime(df['date'])
df['yelpJoinDate'] = pd.to_datetime(df['yelpJoinDate'])

# COMMAND ----------

# Checking the revised Data Types after conversion:-

print(df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC #Data Visualisation 1

# COMMAND ----------

# Importing all the necessary libraries and packages:-

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 5)
import seaborn as sns

# COMMAND ----------

# A look at the count/distribution of not-fake (0) and  fake (1) reviews in the dataset:-

df["flagged"].value_counts()

# COMMAND ----------

# A visual look at the count/distribution of not-fake (0) and  fake (1) reviews in the dataset:-

df["flagged"].value_counts().plot(kind='bar', color = 'green')
plt.title("Count by Fake vs Not-Fake Reviews")
plt.xlabel("is_fake")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution of review "ratings" in the dataset:-

df["rating"].value_counts()

# COMMAND ----------

# A visual look at the count/distribution of review "ratings" in the dataset:-

df["rating"].value_counts().plot(kind='bar', color = 'blue')
plt.title("Count by Review Rating")
plt.xlabel("rating")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution of the top 10 "restaurantID's" in the dataset:-

df["restaurantID"].value_counts()[:10]

# COMMAND ----------

# A visual look at the count/distribution of the top 10 "restaurantID's" in the dataset:-

df["restaurantID"].value_counts()[:10].plot(kind='bar', color = 'red')
plt.title("Count by Restaurant ID - Top 10")
plt.xlabel("restaurantID")
plt.ylabel("count")

# COMMAND ----------

# MAGIC  %md
# MAGIC  # Feature Engineering                                                                                      

# COMMAND ----------

# MAGIC %md
# MAGIC The following are the features that we incorporated in our dataset: 
# MAGIC 
# MAGIC •	Length of review in Characters:  The idea is that Shorter reviews have a higher probability to be fake due to the reviewer being unable to mention details about their experience.
# MAGIC 
# MAGIC •	Length of review in Words:  The idea is that Shorter reviews have a higher probability to be fake due to the reviewer being unable to mention details about their experience. Utilizing this 
# MAGIC     information helped us achieve a model that is more effective in rooting out fake reviews. This feature is similar to length of reviews in Characters.
# MAGIC 
# MAGIC •	Date (Join Date – Review Date): In the dataset we have two columns that show ‘Join date’ and ‘Review Date’. We used this to determine the duration between a user’s account creation date and the 
# MAGIC     date the review was posted. A new account posting highly negative or positive reviews shortly after creating an account is a possible red flag and identifying these reviews is vital.
# MAGIC 
# MAGIC •	One hot encoding for Gender: Feature engineering for a gender variable using one hot Encoding to create dummy variables. 
# MAGIC 
# MAGIC •	Positive and Negative words in a Review:  Too many positive or negative words could suggest a possible fake review which is why we felt this was an important feature to include.
# MAGIC 
# MAGIC •	Average word length of each Review: Similar to length of words, a review with a shorter average word length is more likely to be fake which is why we also took this into consideration.

# COMMAND ----------

# MAGIC %md
# MAGIC 1) Number of Characters (Without Spaces) in the Review Content

# COMMAND ----------

# Removing all the spaces in the "Review Content":-

NoSpacesinContent = df['reviewContent'].str.replace(" ", "")

# COMMAND ----------

# A quick look at the review content without the spaces:-

NoSpacesinContent

# COMMAND ----------

# Calculating and Adding "ReviewLengthCharacters" as a feature/column in the DataFrame:-

df['ReviewLengthCharacters'] = NoSpacesinContent.str.len()

# COMMAND ----------

# Snapshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# Renaming the "mnr", "rl" and "rd" Columns:-

df.rename(columns = {'mnr' : 'MaximumNumberOfReviews-Day', 'rl' : 'ReviewNumberofWords', 'rd' : 'ReviewDeviation'}, inplace = True)

# COMMAND ----------

# Snapshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 2) Number of Days between the join date and the review post date 

# COMMAND ----------

# Calculating and Adding "difference_in_datetime" as a feature/column in the DataFrame:-

df['difference_in_datetime'] = abs(df['date'] - df['yelpJoinDate'])
print(df['difference_in_datetime'])

# COMMAND ----------

# Snapshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# A look at the columns in the DataFrame:-

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC 3) Number of Gender Features that can be taken into account  

# COMMAND ----------

# Imputing gender feature:-

df["gender"] = ""
print(df['gender'])

# COMMAND ----------

# Imputing male gender values:-

df['gender'] = df['gender'].replace(df['gender'], 'Male')
print(df)

# COMMAND ----------

# Screenshot of the top few rows of the DataFrame:-
  
df.head()

# COMMAND ----------

# Introducing female gender into into df:-

df.loc[1:15000,'gender'] = 'female'

df = df.sample(frac = 1) 
df

# COMMAND ----------

# Perform one-hot encoding:-

df = pd.get_dummies(df, columns=['gender'])
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 4) Positive and Negative words for each review

# COMMAND ----------

# Binary variable for positivity and negativity classifcation:- 

df = df[df['rating']!= 3]
df['Positively_Rated'] = np.where(df['rating']>3, 1, 0)
df.head(10)

# COMMAND ----------

# To find positive and negative words for each review:-

pd.crosstab(index = df['Positively_Rated'], columns="Total count")

# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #Data Visualisation 2

# COMMAND ----------

# A look at the count/distribution of "Maximum Number of Reviews in a Day" in the dataset:-

df["MaximumNumberOfReviews-Day"].value_counts()

# COMMAND ----------

# A visual look at the count/distribution of "Maximum Number of Reviews in a Day" in the dataset:-

df["MaximumNumberOfReviews-Day"].value_counts().plot(kind='bar', color = 'orange')
plt.title("Maximum Number of Reviews in a Day")
plt.xlabel("MaximumNumberOfReviews-Day")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution of "Number of Words in A Review" in the dataset:-

df["ReviewNumberofWords"].value_counts()[:10]

# COMMAND ----------

# A visual look at the count/distribution of "Number of Words in A Review" in the dataset:-

df["ReviewNumberofWords"].value_counts()[:10].plot(kind='bar', color = 'purple')
plt.title("Number of Words in A Review")
plt.xlabel("ReviewNumberofWords")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution of "Number of Characters in A Review" in the dataset:-

df["ReviewLengthCharacters"].value_counts()[:10]

# COMMAND ----------

# A visual look at the count/distribution of "Number of Characters in A Review" in the dataset:-

df["ReviewLengthCharacters"].value_counts()[:10].plot(kind='bar', color = 'turquoise')
plt.title("Number of Characters in A Review")
plt.xlabel("ReviewLengthCharacters")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution of "Top 10 Number of days between Joining Date and Posting Date" in the dataset:-

df["difference_in_datetime"].value_counts()[:10]

# COMMAND ----------

# A visual look at the count/distribution of "Top 10 Number of days between Joining Date and Posting Date" in the dataset:-

df["difference_in_datetime"].value_counts()[:10].plot(kind='bar', color = 'grey')
plt.title("Top 10 Number of days between Joining Date and Posting Date")
plt.xlabel("difference_in_datetime")
plt.ylabel("count")

# COMMAND ----------

#Looking at the crosstable for "difference_in_datetime":-

difference_in_datetime_counts = pd.crosstab([df.difference_in_datetime], df.flagged.astype(bool))
difference_in_datetime_counts

# COMMAND ----------

# A look at the count/distribution by Gender:-

df["gender_Male"].value_counts()

# COMMAND ----------

# A visual look at the count/distribution by Gender:-

df["gender_Male"].value_counts().plot(kind='bar', color = 'black')
plt.title("Breakdown by Gender")
plt.xlabel("gender_Male")
plt.ylabel("count")

# COMMAND ----------

# A look at the count/distribution by Positively or Negatively Rated:-

df["Positively_Rated"].value_counts()

# COMMAND ----------

# A visual look at the count/distribution by Positively or Negatively Rated:-

df["Positively_Rated"].value_counts().plot(kind='bar', color = 'orange')
plt.title("Breakdown by Positively or Negatively Rated")
plt.xlabel("Positively_Rated")
plt.ylabel("count")

# COMMAND ----------

# MAGIC %md
# MAGIC # Tokenization

# COMMAND ----------

# Tokenization:-

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import re




from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer

df['reviewContent']=df['reviewContent'].apply(str) 

df.head()



df['tokenized_reviewContent'] = df['reviewContent'].apply(word_tokenize)


# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# Bag of words:-

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = 'word')
df_bow = bow.fit_transform(df['reviewContent'])
df_bow

# COMMAND ----------

# Importing the necessary libraries:-

import os
import numpy as np 
import pandas as pd

# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Additional Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 5) Average Word Length of Each Review

# COMMAND ----------

# Calculating and Adding "Average Word Length" as a feature/column in the DataFrame:-

for index, row in df.iterrows():
  if row['reviewContent']:
    words = row['reviewContent'].split()
    df.loc[index, 'avg_word_len'] = sum(len(element) for element in words) / len(words)
  else:
    df.loc[index, 'avg_word_len'] = 0

# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

##Preprocessing reviewscontent column before TF-IDF:-

features = df.iloc[:, 6].values
labels = df.iloc[:, 7].values

# COMMAND ----------

# A look at the "features":-

features

# COMMAND ----------

# A look at the "labels":-

labels

# COMMAND ----------

# Creating a list of Processed Features:-

import re
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

# COMMAND ----------

# A look at the list of Processed Features:-

processed_features

# COMMAND ----------

# TF-IDF

# Bag of Words - In the bag of words approach, each word has the same weight. The idea behind the TF-IDF approach is that the words that occur less in all the documents and more in individual document contribute more towards classification.

# TF-IDF is a combination of two terms. Term frequency and Inverse Document frequency. They can be calculated as:

# TF = (Frequency of a word in the document)/(Total words in the document)

# IDF = Log((Total number of documents)/(Number of documents containing the word))

# COMMAND ----------

# Install the "nltk" package:-

%%sh
pip install nltk

# COMMAND ----------

# Import the "nltk" library:-

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# COMMAND ----------

# Install the "sklearn" package:-

%%sh
pip install sklearn

# COMMAND ----------

# Import the "TfidfVectorizer" library from "sklearn", and develop a list of Processed Features:-

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

# COMMAND ----------

# A look at the Processed Features:-

processed_features

# COMMAND ----------

# A look at the length of Processed Features:-

len(processed_features)

# COMMAND ----------

# Creating a column for Processed Features in the DataFrame, and adding them as a list:-

df['processed_features'] = processed_features.tolist()

# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Building

# COMMAND ----------

# MAGIC %md
# MAGIC Prepping the data for use in the models

# COMMAND ----------

# Importing all the necessary libraries and packages:-

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import time

# COMMAND ----------

# Taking a look at the data types of all the features:-

df.dtypes

# COMMAND ----------

# Taking a look at the general shape of the DataFrame:-

df.shape

# COMMAND ----------

# Converting the "tokenized_reviewContent, processed_features, reviewID, and location" features to a String data type, so that we can Encode them in the next step:-

df['tokenized_reviewContent'] = df['tokenized_reviewContent'].astype(str)
df['processed_features'] = df['processed_features'].astype(str)
df['reviewID'] = df['reviewID'].astype(str)
df['location'] = df['location'].astype(str)

# COMMAND ----------

# MAGIC %md
# MAGIC II. Encoding Categorical Features/Columns using "LabelEncoder"

# COMMAND ----------

# Encoding the Categorical Features/Columns with dummy values, in order to add them to the models:- 

from sklearn.preprocessing import LabelEncoder
df['reviewID_encoded'] = LabelEncoder().fit_transform(df.reviewID)
df['reviewerID_encoded'] = LabelEncoder().fit_transform(df.reviewerID)
df['restaurantID_encoded'] = LabelEncoder().fit_transform(df.restaurantID)
df['reviewContent_encoded'] = LabelEncoder().fit_transform(df.reviewContent)
df['name_encoded'] = LabelEncoder().fit_transform(df.name)
df['location_encoded'] = LabelEncoder().fit_transform(df.location)
df['tokenized_reviewContent_encoded'] = LabelEncoder().fit_transform(df.tokenized_reviewContent)
df['processed_features_encoded'] = LabelEncoder().fit_transform(df.processed_features)
df['date_encoded'] = LabelEncoder().fit_transform(df.date)
df['yelpJoinDate_encoded'] = LabelEncoder().fit_transform(df.yelpJoinDate)
df['difference_in_datetime_encoded'] = LabelEncoder().fit_transform(df.difference_in_datetime)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We decided to use "sklearn's - LabelEncoder", to encode all the categorical features, as it essentially creates just one encoded column/feature for each categorical feature, instead of having multiple  columns/features for each and every category/element within that feature. It essentially creates a more manageble number of columns, and does not create an unnecessary number of columns, which would only complicate our analysis and models to come.

# COMMAND ----------

# Screenshot of the DataFrame, with all the Encoded Columns added to the far right - Please use the scroll bar below the DataFrame to scroll to the far right, to examine the encoded columns:-

df.head()

# COMMAND ----------

# Taking a look at the data types of all the features, to ensure all the encoded ones are of an "Numeric" data type:-

df.dtypes

# COMMAND ----------

# Taking a look at the REVISED general shape of the DataFrame:-

df.shape

# COMMAND ----------

# Examine the current class distribution for the "flagged" variable:-

pd.crosstab(index = df['flagged'], columns="Total count")

# COMMAND ----------

# Encoding the "flagged" column/feature with dummy values, "0" and "1", in order to effectively add it to the models:-

df['flagged_encoded'] = LabelEncoder().fit_transform(df.flagged)

# COMMAND ----------

# A quick look at the new set of features, after encodeing the "flagged" feature:-

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC III. Re-arranging all the features/columns in the DataFrame, to keep all the Numeric Data Types together and the String Data Types together. This is not just for visual purposes, but most importantly, it will enable us to conduct a train-test split more effectively later on.

# COMMAND ----------

# Re-arranging the features/columns in the Revised DataFrame:-

column_names = ["flagged", "reviewID", "reviewerID", "restaurantID", "reviewContent", "name", "location", "tokenized_reviewContent", "processed_features", "date", "yelpJoinDate", "difference_in_datetime", "flagged_encoded", "rating", "reviewUsefulCount", "friendCount", "reviewCount", "firstCount", "usefulCount", "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount", "restaurantRating", "MaximumNumberOfReviews-Day", "ReviewNumberofWords", "ReviewDeviation", "Maximum Content Similarity", "ReviewLengthCharacters", "gender_Male", "gender_female", "Positively_Rated", "avg_word_len", "reviewID_encoded", "reviewerID_encoded", "restaurantID_encoded", "reviewContent_encoded", "name_encoded", "location_encoded", "tokenized_reviewContent_encoded", "processed_features_encoded", "date_encoded", "yelpJoinDate_encoded", "difference_in_datetime_encoded"]
df = df.reindex(columns=column_names)

# COMMAND ----------

# Looking at the column data types, and the revised order of the columns/features:-

df.dtypes

# COMMAND ----------

# Taking a look at the REVISED general shape of the DataFrame after re-arranging the columns, to ensure we did not miss any columns, and retain all 45 columns:-

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC IV. Revised Correlation Analysis -
# MAGIC 
# MAGIC After encoding categorical variables, our next focus area was to develop and use a correlation matrix, to identify which features (predictive variables (x variables)) are highly correlated with each other. This forms part of our Feature Reduction process, to avoid issues of multicollinearity, and improve the predictive capabilities and the trustworthiness of our models.  

# COMMAND ----------

# Correlation matrix for all Numeric Features/Columns:-

import seaborn as sns

import matplotlib.pyplot as plt
f = plt.figure(figsize=(50, 50))

df_small = df.iloc[12:,:45]

correlation_mat = df_small.corr()

sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Restaurant")

plt.xlabel("Review Features")

plt.ylabel("Review Features")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see from the vertical scale on the far right, the lighter and the darker the color, the higher the correlation there is. Keeping a threshold of 0.60, we decided to remove those variables that had a correlation of 0.6 and above, in order to avoid the problem of multicollinearity. We felt a correlation of 0.6 or higher would capture most of the multicollinearity issues, and thus we used 0.6 as our threshold.

# COMMAND ----------

# Another look at the Correlation Matrix, in a different format:-

correlation_mat

# COMMAND ----------

# MAGIC %md
# MAGIC #Dimension Reduction

# COMMAND ----------

# MAGIC %md
# MAGIC We decided to use a Correlation threshold of 0.60 (>=0.60), and thus remove any predictive (x) features with a correlation of 0.60 or above, with other predictive (x) features, to reduce the effects of multicollinearity in our predictive models.

# COMMAND ----------

# Finding highly correlated features (>=0.60):-

correlated_features = set()

for i in range(len(correlation_mat.columns)):
    for j in range(i):
        if abs(correlation_mat.iloc[i, j]) >= 0.6:
            colname = correlation_mat.columns[i]
            correlated_features.add(colname)

# COMMAND ----------

# Checking the number of highly correlated features:-

len(correlated_features)

# COMMAND ----------

# Looking at all the highly correlated features:-

print (correlated_features)

# COMMAND ----------

# Creating a revised Dataframe created after removal of highly correlated features of 0.60 and above:-

column_names = ["flagged_encoded", "rating", "reviewUsefulCount", "friendCount",  "tipCount", "restaurantRating", "MaximumNumberOfReviews-Day", "ReviewNumberofWords", "ReviewDeviation", "Maximum Content Similarity", "gender_Male", "avg_word_len", "reviewID_encoded", "reviewerID_encoded", "restaurantID_encoded", "reviewContent_encoded", "name_encoded", "location_encoded", "processed_features_encoded", "date_encoded", "difference_in_datetime_encoded"]
df = df.reindex(columns=column_names)

# COMMAND ----------

# A look at the revised shape of our DataFrame:-

df.shape

# COMMAND ----------

# A look at the revised basic Correlation Matrix:-

df.corr()

# COMMAND ----------

# A look at the revised correlation matrix, using the "Seaborn" package, for a more informative visual and analysis. We now have all correlations of < 0.6:-

import seaborn as sns

import matplotlib.pyplot as plt
f = plt.figure(figsize=(19, 15))
# taking all rows but only 26 columns
df_small = df.iloc[:,:]

correlation_mat = df_small.corr()

sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of Restaurant")

plt.xlabel("Review Features")

plt.ylabel("Review Features")

plt.show()

# COMMAND ----------

# A quick look at the remaining features in our DataFrame:-

df.dtypes

# COMMAND ----------

# Screenshot of the first few rows in the DataFrame:-

df.head()

# COMMAND ----------

# A quick look at the revised shape of the DataFrame:-

df.shape

# COMMAND ----------

# Examine the current class distribution for the "flagged_encoded" variable:-

pd.crosstab(index = df['flagged_encoded'], columns="Total count")

# COMMAND ----------

# MAGIC %md
# MAGIC # I. Logistic Regression 

# COMMAND ----------

# MAGIC %md
# MAGIC In order to run our Logistic Regression Model, our first order of business was to perform an "80-20 Train-Test Split" on the data, in order to train the model on 80% of the data, and then test it on the balance 20% of the data, to check how well our model works.

# COMMAND ----------

# Loading all the necessary packages and libraies, such as "sklearn":-

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

# COMMAND ----------

# Replace infinity values:-

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# COMMAND ----------

# Drop all empty cells:-

df.dropna(inplace=True)

# COMMAND ----------

# Train-test split (80-20 Split):-

xcols = df.columns[1 : len(df.columns)].to_list()
print(xcols,'\n')

X_train, X_test, y_train, y_test = train_test_split(df[xcols], df['flagged_encoded'], 
                                                    train_size=0.8, random_state=1)
print('training data:', X_train.shape)
print('test data:', X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC For the Logistic Regression Analysis, we essentially developed 2 sub-models here:-                                                                                                                     1) Logistic Regression - Including ALL the Features                                                                                                                                                    2) Logistic Regression - Including only to TOP 5 largest Positive and Negative features

# COMMAND ----------

# Importing and Loading all the necessary packages:-

# data packages
import pandas as pd
import numpy as np

# algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# organizing tests
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# some metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC #1) LR Model 1 - Logistic Regression - Including ALL the Features

# COMMAND ----------

# MAGIC %md
# MAGIC We first developed a Linear Regression model, using all the features in the DataFrame to make predictions if a particular review is Fake or True. The analysis is seen below.

# COMMAND ----------

# Defining/Performing, Fitting and checking the accuracy of the Logistic regression model, using ALL the features/columns:-

log_reg = LogisticRegression(solver='lbfgs', max_iter=1000000000)
clf_reg = log_reg.fit(X_train, y_train)
print('training accuracy: {}'.format(clf_reg.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(clf_reg.score(X_test, y_test).round(4)))

confusionmat = confusion_matrix(y_test,clf_reg.predict(X_test))
print("Confusion Matrix :\n", confusionmat)
    
classirep = classification_report(y_test,clf_reg.predict(X_test))
print("Classification Report :\n", classirep)
    
mcc = matthews_corrcoef(y_test, clf_reg.predict(X_test))
print("Mathew's Correlation Coeffiecient MCC :", mcc)
            
precisionrecall = precision_recall_fscore_support(y_test, clf_reg.predict(X_test), average=None)
print("Precision Recall Score:", precisionrecall)

print('Overall training accuracy: {}'.format(clf_reg.score(X_train, y_train).round(4)))
print('Overall test accuracy: {}'.format(clf_reg.score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_01 = clf_reg.predict(X_test)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_01, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_01, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_01, pos_label=1)))
cm01 = confusion_matrix(y_test, y_pred_test_01)
tn, fp, fn, tp = cm01.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm01)

# COMMAND ----------

# MAGIC %md
# MAGIC The above results is the output of this first model. Essentially, it does quite well in terms of accuracy, precision, recall and f1-score. The confusion matrix also depicts a decent model in terms of identifying True Positives (TP) and True Negatives (TN).

# COMMAND ----------

# Screenshot of the DataFrame:-

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #2) LR Model 2 - Logistic Regression - Including only to TOP 5 largest Positive and Negative features

# COMMAND ----------

# MAGIC %md
# MAGIC We have several feaures in our model, but Which are the Top 5 largest positive and negative features, a total of 10 features, which have the greatest influence on the "y" variable - "flagged_encoded"? This was our main focus for the next part of our analysis. Let's take a deeper look at this, and then analyze the coefficients associated with each of these top 5 largest positive and negative features. 

# COMMAND ----------

# Put the coefficients into a new dataframe and identify the top 5 largest positive and negative features:-

coef = pd.concat([pd.DataFrame(xcols),pd.DataFrame(np.transpose(clf_reg.coef_))], axis = 1)
coef.columns = ['feature','coefficient']
coef.sort_values(by=['coefficient'], ascending=False, inplace=True)
# examine the features with the largest coefficients
print('Five largest positive features:\n', coef.head(5), '\n')
print('Five largest negative features:\n', coef.tail(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Below is a list of the Top 5 largest positive and negative features as identified above.

# COMMAND ----------

# Putting the top 5 largest positive and negative features into a list:-

xcols2 = coef.feature[0:5].to_list()
xcols2 += coef.feature[-5:].to_list()
print(xcols2)

# COMMAND ----------

# Performing a revised train-test split (80-20 Split), using only the top 5 largest positive and negative features:-

X_train2, X_test2, y_train2, y_test2 = train_test_split(df[xcols2], df['flagged_encoded'], 
                                                    train_size=0.8, random_state=1)
print('training data:', X_train2.shape)
print('test data:', X_test2.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC We now develop a second regression model, using only this reduced feature list. We decide to do this to see if by focussing only on these top 5 largest positive and negative features, it will give our model a lift, to see if it improves the model in any way.

# COMMAND ----------

# Re-Running the Logistic Regression Model after reducing the feature set to the top 5 largest positive and negative features:-

log_reg_2 = LogisticRegression(solver='lbfgs', max_iter=1000000000)
clf_reg_2 = log_reg_2.fit(X_train2, y_train2)

print('training accuracy: {}'.format(clf_reg_2.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(clf_reg_2.score(X_test2, y_test2).round(4)))

confusionmat2 = confusion_matrix(y_test2,clf_reg_2.predict(X_test2))
print("Confusion Matrix :\n", confusionmat2)
    
classirep2 = classification_report(y_test2,clf_reg_2.predict(X_test2))
print("Classification Report :\n", classirep2)
    
mcc2 = matthews_corrcoef(y_test2, clf_reg_2.predict(X_test2))
print("Mathew's Correlation Coeffiecient MCC :", mcc2)
            
precisionrecall2 = precision_recall_fscore_support(y_test2, clf_reg_2.predict(X_test2), average=None)
print("Precision Recall Score:", precisionrecall2)

print('Overall training accuracy: {}'.format(clf_reg_2.score(X_train2, y_train2).round(4)))
print('Overall test accuracy: {}'.format(clf_reg_2.score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_02 = clf_reg_2.predict(X_test2)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_02, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_02, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_02, pos_label=1)))
cm02 = confusion_matrix(y_test2, y_pred_test_02)
tn, fp, fn, tp = cm02.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm02)

# COMMAND ----------

# MAGIC %md
# MAGIC The above results is the output of this second model. Essentially, it also does quite well in terms of accuracy, precision, recall and f1-score. The confusion matrix also depicts a decent model in terms of identifying True Positives (TP) and True Negatives (TN).

# COMMAND ----------

# MAGIC %md
# MAGIC Which Model performs better?
# MAGIC Both the models are very close in terms of their performance, but Model 1 - "Logistic Regression - Including ALL the Features" performs better between the 2, as it can identify the greatest number of True Positives (TP) and True Negatives (TN), and also does better when we look at the evaluation metrics, precision, recall, f1-score, as well as the overall accuracy, which are all higher for Model 1, when compared to Model 2.

# COMMAND ----------

# MAGIC %md
# MAGIC But wait a minute, looking at the imbalanced nature of our data, i.e. there are many more "N - not-fake" reviews when compared to much fewer "Y - fake" reviews, wouldn't that affect the predictive power and believability of the model? The model may be able to predict the majority ("N - not-fake"), but what abot the minority ("Y - fake")? As seen earlier, 17,823 "N - not-fake" reviews and only 5,697 "Y - fake" reviews in our dataset. We will need to account for this. This is the focus of the next section of our analysis, to try and "Remedy the Imbalanced Classification".  

# COMMAND ----------

# MAGIC %md
# MAGIC #Remedying Imbalanced Classification

# COMMAND ----------

# Importing and Loading all the necessary packages:-

# data packages
import pandas as pd
import numpy as np

# algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# organizing tests
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# some metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC As mentioned earlier, our data does seem to be imbalanced, given the nature, purpose and analyis we are performing. One would expect there to be much less "Fake" reviews, when compared to "Not-Fake" reviews. Please see below.

# COMMAND ----------

# Examine the current class distribution:-

pd.crosstab(index = df['flagged_encoded'], columns="Total count")

# COMMAND ----------

# MAGIC %md
# MAGIC We will need to correct for this imbalance, to see if our model improves.

# COMMAND ----------

# MAGIC %md
# MAGIC We will use a method called the "Modify Algorithm Approach" to remedy this imbalance, which uses a "Balanced" class weight attribute. Once again, we will use this method, and build 2 sub-models:-      1) Using ALL the Features/X columns                                                                                                                                                                      2) Using ONLY the TOP 5 largest positive and negative features, as the X columns.  

# COMMAND ----------

# MAGIC %md
# MAGIC Modify Algorithm Approach -

# COMMAND ----------

# MAGIC %md
# MAGIC #3) LR Model 3 Balanced - Logistic Regression - Using ALL the Features/X columns

# COMMAND ----------

"""Remedying Class Imbalance using ALL the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ALL the X columns:-

log_reg_mod1 = LogisticRegression(solver='lbfgs', max_iter=1000000000, class_weight='balanced')
clf_reg_mod1 = log_reg_mod1.fit(X_train, y_train)
print('training accuracy: {}'.format(clf_reg_mod1.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(clf_reg_mod1.score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_1 = clf_reg_mod1.predict(X_test)
print('  Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_1, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_1, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_1, pos_label=1)))
cm = confusion_matrix(y_test, y_pred_test_1)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X1, y1 = df[xcols], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_1 = clf_reg_mod1.predict(X1)
print('Accuracy: {:.3f}'.format(clf_reg_mod1.score(X1, y1)))
print('Recall: {:.3f}'.format(recall_score(y1, y_pred_1, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y1, y_pred_1, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y1, y_pred_1, pos_label=1)))
cm1 = confusion_matrix(y1, y_pred_1)
tn, fp, fn, tp = cm1.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm1)

# COMMAND ----------

# MAGIC %md
# MAGIC This model does quite well, especially when you compare all the evaluation metrics across the 3 models till now.

# COMMAND ----------

# MAGIC %md
# MAGIC #4) LR Model 4 Balanced - Using ONLY the TOP 5 largest positive and negative features, as the X columns

# COMMAND ----------

"""Remedying Class Imbalance using ONLY the top 5 largest positive and negative features, as the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ONLY the top 5 largest positive and negative features, as the X columns:-

log_reg_mod1_A = LogisticRegression(solver='lbfgs', max_iter=1000000000, class_weight='balanced')
clf_reg_mod1_A = log_reg_mod1_A.fit(X_train2, y_train2)
print('training accuracy: {}'.format(clf_reg_mod1_A.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(clf_reg_mod1_A.score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_1_A = clf_reg_mod1_A.predict(X_test2)
print('  Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_1_A, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_1_A, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_1_A, pos_label=1)))
cm2 = confusion_matrix(y_test2, y_pred_test_1_A)
tn, fp, fn, tp = cm2.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm2)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X1_A, y1_A = df[xcols2], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_1_A = clf_reg_mod1_A.predict(X1_A)
print('Accuracy: {:.3f}'.format(clf_reg_mod1_A.score(X1_A, y1_A)))
print('Recall: {:.3f}'.format(recall_score(y1_A, y_pred_1_A, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y1_A, y_pred_1_A, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y1_A, y_pred_1_A, pos_label=1)))
cm3 = confusion_matrix(y1_A, y_pred_1_A)
tn, fp, fn, tp = cm3.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm3)

# COMMAND ----------

# MAGIC %md
# MAGIC Once again, this model does quite well, especially when you compare all the evaluation metrics across the 4 models till now.

# COMMAND ----------

# MAGIC %md
# MAGIC We will perform a model comparison at the end of all our models and analysis. Let us now look at the ROC Curve to evaluate it better.

# COMMAND ----------

# MAGIC %md
# MAGIC ROC Curve

# COMMAND ----------

"""On the Original data using ALL the features"""

# COMMAND ----------

# Simple ROC curve:-

log_reg_Ev = LogisticRegression(solver='lbfgs', max_iter=1000000000)
clf_reg_Ev = log_reg_Ev.fit(X_train, y_train)

plot_roc_curve(clf_reg_Ev, X_test, y_test)
x_1 = np.linspace(0, 1.0)
plt.plot(x_1, x_1, color='grey', ls='--')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC End of Logistic Regression model

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Decision Tree

# COMMAND ----------

# MAGIC %md
# MAGIC #1) DT Model 1 - Using ALL the Features/X columns 

# COMMAND ----------

# Importing the libraries and packages and building a Decision Tree model using ALL the Features/X columns:-

from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier(criterion="gini")

# COMMAND ----------

# Train the model:-

model3.fit(X_train, y_train)

# COMMAND ----------

# Run the model and get the metrics:-

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('training accuracy: {}'.format(model3.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(model3.score(X_test, y_test).round(4)))

confusionmat4 = confusion_matrix(y_test,model3.predict(X_test))
print("Confusion Matrix :\n", confusionmat4)
    
classirep4 = classification_report(y_test,model3.predict(X_test))
print("Classification Report :\n", classirep4)
    
mcc4 = matthews_corrcoef(y_test, model3.predict(X_test))
print("Mathew's Correlation Coeffiecient MCC :", mcc4)
            
precisionrecall4 = precision_recall_fscore_support(y_test, model3.predict(X_test), average=None)
print("Precision Recall Score:", precisionrecall4)

print('Overall training accuracy: {}'.format(model3.fit(X_train, y_train).score(X_train, y_train).round(4)))
print('Overall test accuracy: {}'.format(model3.fit(X_train, y_train).score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_03 = model3.fit(X_train, y_train).predict(X_test)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_03, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_03, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_03, pos_label=1)))
cm03 = confusion_matrix(y_test, y_pred_test_03)
tn, fp, fn, tp = cm03.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm03)

# COMMAND ----------

# MAGIC %md
# MAGIC #2) DT Model 2 - Using Top 5 largest positive and negative features

# COMMAND ----------

## Re-running the Decision Tree model using ONLY the top 5 largest positive and negative features, as the X columns:-

from sklearn.tree import DecisionTreeClassifier
model4= DecisionTreeClassifier(criterion="gini")

# COMMAND ----------

# Train the model:-

model4.fit(X_train2, y_train2)

# COMMAND ----------

# Run the model and get the metrics:-

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('training accuracy: {}'.format(model4.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(model4.score(X_test2, y_test2).round(4)))

confusionmat5 = confusion_matrix(y_test2,model4.predict(X_test2))
print("Confusion Matrix :\n", confusionmat5)
    
classirep5 = classification_report(y_test2,model4.predict(X_test2))
print("Classification Report :\n", classirep5)
    
mcc5 = matthews_corrcoef(y_test2, model4.predict(X_test2))
print("Mathew's Correlation Coeffiecient MCC :", mcc5)
            
precisionrecall5 = precision_recall_fscore_support(y_test2, model4.predict(X_test2), average=None)
print("Precision Recall Score:", precisionrecall5)

print('Overall training accuracy: {}'.format(model4.fit(X_train2, y_train2).score(X_train2, y_train2).round(4)))
print('Overall test accuracy: {}'.format(model4.fit(X_train2, y_train2).score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_04 = model4.fit(X_train2, y_train2).fit(X_train2, y_train2).predict(X_test2)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_04, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_04, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_04, pos_label=1)))
cm04 = confusion_matrix(y_test2, y_pred_test_04)
tn, fp, fn, tp = cm04.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm04)

# COMMAND ----------

# MAGIC %md
# MAGIC Modify Algorithm Approach -

# COMMAND ----------

# MAGIC %md 
# MAGIC #3) DT Model 3 Balanced - Using ALL the Features/X columns

# COMMAND ----------

"""Remedying Class Imbalance using ALL the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ALL the X columns:-

model5= DecisionTreeClassifier(criterion="gini", class_weight='balanced')
clf_dt_mod5 = model5.fit(X_train, y_train)
print('training accuracy: {}'.format(clf_dt_mod5.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(clf_dt_mod5.score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_5 = clf_dt_mod5.predict(X_test)
print('  Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_5, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_5, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_5, pos_label=1)))
cm5 = confusion_matrix(y_test, y_pred_test_5)
tn, fp, fn, tp = cm5.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm5)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X5, y5 = df[xcols], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_5 = clf_dt_mod5.predict(X5)
print('Accuracy: {:.3f}'.format(clf_dt_mod5.score(X5, y5)))
print('Recall: {:.3f}'.format(recall_score(y5, y_pred_5, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y5, y_pred_5, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y5, y_pred_5, pos_label=1)))
cm6 = confusion_matrix(y5, y_pred_5)
tn, fp, fn, tp = cm6.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm6)

# COMMAND ----------

# MAGIC %md
# MAGIC #4) DT Model 4 Balanced - Using ONLY the TOP 5 largest positive and negative features, as the X columns

# COMMAND ----------

"""Remedying Class Imbalance using ONLY the top 5 largest positive and negative features, as the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ONLY the top 5 largest positive and negative features, as the X columns:-

model6= DecisionTreeClassifier(criterion="gini", class_weight='balanced')
clf_dt_mod6 = model6.fit(X_train2, y_train2)
print('training accuracy: {}'.format(clf_dt_mod6.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(clf_dt_mod6.score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_6 = clf_dt_mod6.predict(X_test2)
print('  Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_6, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_6, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_6, pos_label=1)))
cm7 = confusion_matrix(y_test2, y_pred_test_6)
tn, fp, fn, tp = cm7.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm7)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X6, y6 = df[xcols2], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_6 = clf_dt_mod6.predict(X6)
print('Accuracy: {:.3f}'.format(clf_dt_mod6.score(X6, y6)))
print('Recall: {:.3f}'.format(recall_score(y6, y_pred_6, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y6, y_pred_6, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y6, y_pred_6, pos_label=1)))
cm8 = confusion_matrix(y6, y_pred_6)
tn, fp, fn, tp = cm8.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm8)

# COMMAND ----------

# MAGIC %md
# MAGIC ROC Curve 

# COMMAND ----------

# Simple ROC curve:-

model_ROC= DecisionTreeClassifier(criterion="gini")
clf_dt_mod_ROC = model_ROC.fit(X_train, y_train)

plot_roc_curve(clf_dt_mod_ROC, X_test, y_test)
x_2 = np.linspace(0, 1.0)
plt.plot(x_2, x_2, color='grey', ls='--')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC End of the Decision Tree model

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC #1) RF Model 1 - Using ALL the Features/X columns

# COMMAND ----------

# Importing the libraries and packages and train the model, using ALL the Features/X columns:-

from sklearn.ensemble import RandomForestClassifier

model_ran = RandomForestClassifier(n_estimators=200, random_state=0)
model_ran.fit(X_train, y_train)

# COMMAND ----------

# Run the model and get the metrics:-

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('training accuracy: {}'.format(model_ran.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(model_ran.score(X_test, y_test).round(4)))

confusionmat_ran = confusion_matrix(y_test,model_ran.predict(X_test))
print("Confusion Matrix :\n", confusionmat_ran)
    
classirep_ran = classification_report(y_test,model_ran.predict(X_test))
print("Classification Report :\n", classirep_ran)
    
mcc_ran = matthews_corrcoef(y_test2, model_ran.predict(X_test))
print("Mathew's Correlation Coeffiecient MCC :", mcc_ran)
            
precisionrecall_ran = precision_recall_fscore_support(y_test, model_ran.predict(X_test), average=None)
print("Precision Recall Score:", precisionrecall_ran)

print('Overall training accuracy: {}'.format(model_ran.fit(X_train, y_train).score(X_train, y_train).round(4)))
print('Overall test accuracy: {}'.format(model_ran.fit(X_train, y_train).score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_05 = model_ran.fit(X_train, y_train).fit(X_train, y_train).predict(X_test)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_05, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_05, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_05, pos_label=1)))
cm05 = confusion_matrix(y_test, y_pred_test_05)
tn, fp, fn, tp = cm05.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm05)

# COMMAND ----------

# MAGIC %md
# MAGIC #2) RF Model 2 - Using Top 5 largest positive and negative features

# COMMAND ----------

## Re-running and training the Random Forest model using ONLY the top 5 largest positive and negative features, as the X columns:-

from sklearn.ensemble import RandomForestClassifier

model_ran2 = RandomForestClassifier(n_estimators=200, random_state=0)
model_ran2.fit(X_train2, y_train2)

# COMMAND ----------

# Run the model and get the metrics:-

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('training accuracy: {}'.format(model_ran2.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(model_ran2.score(X_test2, y_test2).round(4)))

confusionmat_ran2 = confusion_matrix(y_test2,model_ran2.predict(X_test2))
print("Confusion Matrix :\n", confusionmat_ran2)
    
classirep_ran2 = classification_report(y_test2,model_ran2.predict(X_test2))
print("Classification Report :\n", classirep_ran2)
    
mcc_ran2 = matthews_corrcoef(y_test2, model_ran2.predict(X_test2))
print("Mathew's Correlation Coeffiecient MCC :", mcc_ran2)
            
precisionrecall_ran2 = precision_recall_fscore_support(y_test2, model_ran2.predict(X_test2), average=None)
print("Precision Recall Score:", precisionrecall_ran2)

print('Overall training accuracy: {}'.format(model_ran2.fit(X_train2, y_train2).score(X_train2, y_train2).round(4)))
print('Overall test accuracy: {}'.format(model_ran2.fit(X_train2, y_train2).score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_06 = model_ran2.fit(X_train2, y_train2).fit(X_train2, y_train2).predict(X_test2)
print('  Overall Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_06, pos_label=1)))
print('  Overall Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_06, pos_label=1)))
print('  Overall F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_06, pos_label=1)))
cm06 = confusion_matrix(y_test2, y_pred_test_06)
tn, fp, fn, tp = cm06.ravel()
specificity = tn / (tn+fp)
print('  Overall Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm06)

# COMMAND ----------

# MAGIC %md
# MAGIC Modify Algorithm Approach -

# COMMAND ----------

# MAGIC %md
# MAGIC #3) RF Model 3 Balanced - Using ALL the Features/X columns

# COMMAND ----------

"""Remedying Class Imbalance using ALL the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ALL the X columns/features:-

model_ran_6 = RandomForestClassifier(n_estimators=200, random_state=0, class_weight='balanced')
clf_rf_mod6 = model_ran_6.fit(X_train, y_train)
print('training accuracy: {}'.format(clf_rf_mod6.score(X_train, y_train).round(4)))
print('test accuracy: {}'.format(clf_rf_mod6.score(X_test, y_test).round(4)))
print('other test stats:')
y_pred_test_rf_6 = clf_rf_mod6.predict(X_test)
print('  Recall: {:.3f}'.format(recall_score(y_test, y_pred_test_rf_6, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test, y_pred_test_rf_6, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test, y_pred_test_rf_6, pos_label=1)))
cm9 = confusion_matrix(y_test, y_pred_test_rf_6)
tn, fp, fn, tp = cm9.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm9)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X9, y9 = df[xcols], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_rf_6 = clf_rf_mod6.predict(X9)
print('Accuracy: {:.3f}'.format(clf_rf_mod6.score(X9, y9)))
print('Recall: {:.3f}'.format(recall_score(y9, y_pred_rf_6, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y9, y_pred_rf_6, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y9, y_pred_rf_6, pos_label=1)))
cm10 = confusion_matrix(y9, y_pred_rf_6)
tn, fp, fn, tp = cm10.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm10)

# COMMAND ----------

# MAGIC %md
# MAGIC #4) RF Model 4 Balanced - Using ONLY the TOP 5 largest positive and negative features, as the X columns

# COMMAND ----------

"""Remedying Class Imbalance using ONLY the top 5 largest positive and negative features, as the X columns."""

# COMMAND ----------

# Modify the algorithm's objective with a class_weight attribute, using ONLY the top 5 largest positive and negative features, as the X columns:-

model_ran_7 = RandomForestClassifier(n_estimators=200, random_state=0, class_weight='balanced')
clf_rf_mod7 = model_ran_7.fit(X_train2, y_train2)
print('training accuracy: {}'.format(clf_rf_mod7.score(X_train2, y_train2).round(4)))
print('test accuracy: {}'.format(clf_rf_mod7.score(X_test2, y_test2).round(4)))
print('other test stats:')
y_pred_test_rf_7 = clf_rf_mod7.predict(X_test2)
print('  Recall: {:.3f}'.format(recall_score(y_test2, y_pred_test_rf_7, pos_label=1)))
print('  Precision: {:.3f}'.format(precision_score(y_test2, y_pred_test_rf_7, pos_label=1)))
print('  F1 score: {:.3f}'.format(f1_score(y_test2, y_pred_test_rf_7, pos_label=1)))
cm11 = confusion_matrix(y_test2, y_pred_test_rf_7)
tn, fp, fn, tp = cm11.ravel()
specificity = tn / (tn+fp)
print('  Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm11)

# COMMAND ----------

# Creating and inputing the X and y columns into two separate DataFrames:-

X10, y10 = df[xcols2], df['flagged_encoded']

# COMMAND ----------

# How well does it do on the original dataset?:-

y_pred_rf_7 = clf_rf_mod7.predict(X10)
print('Accuracy: {:.3f}'.format(clf_rf_mod7.score(X10, y10)))
print('Recall: {:.3f}'.format(recall_score(y10, y_pred_rf_7, pos_label=1)))
print('Precision: {:.3f}'.format(precision_score(y10, y_pred_rf_7, pos_label=1)))
print('F1 score: {:.3f}'.format(f1_score(y10, y_pred_rf_7, pos_label=1)))
cm11 = confusion_matrix(y10, y_pred_rf_7)
tn, fp, fn, tp = cm11.ravel()
specificity = tn / (tn+fp)
print('Specificity: {:.3f}'.format(specificity))
print('confusion matrix:\n', cm11)

# COMMAND ----------

# MAGIC %md
# MAGIC ROC Curve

# COMMAND ----------

# Simple ROC curve:-

model_rf_ROC = RandomForestClassifier(n_estimators=200, random_state=0)
clf_rf_mod_ROC = model_rf_ROC.fit(X_train, y_train)

plot_roc_curve(clf_rf_mod_ROC, X_test, y_test)
x_3 = np.linspace(0, 1.0)
plt.plot(x_3, x_3, color='grey', ls='--')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC End of the Random Forest model

# COMMAND ----------

# MAGIC %md
# MAGIC #Model Selection

# COMMAND ----------

# MAGIC %md
# MAGIC To Summarize, we essentially built and ran 3 Primary Models - 1) Linear Regression, 2) Decision Tree, and 3) Random Forest.
# MAGIC For each of these 3 models, we built 4 sub-models each, a total of 12 sub-models:-
# MAGIC 1) Model built on an Unbalanced Dataset, and using ALL features/columns as the X variables.
# MAGIC 2) Model built on an Unbalanced Dataset, and using ONLY to TOP 5 largest positive and negative features as the X columns.
# MAGIC 3) Model built on a Balanced Dataset, and using ALL features/columns as the X variables.
# MAGIC 4) Model built on a Balanced Dataset, and using ONLY to TOP 5 largest positive and negative features as the X columns.

# COMMAND ----------

# MAGIC %md
# MAGIC Which is our model of choice?:-
# MAGIC After evaluating all the 12 sub-models, we decided to recommend using either of the 2 Random Forest Model's built on a Balanced Dataset. Essentially we recommend using either:-
# MAGIC 1) "Random Forest (RF) Model 3 Balanced - Using ALL the Features/X columns" - Model built on a Balanced Dataset, and using ALL features/columns as the X variables
# MAGIC OR
# MAGIC 2) "Random Forest (RF) Model 4 Balanced - Using ONLY the TOP 5 largest positive and negative features, as the X columns" - Model built on a Balanced Dataset, and using ONLY to TOP 5 largest positive and negative features as the X columns.

# COMMAND ----------

# MAGIC %md
# MAGIC We decided that we could choose either one, as they are essentially very close in terms of all the evaluation matrices, such as Recall, Precision, F1 Score, Specificity, and Accuracy. One performs slightly better in certain metrics, while the other performs slightly better on the other metrics. Due to the imbalanced nature of the data, these 2 models correct/accomodate for such imbalances, and thus outperform the regular Unbalanced Models across all the primary Models used, Logistic Regression, Decison Tree and Random Forest. Due to the imbalanced nature of the data, we do not rely on the "Accuracy" values across all 12 models. We essentially use the remaining evaluation metrics to choose our models.
# MAGIC 
# MAGIC With the data we have, it had to be corrected for imbalance, in order to develop more robust and trustworty models. Running the 2 selected models on multiple itterations of the datasets at Yelp, would enable them to identify the 1 of the 2 selected models that works best for them, on average. The key features identified in spotting "Fake Reviews" are target variables/features for Yelp to look at to quickly spot possible fake reviews. By using these machine learning models, the company can automate this process even further, and thus comb through Big Data, which will enable them to significantly reduce possible fake reviews, and maybe even eliminate them in the long-run, thus ensuring they stay relevant among their customers and clients.

# COMMAND ----------

# MAGIC %md
# MAGIC #Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC These models can be used for:-
# MAGIC 1) Determining fake vs real reviews
# MAGIC 2) Help achieve better results & take corrective actions
# MAGIC 3) Can be used for similar business problems
# MAGIC 4) Build brand loyalty.

# COMMAND ----------

# MAGIC %md
# MAGIC Using these models, Yelp can run several of their datasets through them, and identify possible fake reviews.
# MAGIC 
# MAGIC Through the power of Text Mining, we have essentially been able to build effective machine learning models which have application across several types of companies that have an online presence, and who use a review system. The model would only be needed to be tweaked to the specific company, based on the features in their dataset. 

# COMMAND ----------

# MAGIC %md
# MAGIC End of Project - Thank you for Reading!

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix - Additional Models We Tried

# COMMAND ----------

# MAGIC %md
# MAGIC We tried 2 other models, the "XG Boost Model" and the "SVM" models, but did not include them in our main report above, since they were taking too long to run. Below is the code we developed for them. Including them here just for your reference.

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. XG Boost Model 

# COMMAND ----------

###Xg boost:

# COMMAND ----------

#from xgboost import XGBClassifier

# COMMAND ----------

#xgb_model = XGBClassifier()
#xgb_model.fit(X_train, y_train)

# COMMAND ----------

# Predicting the Test set results
#xgb_pred = xgb_model.predict(X_test)

# COMMAND ----------

#from sklearn.metrics import classification_report
#print(classification_report(y_test, xgb_pred))

# COMMAND ----------

##Xg boost confusion matrix 

# COMMAND ----------

#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,xgb_pred)
#cm

# COMMAND ----------

#import matplotlib.pyplot as plt
#import seaborn as sns 

#f, ax=plt.subplots(figsize=(5,5))
#sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
#plt.xlabel("xgb_pred")
#plt.ylabel("y_test")
#plt.show()

# COMMAND ----------

#KNN 

# COMMAND ----------

#from sklearn.neighbors import KNeighborsClassifier
#classifier1 = KNeighborsClassifier(n_neighbors = 5) #no of neighbors is hyper parameter
#classifier1.fit(X_train, y_train)

# COMMAND ----------

#predictions = classifier1.predict(X_test)

# COMMAND ----------

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#print(accuracy_score(y_test, predictions))

# COMMAND ----------

# End of the XG Boost Model

# COMMAND ----------

# MAGIC %md
# MAGIC # V. SVM - Support Vector Machine 

# COMMAND ----------

# MAGIC %md
# MAGIC Please Note - Had to cancel the SVM model for our analysis, due to the size and nature of our dataset, the SVM code was taking extremely long to run. Due to this we decided to terminate the model. Below is the SVM model we started building, but scrapped. Included here just as an FYI.

# COMMAND ----------

# Creating, Fitting and Testing a Support Vector Machine Model on the training data:-

#svm = SVC(kernel = 'linear')
#clf_svm = svm.fit(X_train, y_train)

#print('training accuracy: {}'.format(clf_svm.score(X_train, y_train).round(4)))
#print('test accuracy: {}'.format(clf_svm.score(X_test, y_test).round(4)))

# COMMAND ----------

# Regularization with the C parameter:-

#cset = [.001, .01, .1, 1, 10]
#for i in cset:
    #print('C =', i)
    #svm1 = SVC(kernel = 'linear', C = i)
    #clf_svm1 = svm1.fit(X_train, y_train)
    #print('training accuracy: {}'.format(clf_svm1.score(X_train, y_train).round(4)))
    #print('test accuracy: {}'.format(clf_svm1.score(X_test, y_test).round(4)), '\n')

# COMMAND ----------

# Cross-validation with SVM:-

#svm2 = SVC(kernel = 'linear', C = .01)
#scores2 = cross_val_score(svm2, df[xcols], df['flagged'], cv=5)
#print(scores2)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

# COMMAND ----------

# Re-visit the Logistic Regression with Cross-Validation

# COMMAND ----------

# Regularization w/ logistic regression with the C parameter:-

#cset1 = [.001, .01, .1, 1, 10]
#for i in cset1:
    #print('C =', i)
    #log_reg1 = LogisticRegression(solver='lbfgs', max_iter=1000000000, C=i)
    #clf_reg1 = log_reg1.fit(X_train, y_train)
    #print('training accuracy: {}'.format(clf_reg1.score(X_train, y_train).round(4)))
    #print('test accuracy: {}'.format(clf_reg1.score(X_test, y_test).round(4)), '\n')

# COMMAND ----------

# Cross-validation with logistic regression:-

#for i in cset1:
    #print('C =', i)
    #log_reg2 = LogisticRegression(solver='lbfgs', max_iter=1000000000, C=i)
    #scores3 = cross_val_score(log_reg2, df[xcols], df['flagged'], cv=5)
    #print(scores)
    #print("Accuracy: %0.3f (+/- %0.3f)" % (scores3.mean(), scores3.std() * 2), '\n')

# COMMAND ----------

# Train the best model for prediction and/or validation:-

#log_reg3 = LogisticRegression(solver='lbfgs', max_iter=1000000000, C=.01)
#clf_reg2 = log_reg.fit(df[xcols], df['flagged'])

# COMMAND ----------

# Get a list of values for each feature that we can use to retrieve a prediction:-

#vals = []
#print('Values for features w/ positive coefficients:')
#for i in xcols[0:10]:
    #val = df[i].mean() + df[i].std()
    #print(' ', i, ':', val)
    #vals.append(val)
    
#print('\nValues for features w/ negative coefficients:')
#for i in xcols[10:20]:
    #val = df[i].mean() - df[i].std()
    #print(' ', i, ':', val)
    #vals.append(val)

# COMMAND ----------

# Get predictions and probabilities for Logistic Regression:-

#ypred = clf_reg2.predict([vals])
#yprob = clf_reg2.predict_proba([vals])
#print('predicted class:', ypred[0])
#print('probability for 0 and 1 classes:', yprob[0])
#print('probability for class 1:', yprob[0][1])

# COMMAND ----------

# Get predictions and probabilities for SVM:-

#svm3 = SVC(kernel = 'linear', C = .01, probability=True)
#clf_svm2 = svm3.fit(df[xcols], df['flagged'])
#ypred = clf_svm2.predict([vals])
#yprob = clf_svm2.predict_proba([vals])
#print('predicted class:', ypred[0])
#print('probability for 0 and 1 classes:', yprob[0])
#print('probability for class 1:', yprob[0][1])

# COMMAND ----------

# End of the SVM model

# COMMAND ----------

# MAGIC %md
# MAGIC End of Appendix
