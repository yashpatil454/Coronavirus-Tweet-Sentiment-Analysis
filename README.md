<h1 align="center"> Coronavirus Tweet Sentiment Analysis </h1>

<p align="center"> </p>
<h2> :floppy_disk: Problem Statement and Project Description</h2>
<p>Coronavirus Tweet Sentiment Analysis NLP project aims to classify tweets related to Coronavirus into three categories: negative, neutral, and positive using natural language processing techniques such as sentiment analysis, text classification, and machine learning algorithms. The goal is to gain a better understanding of the overall sentiment of tweets related to Coronavirus and how it is changing over time, and also identify any trending topics or concerns related to Coronavirus that may be of particular concern to the public. Data visualization tools will be used to present the results in an easily understandable format, making it useful for public health officials, researchers, and other stakeholders in the fight against the pandemic.</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/67974590/215528305-33cb1eec-8634-48cc-9b26-8294965ec371.gif">
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :floppy_disk: Table of Content</h2>

  * Problem Statement and Project Description
  * Project Files Description
  * Goal
  * Dataset Information
  * Exploratory Data Analysis
  * Random Forest Model
  * Logistic Regression Model
  * Support Vector Machine Model
  * Multinomial Naive Bayes Model
  * Technologies Used
  
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

 <h2> :floppy_disk: Project Files Description</h2>

<p>This project contains one executable file as follows:</p>
<h4>Executable Files:</h4>
<ul>
  <li><b>Coronavirus_Tweet_Sentiment_Analysis.ipynb</b> - Google Collab notebook containing data summary, exploration, visualisations, modeling, model performance, evaluation and conclusion.</li>
</ul>

<h4>Source Directory:</h4>
<ul>
  <li><b>Data & Resources link :</b> https://drive.google.com/drive/folders/1GSMf7FhgPhMZWqzFkB2pWKy5q9kGxxfF?usp=share_link</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: Goal:</h2>

Our goal here is to provide a comprehensive understanding of sentiment analysis and its applications in various domains such as business, politics, and social media. By analyzing the sentiment of texts, one can gain valuable insights into the public opinion and preferences, leading to better decision-making and increased customer satisfaction. This study will be a valuable resource for researchers, practitioners, and decision-makers who are interested in utilizing sentiment analysis to drive growth and success in their respective fields.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: Dataset information:</h2>

Features in the dataset:
Most of the fields are self-explanatory. The following are descriptions for those that aren't.
* Username - coded Username
* ScreenName - coded ScreenName
* Location - Region of origin
* TweetAt - Tweet Timing
* OriginalTweet - First tweet in the thread
* Sentiment-Target variable - Sentiment of the tweet
    
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :chart_with_upwards_trend: Exploratory Data Analysis</h2>
<p>Visualizing Word Count Distribution showins that to make tweets sentiment positive or negative the requirement of words is more and on the other hand neutral tweets have comparatively less no of words as the histogram plot is positively skewed.
Unique word counts can provide insights into the vocabulary and language used in tweets with different sentiments. For example, you may notice that tweets with a negative sentiment and positive sentiment use more unique words compared to tweets with a neutral sentiment.</p>
<p>Visualizing number of tweets from countries and states all in the location feature gives us an intimation that countries like USA, England, India and etc. are tweeting about coronivirus on a large number.<p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: Random Forest</h2>

<p>Random forest is a supervised learning algorithm. It creates a "forest" out of an ensemble of decision trees, which are commonly trained using the "bagging" method. The bagging method's basic premise is that combining different learning models improves the overall output.
Simply said, random forest combines many decision trees to produce a more accurate and stable prediction.

<p align="center">
  <img src="https://user-images.githubusercontent.com/67974590/214353597-e432ac1d-d4ec-4a93-846f-81dc2cf52f1e.png">
</p>

<p>Furthermore, the random forest classifier is efficient, can handle a large number of input variables, and provides correct predictions in most cases. It's a very strong tool that doesn't require any coding to implement.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

 <h2> :book: Logistic Regression</h2>

<p>Logistic Regression is a widely used algorithm for multi-class classification problems, which is a form of linear regression used for predicting the probability of a categorical dependent variable. It is trained with labeled data and learns the relationship between input features and the target variable. The class with the highest probability is chosen as the prediction.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: Support Vector Classification (SVC)</h2>

<p>Support Vector Machines (SVMs) are a type of supervised learning algorithm that can be used for both binary and multiclass classification.
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/67974590/215536274-23de06dc-35bf-4674-910c-b3496ef1537d.png">
</p>

The OneVsOneClassifier is a strategy for multi-class classification problems. It works by training a separate binary classifier for each pair of classes, and then combining the results of all the classifiers to make a final prediction.

<p>The OneVsRestClassifier creates one binary classifier per class, where the class is treated as positive and all other classes are treated as negative. During prediction, the class with the highest score is chosen as the predicted class.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: MultinomialNB Classifier</h2>

<p>The MultinomialNB classifier is a variation of the Naive Bayes algorithm for text classification. It is specifically designed for text data with discrete features, such as word counts in a document. The classifier uses a multinomial distribution for the features and assumes independence between the features. It is typically used for text classification tasks such as sentiment analysis and spam detection.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
 
<h2> :chart_with_upwards_trend: Results</h2>
<p>In this case, the Random Forest model has a Accuracy score of 88.26%, which is 1.24% higher than the OneVsRest SVM model's score of 87.17%. This suggests that the Random Forest model is able to make better predictions than any othermodels implemented.
<p>Removing Stopwords from the tweets was the most general and basic preprocessing of tweets but it resulted in a major improvement in accuracy.
By handling the class imbalance we saw aprrox. 7-10 % of major increase in accuracy. So the choice of upsampling the minority classes was good. :)
Then we did lemmatization on the cleaned and preprocessed tweets the WordNet Lemmatizer almost lemmatized or stemmed most of the words that gave us an improvement on accuracy of almost 1%. We can Further improve the accuracy by introducing more refined lemmatizing model such as Stanford core NLP model or any other highly refined model.



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: Technologies Used::</h2>

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://user-images.githubusercontent.com/32620288/139657460-40ef4562-76bd-43f5-bbca-47b6bd29863e.png" width=100>](https://numpy.org)    [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)  [<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org) [<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)   [<img target="_blank" src="https://user-images.githubusercontent.com/32620288/137518674-f36c5ad3-3d64-4c7a-a07c-53f247750394.png" width=170>](https://colab.research.google.com/)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CREDITS -->
<h2 id="credits"> :scroll: Credits</h2>

< Yash Patil > | Keen Learner | Machine Learning Enthusiast
<p> <i> Contact me for Data Science Project Collaborations</i></p>
