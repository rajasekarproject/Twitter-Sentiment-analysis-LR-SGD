# Twitter-Sentiment-analysis-LR-SGD
Twitter Sentiment analysis using Logistic Regression, Stochastic Gradient Descent

There are many ways of doing sentiment analysis here i have used Logistic Regression, Stochastic Gradient Descent.

What we need for doing sentiment analysis?
 1.) Library - Which library are we going to use? There are many ways to do sentiment analysis like
       -'NLTK', 'TEXTBLOB', 'Logistic Regression', 'VaderSentiment', 'Naive Bayes algo', 'SGD(Stochastic Gradient                    Descent)' etc.., 

 2.) Data(Training) for Algorithms - Pre classifeid data based on positive words, negative words from a huge text.

     a.) Downlaod data from link - http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

     b.) Text preprocess - Training data

     c.) Split data - Training and Testing

 3.) Fit Model & Predict - Basically we are fitting the model with train data and predict using test data for accuracy.

 4.) Data(Real) Sentiment analysis - Extract anyone of the types direct data like 'string' typed by you,text file,              csv file or social media streaming (Twitter, Facebook, Reddit etc..,) or web scrapping (Web pages)

     a.) Extract data - Here data is extracted from 'Twitter'

     b.) Text preprocessing (We are not performing all the listed below but these are steps taken care using some                    functions)

        - Unstructured data to structured data
        - Removing special characters, symbols
        - Removing stop words
        - BOW (Bag of words) / Tokenization
        - Upper case to Lower case conversion
        - Stemming/ Lemmatization
        - NER ( Named Entity Recognition)
        - Covert 'Word to vector'

 5.) Perform Sentiment Analysis - using algorithms
