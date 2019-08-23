# fivestarreviews
A sentiment analysis data science/nlp project to predict a 1 - 5 star review from the review text.

This project uses the review text and a 1 to 5 star review of products to build a prediction model based on gensim's Word2Vec.

The data used is Amazon reviews of books, courtesy of Julian McAuley's Amazon Product Data, available at http://jmcauley.ucsd.edu/data/amazon/

There are three parts.

### Load and Shuffle
The majority of Data Science is cleaning up the data, and that happens here. I used 8 million Amazon book reviews. The data had all the reviews for one product together, so I shuffled the reviews. To balance out the data, I took the category with the minimum number of reviews, one star, and limited my data to that many number of reviews in each category. That left 1.6 million reviews, which appears to be plenty to get good predictions. The reviews are shuffled, categorized, truncated in each category, and just the relevant columns (amazon product ID, review text, and number of stars) are saved to a csv file.

### Wordnet Model Building
I started with code from Manning's "Real World Machine Learning" and adapted it to do 1 to 5 star reviews instead of binary sentiment. The most accurate modelling used gensim Wordnet so I went with that. Where I could I used Python Multiprocessing to parallelize code over my processors. The Wordnet model and the RandomForestClassifier are saved to disk to serve up from an API. Testing is done predictions are graphed.

### serve-api
A Flask app that serves a simple API for predictions. Review text is Posted, you get back the review prediction and the array of probabilities for each star, 1 to 5.
The text is read in and treated as a "test run" of one sample. I used Postman to test.