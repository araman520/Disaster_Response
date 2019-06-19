#import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle


def load_data(database_filepath):
    """
    Load processed data from SQLite database

    Parameters:
    database_filepath (str): Filepath for the SQLite database

    Returns:
    X (pandas dataframe): Features
    Y (pandas dataframe): Targets
    Y.columns.values (numpy array): Column names for targets    
    """
    #create sqlite engine to read database
    engine = create_engine('sqlite:///' + database_filepath)
    
    #read table from database
    df = pd.read_sql_table("Messages", engine)
    
    #Get features and targets
    X = df["message"]
    Y = df.drop(columns = ["id", "message", "original", "genre"])
    
    #return features, targets and category names
    return(X, Y, Y.columns.values)


def tokenize(text):
    """
    Tokenizer function for CountVectorizer
    Normalize, lemmatize, and tokenize text

    Parameters:
    text (str): text to be tokenized

    Returns:
    clean_tokens (list): Clean tokens in the text    
    """
    
    #tokenize the text
    tokens = word_tokenize(text)
    
    #instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        #normalize and lemmatize tokens
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    #return normalized, lemmatized, and tokenizd text
    return clean_tokens


def build_model():
    """
    Build a model using Pipeline and GridSearchCV

    Parameters:
    none

    Returns:
    cv (model): GridSearchCV model    
    """
    
    #Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #parameters for GridSearchCV
    parameters = {'clf__estimator__max_depth':[None, 10],
                  'clf__estimator__n_estimators':[5, 10, 20],
                  'clf__estimator__min_samples_leaf': [2, 4, 6]}
    
    #Instantiate GridSearchCV model
    cv = GridSearchCV(pipeline, parameters)
    
    #return GridSearchCV model
    return(cv)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model on the test features and targets
    Print the metrics for each category

    Parameters:
    model (model): The trained model
    X_test (pandas dataframe): Test features
    Y_test (pandas dataframe): Test targets
    category_names (numpy array): Target category names

    Returns:
    none    
    """
    
    #Predict using the model
    Y_pred = model.predict(X_test)
    
    #Create dataframe from predictions
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns.values)
    
    #Print headers for metrics
    print("CATEGORY", 12*" ", "PRECISION ", " RECALL ", " F1-SCORE")
    
    for column in category_names:
        
        #metrics report for each category
        report = classification_report(Y_test[column], Y_pred[column])
        print(column, (23-len(column))* " ", report[report.index("avg / total")+18:-10])


def save_model(model, model_filepath):
    """
    Save the model to a pickle file

    Parameters:
    model (model): The trained model
    model_filepath (str): destination filepath for model pickle file

    Returns:
    none    
    """
    
    #save model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Get the ETL processed data,
    build, train, evaluate, and save Machine Learning model

    Parameters:
    none

    Returns:
    none    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        #get features, targets and category names from databse
        X, Y, category_names = load_data(database_filepath)
        
        #split data in to train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #build model
        print('Building model...')
        model = build_model()
        
        #train model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #evaluate model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #save model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()