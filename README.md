# Disaster Response Pipeline Project

### Installation:
This project does not require any libraries outside Anaconda. Python 3 was used for this project.
    
### Project Motivation:
We classify diaster messages in to 36 different categories using machine learning.
    
### File Descriptions:		
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # CSV file with the disaster categories 
|- disaster_messages.csv  # CSV file with the disaster messages
|- process_data.py # Python file to run the ETL pipleine. Outputs the DisasterResponse.db file

- models
|- train_classifier.py # Python file to run the ML pipeline. Outputs the classifier.pkl file
|- classifier.pkl  # Pickle file of the trained model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

	- To run ETL pipeline that cleans data and stores in database
	  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
      		
	- To run ML pipeline that trains classifier and saves
          `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
  `python run.py`

3. Go to http://0.0.0.0:3001/
    
### Results:
The distribution of the messages in the training data can be seen on the homepage of the dashboard. Enter a message in the online dashboard, and the categories predicted will be highlighted.
    
### Licensing, Authors, Acknowledgements:
The data used for this project is in disaster_categories.csv and disaster_messages.csv. Anyone is allowed to use the code used in this project as they please.
