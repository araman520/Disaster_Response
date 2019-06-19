# Disaster Response Pipeline Project

### Installation:
	This project does not require any libraries outside Anaconda. Python 3 was used for this project.
    
### Project Motivation:
	We classify diaster messaged in to 36 different categories using machine learning.
    
### File Descriptions:
	-app
    	-run.py - Python file to run the online dashboard
        -templates
        	-go.html - HTML files for the elements in the online dashboard
            -master.html - HTML file for the online dashboard
	-data
    	-DisasterResponse.db - sqlite database with the ETL output
        -disaster_categories.csv - CSV file with the disaster categories
        -disaster_messages.csv - CSV file with the disaster messages
        -process_data.py - Python file to run the ETL pipleine. Outputs the DisasterResponse.db file
        
    -models
    	-classifier.pkl - Pickle file of the trained model
        -train_classifier.py - Python file to run the ML pipeline. Outputs the classifier.pkl file

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
	Enter a message in the online dashboard, and the categories predicted will be highlighted
    
### Licensing, Authors, Acknowledgements:
	The data used for this project is in disaster_categories.csv and disaster_messages.csv.
    Anyone is allowed to use the code used in this project as they please.
