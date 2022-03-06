Kaggle-NBA-Career-Prediction
==============================

To predict if a rookie player will last at least 5 years in the league based on their stats. Model training for predicting the 5-Year career longevity for NBA Rookies is based on data where:

- y = 0 if career years played < 5
- y = 1 if career years played >= 5

<a href="https://www.kaggle.com/c/uts-advdsi-nba-career-prediction">
   <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsWMVqCPDgjcRSjdHYcf5uILGqUREL9QU_QQ&usqp=CAU" width = "90%">
</a>   

Installation process
------------

The main steps required for installing an executing this <b>Kaggle-NBA-Career-Prediction</b> as follows:

1. Setup the local Git repository
2. Download the training and test datasets
3. Build and/or install required dependencies:
   - Docker container file 
   - Docker image
   - SKLearn
   - XGBoost
   - Hyperopt
4. Jupyter notebook to train models and run predictions


Setup the local Git repository
------------

Create a new folder to store this repository, eg: ~/Projects/nba-career-predict:

<pre>
cd ~
mkdir Projects
cd Projects
mkdir nba_career_predict
cd nba_career_predict
</pre>

To download all the necessary files and folders (apart from the datasets) run command <code>git clone</code>.

<pre>git clone https://github.com/nuwanprabhath/kaggle-nba-career-prediction.git</pre>


Install Train/Test Data
------------

Within your local repository main folder, create sub-folder 'data', and two other sub-folders within 'data':
<pre>
    |
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The raw Train & Test raw datasets.
</pre>

Download **Train** and **Test** datasets from <a href="https://www.kaggle.com/c/uts-advdsi-nba-career-prediction/data">Kaggle</a> and store them in the local repository 'data\raw' folder.


Installing dependencies
------------

This solution is using a pre-built Docker image that ensures the required libraries and their versions are ready to go - SKLearn, XGBoost, Hyperopt.

Within the main repository folder - create a file called `Dockerfile` (no extension) either via:
1. IDE (<a href="https://code.visualstudio.com/">VS Code</a>, <a href="https://www.jetbrains.com/pycharm/">PyCharm</a>, etc...) or 
2. in SSH using <code>vi Dockerfile</code> and add the following content:

<pre>
FROM jupyter/scipy-notebook:0ce64578df46
RUN conda install xgboost
RUN conda install sklearn
RUN conda install hyperopt
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"
RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc
WORKDIR /home/jovyan/work`  
</pre>
   
Build the image from this Dockerfile, on the terminal command line enter:

<code>docker build -t xgboost-notebook:latest .</code>


Execute Notebooks
---------------

Run the built Docker image

<pre>
docker run  -dit --rm --name nba_caree_predict -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v ~/Projects/nba_career_predict:/home/jovyan/work xgboost-notebook:latest 
</pre>

Locate the URL in the Docker log, and paste it into a browser to launch Jupyter Lab

<pre>docker logs --tail 50 nba_caree_predict</pre>

Execute the notebooks in the following order:

1. Data prep
2. Logistic Regression
3. XGBoost

Kaggle Submission
---------------

To determine the overall performance of the models, submit the prediction CSV outputs to <a href="https://www.kaggle.com/c/uts-advdsi-nba-career-prediction/submit">Kaggle</a> submissions. Prediction CSV output are saved in:

<pre>
    ├── data
    │   ├── external       <- Prediction CSV outputs for Kaggle.
</pre>

Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump. 
    |                      <- Train & Test raw datasets.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
