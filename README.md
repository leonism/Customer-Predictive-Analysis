![FIFA Logo](/images/fifa.png)

# Dataiku's Conundrum Challenge on FIFA Dataset.

# Introduction
If there's one thing similar about an interesting dataset and a good football's match, is that they're all keeping everyone's safe at home during this time of the pandemic. And in all honesty, I'm no data-scientist nor a dev guy. I just recently got myself exposed to a Machine Learning/Artificial Intelligent in general, while Dataiku in particular -- somewhere a little over then 3 months ago, but anyway here's my take to this conundrum's challenge.

# Installation
On this repository, you may find my personal projects related to Machine Learning, EDA, Python Jupyter Notebook and couple of Visualization based on the Dataiku Platform exported standard files. Most of the datasets I've been working with, downloaded from Conundrum site. Installation pretty straight forward. Simply download the whole set as a single project and as a ZIP file, everything have been flattened out with plain text files, and no SQL dump was involved, so there wouldn't be any missing system dependencies issue. Simply imported the downloaded Zip file to your working project.

# Jupyter Notebooks
- [Correlations analysis on Conundrum_13_Data_prepared (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/Correlations%20analysis%20on%20Conundrum_13_Data_prepared%20(admin).ipynb) 
- [Correlations analysis on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/Correlations%20analysis%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb) 
- [High dimensional data visualization (t-SNE) on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/High%20dimensional%20data%20visualization%20(t-SNE)%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb)
- [PCA on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/PCA%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb)
- [Statistics and tests on a single population on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/Statistics%20and%20tests%20on%20a%20single%20population%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb)
- [Statistics and tests on multiple populations on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/Statistics%20and%20tests%20on%20multiple%20populations%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb)
- [Topic modeling on Conundrum_13_Data_prepared_scored (admin).ipynb](https://github.com/leonism/dataiku-FIFA/blob/master/ipython_notebooks/Topic%20modeling%20on%20Conundrum_13_Data_prepared_scored%20(admin).ipynb)

# Data Flow
And since the challenge is not to 'predict' anything, rather to group/cluster the player's skillsets in reflect to their wages rate. Here's what my current flow would look like, and don't bother much on the 2 additional datasets, as they're merely exported from the existing model, so that I may explore them further.
![main-flow.png](/images/main-flow.png)

# Prepare Recipes
And here's how I go about on the prepare recipes, nothing out of the ordinary. Just converting *categorical* to *numerical* values with one-hot encoding and filling up the *'NaN'* with median values, while grouping them to have better clarity, if ever I need to go back and revise anything.
![recipes-prepared.jpg](/images/recipes-prepared.jpg)

# Modeling & Training 
While on modeling/training steps, I choose the '_Interactive Clustering_' which in return, delivered me a sufficient scoring value.
![model-score.png](/images/model-score.png)


# Clustering Classification
On to the clustering variables name, I simply identify them in the grading manner, starting from '_Grading A_', as the most top-knot performer, all the way down to the least performing one marked with '_Grading E_'.
![clusttering.jpg](/images/clusttering.jpg)


# Cluster Plot
And here's how my _cluster plot_ would look like, obviously the better the grade, the least volume of players getting included in them.

**Acceleration x Wage**
![scatter-plot-a.png](/images/scatter-plot-a.png)

**Sliding Tackle x Wage**
![scateter-plot-b.png](/images/scateter-plot-b.png)

# Variables Significant Level
And for sure those who sits at the 'Grading A' level would stand above the average threshold (though, that's not always the case with other variables).<br /><br />
![grading-a-variables.jpg](/images/grading-a-variables.jpg)


# Value Proposition
And coming back again to the initial question,  _"creating a flow that outputs a value proposition in term of their wages"_. I think I didn't include the players name and their nationalities in my modeling for a couple of reasons. In my opinions, those two variables are just too subjective to get included. In a sense that you could be a top-knot player, regardless of what your 'Names' would sound like, and of course your 'Nationalities'.

So I've done the DSS flow diagram, while the followings are my list of 'value proposition' that contributed of being one '_Grading-A_' player in the field.

**Top 5 Values Proposition**
![fig1.png](/images/fig1.png)

**Top 5 Values Proposition By Distribution.**
![fig3.png](/images/fig2.png)

**Top 5 Values Proposition By Grade.**
![fig3.png](/images/fig3.png)

# Correlation Matrix

The very first correlation analysis consists of plotting the "Correlation matrix" for numerical variables. For each couple of numerical variables, this computes the "strength" of the correlation (called the Pearson coefficient):

- 1.0 means a perfect correlation
- 0.0 means no correlation
- -1.0 means a perfect "inverse" correlation

Since it does not really make sense to print this correlation plot for hundred of variables, we are restricting it to the first 50 numerical variables of the dataset.
![download-1.png](/images/download-1.png)
![download.png](/images/download.png)


Been enjoying exploring this dataset for sure, and certainly it was fun doing it, stays safe everyone! 😊


# Disclaimer
And please remember, as this is only a weekend pet project, which I'm doing them for my personal interest only.