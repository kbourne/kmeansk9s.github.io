# Wealth and Sourcing on Dog Health

## Project description

This analysis leveraged the Owner and Environment data sets from the [Dog Aging Project](dogagingproject.org) to analyze potential relationships betweet:
* Owner income and neighborhood wealth on the presence of health conditions in the data set.
* Source of acquisition (e.g., breeder, rescue shelter, pet shop) on the prevalence of health conditions in the data set.

Contents
========

 * [What is in this repository?](#Repository)
 * [How can I access this data?](#Data_access)
 * [Which files from the Dog Aging Project are needed for this analysis?](#Files)
 * [What libraries are needed to run this code?](#Libraries)

### Repository
This repository contains documentation, code, and instructions to run the analysis on data from the Dog Aging Project specific to exploring the potential relationship between owner wealth and dog health and the relationship between source of acquisition and dog health. The following files are present in this portion of the repository:
* IPYNB file - this contains the code used to run the analysis.
* this README file with instructions and tips for running the provided code.

### Data_access
* The data for this analysis was retrieved from the [Dog Aging Project](https://dogagingproject.org/).
    * Note that you must request access to the Dog Aging Project in order to gain access to the data collected through the project.
    * Access to the Dog Aging Project files requires signing a legal agreement.

### Files
* The analysis of Owner Income and Acquisition Source requires only 2 of the files available from the Dog Aging Project. These are the "Owner" and "Environment" data sets.
* In order to perform the analysis using the median income of the neighborhood where each dog lives, you must join the Owner and Environment data sets. These data sets can be joined using the "dog_id" column from each data set.
    * After joining the Owner and Environment data sets, you must make sure to drop duplicates. Otherwise, this will result in repeated records as multiple records in the Environment data set can use the same dog_id. 

### Libraries
The following libraries are required to run this code. These can be imported directly into a Python notebook:
* Pandas (1.3.5 used at the time this report was created)
* Matplotlib
* Seaborn
* Numpy
* Sklearn
    * train_test_split
    * Random Forest Classifier
    * Classification Report
    * Confusion Matrix
    * Linear Regression
    * Dummy Classifier
    * Plot_confusion_matrix
* Scipy
    * chi2_contingency
