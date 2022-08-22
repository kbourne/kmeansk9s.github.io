# CANCER TRAJECTORY ANALYSIS FOR CANCER IN DOGS - CODE

Masters of Applied Data Science Capstone Project

`CANCER TRAJECTORY ANALYSIS FOR CANCER IN DOGS` focused on exploring the use of machine learning in predicting cancer and its trajectory among companion dogs.

<img src="../images/victor-grabarczyk-x5oPmHmY3kQ-unsplash.jpg" width="340" align="center"> <img src="../images/taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg" width="302" align="center"> <img src="../images/t-r-photography-TzjMd7i5WQI-unsplash.jpg" width="338" align="center"> 

Contents
========

 * [Why?](#why)
 * [What is in this repo?](#what-is-in-this-repo)
 * [What do I need to run this code?](#what-do-I-need-to-run-this-code)

### Why?

It is widely recognized in the veterinary world that dogs provide a unique model for health research that parallels the human environment. Dogs are exposed to similar social and environmental elements as humans, exhibiting increases in many chronic conditions with dynamics similar to human patterns. Dogs also have shorter life spans, which allows researchers to observe their entire life course in a much more condensed time frame. Use of machine learning in human healthcare has advanced rapidly in recent years, paving the way for new and deeper insights into how data can be used to improve human healthcare. Due to the similarities between human and dog healthcare, we seek to bring these analytical innovations to dog healthcare, with the hopes of finding deeper insights that can help both canine and human care. This analysis is focused on determining if the application of two cutting edge techniques that have emerged in human healthcare can be applied to dog healthcare with the same success, helping both fields advance. 

### What is in this repo?
---

In this repo are Jupyter Notebooks that run:

+ _EDA.ipynb_ - A quick, initial look at the data.
+ _DogCancerShallowMLPredictions.ipynb.ipynb_ - Analysis of traditional (shallow) machine learning models.
+ _NN-BinaryClass.ipynb.ipynb_ - Analysis of multi-layer perceptron neural network (deep) machine learning model.
+ _DogCancerTrajectoryPredictions.ipynb_ - Analysis of TimeGANs (for complex synthetic data generation) and Attentive State Space Model and the application of these advanced human healthcare based machine learning models to dog health data

There are also python files that are used by the Notebooks that contain most of the model-related

+ _assm.py_ - ASSM related functions
+ _constants.py_ - constants for this project
+ _data_proc.py_ - Data processing related functions
+ _metrics.py_ - evaluation related functions for the TimeGANs
+ _ml_models.py_ - shallow learning related functions
+ _timegan.py_ - TimeGANs related functions

The remaining files are a saved version and related files of a trained ASSM model.

### What do I need to run this code?
---

Each file is set up to run in various environments, like Google Colab, Google Workbook (on GCP), and other files.  They are all commented out by default in the first cell in each file.  You will need to determine your environment and uncomment the proper lines, while making sure the other lines are commented out. 

Once you have run the first cell, comment out the lines that have specific versions, and then "reset the kernel" for the new versions to take effect.

At this point, you should be able to run the cells in consecutive order, as they build up data and model definitions.  Be sure to run them in order!

Versions are listed in the requirements.txt file.
