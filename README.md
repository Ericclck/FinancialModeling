# Financial Modeling

This repository contains a comprehensive pipeline for financial machine learning. It implements Purged K-Fold Cross Validation, Combinatorial Purged K-Fold Cross Validation, and Feature Importance evaluation (MDI, MDA).

## Pipeline Overview

The pipeline consists of the following steps:

1. Signal extraction using Signaler
2. Labeling with Labeler
3. Preprocessing and feature engineering with FeatureEngineer
4. Modeling

The pipeline is designed for customization. Each step is a class that can be customized or replaced. It is compatible with any processes suggested by "Advances in Financial Machine Learning", specifically the sections before and including Combinatorial Purged K-Fold Cross Validation.

FeatureEngineer includes functionality for shuffling data before or after PCA for the purpose of feature importance evaluation. The sklearn cross validation object was not suitable for this pipeline's cross-validation process, so a custom one was built, along with a scorer for cross-validation.

## Cross Validation Process

1. The model is first fitted to the training data to extract starting point dependent signals and the first touch (barrier) time of each signal.
2. The first touch time is reindexed to the close index and fed into the cv object for a non-overlapping split. Signals are fed into the pipeline.
3. Classic cross-validation is then performed.

MDA is also modified in a similar manner. Hyperparameter tuning is now fully automated.

## Feature Engineering Process

1. classic engineering
2. primary model can be replaced with post_primary_model
3. additional feature engineering like detecting market regimes and entropy can be done here

## Improvement

1. Wrap sampling dates creation and update into wrapper
2. re-index labeler with sampling dates
3. Wrap primary model replacement into labeler

## Details of data science and strategy built with this pipeline

In my personal website : ericclck.com
