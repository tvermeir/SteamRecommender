# Recommending Similar Games Based on User Playtime

## Summary

This project investigates whether we can predict which unplayed games a user might enjoy by analyzing their playtime-weighted similarity to other games they already own and play.
Rather than relying on binary purchase data, this system leverages actual engagement patterns, using playtime as a proxy for user preference to generate personalized game recommendations.

The core algorithm is Item-Item Collaborative Filtering, implemented at scale with Dask to process a large user-game matrix. The result is a recommender system that aligns more closely with what players actually play, not just what they buy 

## Dataset 
Source: [Steam Dataset – UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data).
This project uses:

australian_users_items.json: playtime histories from Steam users

cleaned_steam_games.json: metadata for over 15,000 games (genres, tags, price, etc.)

**This dataset contains:**
  * 2.5 million users

  * 7.7 million user reviews

  * Detailed playtime engagement data

## Process
### 1. Preprocessing
Parsed raw user data into user–game–playtime triples

Transformed playtime data using log1p(playtime) to reduce skew from outliers

### 2. Matrix Construction
Built a NumPy matrix (Users x Games) with log-transformed playtime values

Converted to Dask array for scalable, distributed computation

### 3. Similarity Calculation
Computed cosine similarity between all game vectors (item-item collaborative filtering)

Produced a full item-item similarity matrix

### 4. Recommendation Generation
For each user:

* Identify most-played games

* Recommend unplayed games similar to those, weighted by playtime

Used Dask Delayed to parallelize this across user chunks

## Results

Successfully created a personalized game recommender using playtime data

Observed weak scalability due to overhead in parallelization





