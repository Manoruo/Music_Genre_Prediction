# Music Genre Classification Project Overview

## Project Description

This project focuses on music genre classification using machine learning models. The goal is to predict music genres based on song lyrics. Two separate aspects are addressed: predicting a single genre per song and predicting multiple genres for a song. The project involves creating custom datasets from [LastFM](https://www.last.fm/home) and [Genius Lyrics](https://genius.com/), as well as processing a premadee [Kaggle dataset](https://www.kaggle.com/code/jvedarutvija/music-genre-classification) to build models for genre prediction.

<p align="center">
  <img src="images/last_fm.png" alt="LastFM" height="200"/> 
  <img src="images/Genuis_Lyrics.png" alt="Genius Lyrics" height="200"/>
</p>

## Key Files

1. **create_custom_dataset.ipynb:**
    - Creates a custom dataset by combining information from LastFM and Genius Lyrics.
    - Obtains song metadata, lyrics, and genres.
  
2. **create_modified_kaggle_dataset.ipynb:**
    - Generates chunk files from a larger Kaggle dataset.
    - Modifies the dataset using tags obtained from LastFM, enhancing genre information.

3. **genre_prediction_model_build.ipynb:**
    - Builds LSTM and CNN models capable of predicting a single genre for a given song.
    - Utilizes a configuration file (`config.json`) for settings such as word embeddings file, embedding dimensions, input length, and genre mappings.
  
4. **multi_class_genre_prediction_build.ipynb:**
    - Constructs LSTM and CNN models capable of predicting multiple genres for a given song.
    - Extends the single-genre prediction model to handle multiple genres.

## Key Points

- **Configuration File (`config.json`):**
    - Contains essential settings like word embeddings file, embedding dimensions, input length, and genre mappings.
    
- **Main Libraries Used:**
    - Pandas, NumPy, TensorFlow (Keras), Scikit-learn, Seaborn, Matplotlib, JSON.
    
- **Application Areas:**
    - Music genre classification can be applied in various domains such as music recommendation systems, playlist generation, and music streaming platforms.

## Execution Steps

1. **Custom Dataset Creation:**
    - Execute `create_custom_dataset.ipynb` to create a custom dataset from LastFM and Genius Lyrics.
  
2. **Kaggle Dataset Processing:**
    - Run `create_modified_kaggle_dataset.ipynb` to generate chunk files and enhance genre information using LastFM tags.
  
3. **Genre Prediction Models:**
    - Execute `genre_prediction_model_build.ipynb` to build LSTM and CNN models for predicting a single genre per song.
  
4. **Multi-class Genre Prediction Models:**
    - Run `multi_class_genre_prediction_build.ipynb` to extend the models for predicting multiple genres for a song.

## Future Work

- Experiment with different model architectures and hyperparameters for enhanced performance.
- Explore additional feature engineering techniques for improved genre prediction.
- Obtain a more diverse dataset (most songs are hiphop and r&b causing overfitting)

This project aims to explore the exciting intersection of music and machine learning, providing valuable insights into genre prediction from song lyrics.

## Helpful links

- The kaggle dataset can be found [here](https://www.kaggle.com/code/jvedarutvija/music-genre-classification). Download it and place it in the root of the repo. Ensure it is named "song_lyrics.csv"

- Last.fm API can be found [here](https://www.last.fm/api)

- Genuis API can be found [here](https://docs.genius.com/)

