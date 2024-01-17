import pandas as pd
import ast
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from lastfm_functions import get_tags
from genre_helper import tags_to_genre
from sklearn.model_selection import train_test_split
from copy import deepcopy
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D, GlobalMaxPooling1D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import matplotlib.pyplot as plt 
import random 

DIMENSION_LENGTH = 100 # this should be the size of the word embedding
WORDS_PER_LINE = 600 # this is how many words will be in each line of the lyric

ALL_GENRES = {
   "pop": ['pop'],
   "rock": ['rock'],
   "hip hop": ['hip hop', 'rap'], 
   "dance": ['dance', 'eccentric', 'party', 'festive'],
   "country": ['country'],
   "jazz": ['jazz'],
   "blues": ['blues'],
   "r&b": ['r&b', 'soul'],
   "metal": ['metal'],
   'gospel': ['gospel', 'church', 'spirtual'],
   'alternative': ['alternative', 'indie']
   }

INDEX_TO_GENRE = { i:v for i, v in enumerate(ALL_GENRES.keys()) }
GENRE_TO_INDEX = { v:i for i, v in enumerate(ALL_GENRES.keys()) }
DATASET_SIZE = 500 

WORD_EMBEDDINGS = dict()
def add_to_dict(d, filename):
    with open(filename, 'r', encoding="utf8") as f:
        for line in f.readlines():
            line = line.split(' ')
            
            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue 
add_to_dict(WORD_EMBEDDINGS, 'glove/glove.6B.100d.txt')


def read_csv(file_name, chunk_size):
    for chunk in pd.read_csv(file_name, chunksize=chunk_size):
        yield chunk

def add_to_dict(d, filename):
    with open(filename, 'r', encoding="utf8") as f:
        for line in f.readlines():
            line = line.split(' ')
            
            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue 

def message_to_token_list(s, words=WORD_EMBEDDINGS):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words] # we only care about the token if we have a word embedding for it 
    return useful_tokens

def message_to_word_vectors(message, word_dict=WORD_EMBEDDINGS):
    processed_tokens = message_to_token_list(message)
    
    word_embeddings = []
    
    # convert tokens to word embeddings and add to list
    for token in processed_tokens:
        if token in word_dict:
            word_embeddings.append(word_dict[token])
    return np.array(word_embeddings, dtype=float)

def one_hot_encode(labels, mapping=GENRE_TO_INDEX):
    encoding = np.zeros(shape=(len(mapping.keys())))
    # change this later, but I will only use 1 genre instead of multiple
    #for label in labels:
        #encoding[int(label)] = 1
        #break # choose only 1 genre
    encoding[int(labels[random.randint(0, len(labels)) % len(labels)])] = 1 
    return encoding

def map_genres_to_indexes(genre_list):
    # takes the genre list and converts it into numerical form of relevant genres
    return np.array([GENRE_TO_INDEX[genre] for genre in genre_list]).astype(float)

def format_X(X, desired_sequency_length=WORDS_PER_LINE):
    
    # ensure's all entries in X are the same size 
    
    X_copy = deepcopy(X)
    
    # loop through all the lyric lines 
    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        seq_len_diff = desired_sequency_length - x_seq_len
    
        if seq_len_diff < 0: 
            # over the sequence lenth, so take up to the desired sequence length
            X_copy[i] = x[:desired_sequency_length]
            continue
        
        # what we will pad to each lyric line if its not the right size
        pad = np.zeros(shape=(seq_len_diff, DIMENSION_LENGTH))
        
        # add the padding to the current lyric line
        X_copy[i] = np.concatenate([x, pad])
    
    # finally return padded input 
    return np.array(X_copy).astype(float)

def df_to_X_y(dff):
    
    # songs can be very long, having one vector for an entire song could result in 1000+ words, making 
    # training computationally expensive. I decided to do the embeddings line by line for each song 
    # so the vector would be smaller 
    
    y = []
    
    all_word_vector_sequences = []
    
    for genre, lyric in zip(dff['genres'], dff['lyrics']):
        
        # convert the lyric to vector form 
        lyrics_as_vector_seq = message_to_word_vectors(lyric)
        
        # if no useable tokens are returned 
        if lyrics_as_vector_seq.shape[0] == 0:
            lyrics_as_vector_seq = np.zeros(shape=(1, DIMENSION_LENGTH))
            
        # update tracking lists 
        all_word_vector_sequences.append(lyrics_as_vector_seq) # add to our X input lists
        y.append(genre) # mark the line as whatever the song's genre is 
    
    # make sure to convert labels to numerical representation
    y = pd.Series(y)
    y = y.map(map_genres_to_indexes)
    
    # make sure to convert y_train to correct input
    y = np.array([one_hot_encode(y) for y in y.tolist()])
    
    return format_X(all_word_vector_sequences), y

def visualize_data(labels):
    # this will help you detect outliers 
    genres = [INDEX_TO_GENRE[np.argmax(y)] for y in labels]
    
    # this will help you detect outliers 
    genres = pd.Series(genres)
    genres.value_counts().plot(kind='bar', edgecolor='black')
    plt.show()
    print(genres.describe())

def get_modified_dataset(data):
    df = pd.DataFrame(columns=['song', 'artist', 'genres', 'lyrics'])

    for index, row in data.iterrows():
        print("Song {} / {}".format((index + 1) % len(data) , len(data)))
        song = row['title']
        artist = row['artist']
        tags = [row['tag']]
        additional_tags = get_tags(artist, song)
        if additional_tags:
            tags.extend(additional_tags)
            
        genres = tags_to_genre(tags, ALL_GENRES)
        lyrics = row['lyrics']
        langugae = row['language'] # maybe use this to define a World Music genre
        
        df.loc[len(df.index)] = [song, artist, genres, lyrics]

    return df 
    
def get_lstm():
    model = Sequential([
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu', 
          kernel_regularizer=regularizers.l2(0.01), 
          activity_regularizer=regularizers.l1(0.01)),
    Dense(len(ALL_GENRES.keys()), activation='softmax')  # number of classes (sentiment categories)
    ])
    return model

def get_cnn():
    
    FILTERS = 250 # number of filters in your Convnet
    KERNEL_SIZE = 3 # a window size of 3 tokens (how many tokens to read at a time)

    
    cnn_model = Sequential()
    cnn_model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'valid' , activation = 'relu',strides = 1 , input_shape = (WORDS_PER_LINE, DIMENSION_LENGTH)))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(250))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(len(ALL_GENRES.keys())))
    cnn_model.add(Activation('sigmoid'))
    
    return cnn_model
    
def train_model(model, x, y, val_df, epochs=10):
    
    # get val data 
    x_val, y_val = df_to_X_y(val_df)
    
    # define optimizer and compile
    optimzer = Adam(learning_rate=.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

    # Create early stopping object
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    # train the model
    history = model.fit(x, y, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    return model, history

def eval_model(model, test_df):
    
    # get data we can train with 
    X_test, y_test = df_to_X_y(test_df)

    # Evaluate model on test data
    score = model.evaluate(X_test, y_test, verbose=1)

    # Calculate precision, recall, F1 score
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_test_argmax, y_pred)
    print('Precision:', precision_score(y_test_argmax, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test_argmax, y_pred, average='weighted'))
    print('F1 score:', f1_score(y_test_argmax, y_pred, average='weighted'))
    print('Accuracy:', acc)
    
    return acc 

def predict_genre(model, lyrics):
    
    # convert the lyric to vector form 
    lyrics_as_vector_seq = message_to_word_vectors(lyrics)
    
    # if no useable tokens are returned 
    if lyrics_as_vector_seq.shape[0] == 0:
        lyrics_as_vector_seq = np.zeros(shape=(1, DIMENSION_LENGTH))
    
    # add padding 
    size_diff = WORDS_PER_LINE - len(lyrics_as_vector_seq)
    lyrics_as_vector_seq = list(lyrics_as_vector_seq)
    pad = [ [0] * DIMENSION_LENGTH for _ in range(size_diff)]
    if pad:
        lyrics_as_vector_seq.extend(pad)
    
    if size_diff < 0:
        lyrics_as_vector_seq = lyrics_as_vector_seq[:WORDS_PER_LINE]
    
    X_input  = np.array([lyrics_as_vector_seq]).astype(float)
    
    # make prediction
    y_pred = model.predict(X_input)
    y_pred = np.argmax(y_pred, axis=1)
    
    # convert back to readiable text 
    return INDEX_TO_GENRE[y_pred] 
    

def make_dataset_chunks(chunk_size=DATASET_SIZE):
    ## helper to make smaller dataset chunks out of song_lyrics.csv (huge file)
    
    for i, df in enumerate(read_csv("song_lyrics.csv", chunk_size)):
        
        # get df name 
        folder_path = 'data\kaggle_modified'
        file_name = 'dataset_{}_{}.csv'.format(i, chunk_size)
        file_path = os.path.join(folder_path, file_name)
        
        # dont overwrite datafiles that already exist 
        if os.path.exists(file_path):
            continue
        
        # modify dataset by adding additional tag fields
        df = get_modified_dataset(df)
        df.to_csv(file_path, index=False) # save the dataset  
        
        
if __name__ == "__main__":
    
    
    df = pd.read_csv(file_path)
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))
        
    
    # Split the DataFrame into training (70%), testing (15%), and validation (15%)
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    
    # turn dataframe into inputs we can process 
    X_train, y_train = df_to_X_y(train_df)
    visualize_data(y_train)
    
    models = {
        'lstm': get_lstm(),
        'cnn': get_cnn()
    }
    
    best_score = 0
    best_model = None 
    best_model_name = ''
     
    for model_name in models.keys():
        print("Training:", model_name)
        model = models[model_name]
        model, hist = train_model(model, X_train, y_train, val_df)
        score = eval_model(model, test_df)
        
        if score > best_score:
            best_score = score
            best_model = model 
            best_model_name = model_name
    
    print("The best model is {} with a accuracy of {}".format(best_model_name, best_score)) 
    best_model.save('models/model_{}.keras'.format(best_model_name))