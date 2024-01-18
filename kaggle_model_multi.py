import pandas as pd
import ast
import numpy as np

from utils.data_prep import dataset_to_X_y, load_datachunks
from utils.data_visualization import visualize_data, plot_multilabel_confusion_matrix

from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D, GlobalMaxPooling1D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style

import json



json_file_path = 'config.json'
with open(json_file_path, 'r') as json_file:
    config_dict = json.load(json_file)
    
ALL_GENRES = config_dict['ALL_GENRES']
GENRE_TO_INDEX = config_dict['GENRE_TO_INDEX']
INDEX_TO_GENRE = {index: genre for genre, index in GENRE_TO_INDEX.items()}
WORD_EMBED_FILE = config_dict['WORD_EMBED_FILE']
DIMENSION_LENGTH = config_dict['WORD_EMBED_LEN'] # this should be the size of the word embedding
WORDS_PER_LINE = config_dict['INPUT_LENGTH'] # this is how many words will be in each line of the lyric

def get_word_emb_dict():
    embedding = dict()
    with open(WORD_EMBED_FILE, 'r', encoding="utf8") as f:
        for line in f.readlines():
            line = line.split(' ')       
            try:
                embedding[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue 
    return embedding
# get our word embeddings dictionary to convert words to embeddings
word_embeddings = get_word_emb_dict()

def get_lstm():
    model = Sequential([
    LSTM(256, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu', 
          kernel_regularizer=regularizers.l2(0.01), 
          activity_regularizer=regularizers.l1(0.01)),
    Dense(len(ALL_GENRES.keys()), activation='sigmoid') 
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
    
def train_model(model, train_data, val_data, epochs=10):
    
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # define optimizer and compile
    optimzer = Adam(learning_rate=.001)
    model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])

    # Create early stopping object
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    # train the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    return model, history

def eval_model(model, test_data):
    
    # get data we can train with 
    x_test, y_test = test_data

    # Evaluate model on test data
    score = model.evaluate(x_test, y_test, verbose=1)

    # Calculate precision, recall, F1 score
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int) # round all elements above .5 up to 1 
    
    acc = accuracy_score(y_test, y_pred)
    #print('Precision:', precision_score(y_test_argmax, y_pred, average='weighted'))
    #print('Recall:', recall_score(y_test_argmax, y_pred, average='weighted'))
    #print('F1 score:', f1_score(y_test_argmax, y_pred, average='weighted'))
    #print('Accuracy:', acc)
    
    plot_multilabel_confusion_matrix(y_test, y_pred, INDEX_TO_GENRE)
    
    return acc 

def undersample(df: pd.DataFrame):
    
    unique_genres = len(list(set([genre for genres in df['genres'] for genre in genres])))
    
    
    # Calculate the average number of entries per genre
    average_entries_per_genre = len(df) / unique_genres
    desired_samples = int(average_entries_per_genre)
    
    print("crashing here")
    # Group by 'genre' and select the specified number of entries for each group
    selected_entries = df.groupby('genres')
    print('made it')
    selected_entries = selected_entries.head(desired_samples)
    print('mae it 2')
    
    # Reset the index of the selected entries
    selected_entries = selected_entries.reset_index(drop=True)
    
    return selected_entries
    

if __name__ == "__main__":
    
    # load in data 
    df = load_datachunks(1)
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))
    
    # visualize it so we can see 
    visualize_data(df['genres'])
    
    # Split the DataFrame into training (70%), testing (15%), and validation (15%)
    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    
    # turn dataframes into inputs we can process 
    X_train, y_train = dataset_to_X_y(train_df, word_embeddings, multi_class=True)
    X_test, y_test = dataset_to_X_y(test_df, word_embeddings, multi_class=True)
    X_val, y_val = dataset_to_X_y(val_df, word_embeddings, multi_class=True)
    
    # take a look at our label breakdown for each dataset
    #visualize_data([INDEX_TO_GENRE[y] for y in np.argmax(y_test, axis=1)], False)
    #visualize_data([INDEX_TO_GENRE[y] for y in np.argmax(y_train, axis=1)], False)
    models = {
        #'lstm': get_lstm(),
        'cnn': get_cnn()
    }
    
    best_score = 0
    best_model = None 
    best_model_name = ''
     
    for model_name in models.keys():
        print("Training:", model_name)
        model = models[model_name]
        model, hist = train_model(model, (X_train, y_train), (X_val, y_val))
        score = eval_model(model, (X_test, y_test))
        
        if score > best_score:
            best_score = score
            best_model = model 
            best_model_name = model_name
    
    print("The best model is {} with a accuracy of {}".format(best_model_name, best_score)) 
    best_model.save('models/model_{}.keras'.format(best_model_name))