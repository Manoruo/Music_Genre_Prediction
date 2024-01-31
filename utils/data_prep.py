import pandas as pd 
import glob 
import numpy as np 
import json 
import nltk
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
import random 

json_file_path = 'config.json'
with open(json_file_path, 'r') as json_file:
    config_dict = json.load(json_file)

GENRE_TO_INDEX = config_dict['GENRE_TO_INDEX']   
INPUT_LENGTH = config_dict['INPUT_LENGTH']
WORD_EMBED_LENGTH = config_dict['WORD_EMBED_LEN']

def load_datachunks(chunks=None):
    """
    
    Loads in dataset files into one big dataframe object 
    
    Args:
        chunks (int, optional): The number of dataset csv's to load in (None to load all)
        
    Returns:
        DataFrame: combined dataframe containing specified amount of chunk files
    """
    # Specify the path to your CSV files
    csv_files = glob.glob('data/kaggle_modified/*.csv')

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()
    
    # Iterate over each CSV file and concatenate to the combined DataFrame
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        # load in a specific amount of chunk files
        if chunks and (i + 1) == chunks:
            break 
    
    if len(combined_df) > 0:
        print("Loaded dataframe chunks Successfully")
    else:
        print("Did not load dataframe chunks")
    return combined_df

def message_to_token_list(s, words):
    """
    
    takes given sentence and tokenizes it into words that are in our 
    word embedding dictionary

    Args:
        s (string): the string to convert to tokenize_
        words (dictionary): the word embedding dictionary 

    Returns:
        list: the tokenized version of the input sentence (for words in our word embedder dictionary)
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words] # we only care about the token if we have a word embedding for it 
    return useful_tokens

def message_to_word_vectors(message, word_dict):
    """
    
    Takes in a string and returns the embedded representation 
    
    Args:
        message (string): the string we want word vectors for 
        word_dict (dict): the word embedding dictionary 

    Returns:
        _type_: _description_
    """
    
    processed_tokens = message_to_token_list(message, word_dict)
    
    word_embeddings = []
    
    # convert tokens to word embeddings and add to list
    for token in processed_tokens:
        if token in word_dict:
            word_embeddings.append(word_dict[token])
    return np.array(word_embeddings, dtype=float)

def one_hot_encode(label, mapping=GENRE_TO_INDEX):
    """
    
    Takes in label for a given song dataset and returns the one hot encoded version.
    
    Ex input:
    label = ['hip hop', 'r&b', blues]
    output (not exact): [0, 0, 1, 0, 1, 1] # each 1 represents the correspodning genre
    
    Args:
        label (list): genres for the given song 
        mapping (dictionary, optional): the word embedding dictionary
    Returns:
        _type_: one hot representation of the genre for given label 
    """
    
    encoding = np.zeros(shape=(len(mapping.keys())))
   
    # set the corresponding positions to 1 (meaning song has that genre)
    for genre in label:
        if genre in GENRE_TO_INDEX.keys():
            encoding[GENRE_TO_INDEX[genre]] = 1    
                
    
    return encoding

def pad_X(X, desired_sequency_length=INPUT_LENGTH):
    """
    
    Takes in list of encoded lyrics and pads or shrinks it so they are all the same size 

    Args:
        X (list): list of encoded lyrics for multiple songs 
        desired_sequency_length (int, optional): The size each song should be 

    Returns:
        list: reformatted x 
    """
    X_copy = deepcopy(X)
    
    # loop through all the lyrics 
    for i, x in enumerate(X):
        
        # see how far off current lyric length is 
        x_seq_len = x.shape[0]
        seq_len_diff = desired_sequency_length - x_seq_len
        
        # shrink if needed 
        if seq_len_diff < 0: 
            # over the sequence lenth, so take up to the desired sequence length
            X_copy[i] = x[:desired_sequency_length]
            continue
        
        # what we will pad to each lyric line if its not the right size
        pad = np.zeros(shape=(seq_len_diff, WORD_EMBED_LENGTH))
        
        # add the padding to the current lyric line
        X_copy[i] = np.concatenate([x, pad])
     
    return np.array(X_copy).astype(float)

def dataset_to_X_y(dataset, word_embed_dict):
    """
    
    Takes in a dataframe with columns:
    
    "genre" - list of strings
    "lyrics" - string 
    
    and converts them into inputs for model

    Args:
        dataset (DataFrame): The dataframe housing all the data we want to process  
        word_embed_dict (dict): the word embedding dictionary

    Returns:
        array: the x input and y inputs to our model
    """
    
    y = []
    all_word_vector_sequences = []
    
    # go through all instances in dataset 
    for genre, lyric in zip(dataset['genres'], dataset['lyrics']):
        
        # convert the lyric to vector form 
        lyrics_as_vector_seq = message_to_word_vectors(lyric, word_embed_dict)
        
        # if no useable tokens are returned 
        if lyrics_as_vector_seq.shape[0] == 0:
            lyrics_as_vector_seq = np.zeros(shape=(1, WORD_EMBED_LENGTH))
            
        # update tracking lists 
        all_word_vector_sequences.append(lyrics_as_vector_seq) # add to our X input lists
        y.append(genre) # mark the line as whatever the song's genre is 
    
    y = pd.Series(y)
    
    # make sure to convert y_train (genres) to one hot numerical represnetation 
    y = np.array([one_hot_encode(y) for y in y.tolist()])
    # ensure all lyrics are same size
    x = pad_X(all_word_vector_sequences)
    return x, y

def word_vectors_to_message(message, word_dict, num_to_convert=10):
    
    words = []
    count = 0
    for emb_token in message:
        for word, emb in word_dict.items():
            if np.array_equal(emb_token, emb):
                words.append(word)
                count = count + 1 
        if count == num_to_convert:
            break 
    
    lyric = ' '.join(words)
    return lyric
            
