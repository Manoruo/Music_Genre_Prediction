import json
import pandas as pd 
import os 
from utils.lastfm_functions import get_tags
from utils.genre_helper import tags_to_genre

json_file_path = 'config.json'
with open(json_file_path, 'r') as json_file:
    config_dict = json.load(json_file)

DATASET_SIZE = 500 # How many songs should be in each dataframe chunk ( you don't need to change this really)
OVERWRITE = False # weather or not to overwrite dataset chunk files (you don't need to change this really)
ALL_GENRES = config_dict['ALL_GENRES']

def read_csv(file_name):
    """_summary_

    Args:
        file_name (string): path to the kaggle dataset 

    Yields:
        yield: gives a yield object of the current dataset
    """
    for chunk in pd.read_csv(file_name, chunksize=DATASET_SIZE):
        yield chunk

def modified_dataset(data):
    """
    
    Modifies the kaggle dataframe given to add additional 
    genres by utilizing tags found using lastfm
    
    Args:
        data (DataFrame): the dataframe from kaggle 

    Returns:
        DataFrame: the modified dataframe with additional genres  
    """
    
    df = pd.DataFrame(columns=['song', 'artist', 'genres', 'lyrics'])

    # loop through each song in the dataframe 
    for index, row in data.iterrows():
        print("Song {} / {}".format((index + 1) % len(data) , len(data)))
        song = row['title']
        artist = row['artist']
        tags = [row['tag']] # keep the initial song genre obtain from kaggle dataset  
        lyrics = row['lyrics']
        langugae = row['language'] # maybe use this to define a World Music genre
        
        # only use english songs for now 
        if langugae != 'en':
            continue
        
        # find additional tags using lastfm 
        additional_tags = get_tags(artist, song)
        if additional_tags:
            tags.extend(additional_tags)
        
        # use these tags to determine the genre 
        genres = tags_to_genre(tags, ALL_GENRES)
        
        
        
        df.loc[len(df.index)] = [song, artist, genres, lyrics]

    return df 
    
def make_dataset_chunks(kaggle_dataset_path='song_lyrics.csv', save_dest='data/kaggle_modified'):
    """
    
        The kaggle dataset is extremely large so we process it in chunks. 
        This function splits the main kaggle dataset into chunks and stores it 
        in a dataframe formmated as follows: 
        
        dataset_{chunk_number}_{chunk_size}.csv 
    
    Args:
        kaggle_dataset_path (str, optional): the path to the kaggle dataframe
        save_dest (str, optional): the path where we want to save the dataframe chunks 
    """
    
    # read_csv() breaks the kaggle_dataset into chunks and lets us itterate through each chunk
    for i, df in enumerate(read_csv(kaggle_dataset_path)):
        print("On Chunk: {}".format(i))
        # get df name 
        folder_path = save_dest
        file_name = 'dataset_{}.csv'.format(i)
        file_path = os.path.join(folder_path, file_name)
        
        # dont overwrite datafiles that already exist
        if os.path.exists(file_path) and not OVERWRITE:
            continue
        
        # modify dataset by adding additional tag fields
        df = modified_dataset(df)
        df.to_csv(file_path, index=False) # save the dataset  


if __name__ == '__main__':
    make_dataset_chunks()
    