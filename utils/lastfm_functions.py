import requests
import re 
import spacy

API_KEY = 'ffe4c2ced75a92c53ec0c015befed121'

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

def remove_non_alphanumeric(input_string):
    # Use regex to replace non-alphanumeric characters with an empty string
    return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

def titlesMatch(larger_string, substring):
    # check if titles match (get's rid of any special formatting)
    
    a = remove_non_alphanumeric(larger_string.lower())
    b = remove_non_alphanumeric(substring.lower())
    
    if a.find(b) != -1 or b.find(a) != -1:
        return True
    else:
        return False

def titlesMatch2(string1, string2, thresh=.6):
    # checks if two song titles are the same (even if they have minor differences)
    
    # remove any special characters
    string1 = remove_non_alphanumeric(string1.lower())
    string2 = remove_non_alphanumeric(string2.lower())
    
    # Process the sentences using spaCy
    doc1 = nlp(string1.lower())
    doc2 = nlp(string2.lower())
    similarity_score = doc1.similarity(doc2)
    return similarity_score > thresh
  
def get_track_tags_helper(api_key, artist_name, track_name):
    """
    Helper to get the tags for a track using Last.fm's API

    Parameters:
    - api_key (str): Your Last.fm API key.
    - artist_name (str): The name of the artist.
    - track_name (str): The name of the track.

    Returns:
    - List of tags or none if none found
    """
    # Last.fm API endpoint for track.getInfo
    api_endpoint = 'http://ws.audioscrobbler.com/2.0/'
    params = {
        'method': 'track.getInfo',
        'api_key': api_key,
        'artist': artist_name,
        'track': track_name,
        'format': 'json'
    }

    # Make the API request
    response = requests.get(api_endpoint, params=params)
    data = response.json()

    # Check if the request was successful
    if 'error' in data:
        print(f"Error: {data['message']}")
    else:
        # Extract tag information
        tags = data['track']['toptags']['tag']

        # Print the tags
        if tags:
            tags = [x['name'] for x in tags]
            return tags
        else:
            print(f"No tag information available for {artist_name} - {track_name}.")
            return None 

def search_song(artist_name, song_name, api_key=API_KEY):
    """
    Search for a song by a specific artist using Last.fm's API and return information about the first result.

    Parameters:
    - api_key (str): Your Last.fm API key.
    - artist_name (str): The name of the artist.
    - song_name (str): The name of the song to search for.

    Returns:
    - Dictionary containing information about the first result, or None if no result is found.
    """
    # Last.fm API endpoint for track.search
    api_endpoint = 'http://ws.audioscrobbler.com/2.0/'
    params = {
        'method': 'track.search',
        'api_key': api_key,
        'artist': artist_name,
        'track': song_name,
        'format': 'json'
    }

    # Make the API request
    response = requests.get(api_endpoint, params=params)
    data = response.json()

    # Check if the request was successful
    if 'error' in data:
        print(f"Error: {data['message']}")
        return None
    else:
        # Extract information about the first result
        track_matches = data['results']['trackmatches']['track']
        
        # Check if there are any results
        if not track_matches:
            print(f"No results found for {song_name} by {artist_name}.")
            return None

        # Extract information about the first result
        result = track_matches[0]

        # Return relevant information
        return {
            'song': result.get('name', ''),
            'artist': result.get('artist', ''),
            'url': result.get('url', '')
        }


# main functions # 

def get_tags(artist_name, song_name, api_key=API_KEY):
    """    
    Gets the tags for a track using Last.fm's API 

    Parameters:
    - api_key (str): Your Last.fm API key.
    - artist_name (str): The name of the artist.
    - track_name (str): The name of the track.

    Returns:
    - List of tags or None if no tags found 
    """
    # serach will get the correct name for the song given the artist + some form of the title (doesn't have to be 100% correct) 
    result = search_song(artist_name, song_name)
    
    if result:
        # get the exact spelling of artist + song name as this is needed to find the tag
        artist = result['artist']
        song = result['song']
        
        # make sure it found the right song 
        if not titlesMatch2(song, song_name):
            print("Can not Verify song {} and target song: {} are the same".format(song, song_name))
            #answer = input("Confirm song: {} is the same as target song: {}".format(song, song_name)).strip().lower()
            #if answer != 'y':
                #return
            return
            
        
        # query for the tag of the given song (it should exist since we found the song)
        tag = get_track_tags_helper(API_KEY, artist, song)
        #assert tag != None, "Error, found song {} by {} but could not query for tag".format(song, artist)
        
        return tag
    
def get_top_tracks(total_tracks_to_fetch=50, api_key=API_KEY):
    
    api_endpoint = 'http://ws.audioscrobbler.com/2.0/'
    method = 'chart.getTopTracks'
    limit_per_page = 50
    
    pages_to_fetch = (total_tracks_to_fetch + limit_per_page - 1) // limit_per_page

    all_top_tracks = []

    for page in range(1, pages_to_fetch + 1):
        params = {
            'method': method,
            'api_key': api_key,
            'page': page,
            'format': 'json'
        }

        response = requests.get(api_endpoint, params=params)
        data = response.json()

        if 'tracks' in data:
            top_tracks = data['tracks']['track']
            all_top_tracks.extend(top_tracks)

    return all_top_tracks[:total_tracks_to_fetch]
