import lyricsgenius as lg 
import re 

api_key = "bQamoWDa9VfHDPGGamXTbpHdhE1-ld4m1Z9pBA4tfYHzXqbyBe0Rjn8Tk6Rag5zy"
genius = lg.Genius(api_key)
genius = lg.Genius(api_key, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True, retries=5, verbose=False)


def get_lyrics(artist_name, artist_song):
    # Search for artist and song
    try:
        song = genius.search_song(artist_song, artist_name)

        if song:
            return song.lyrics
    except:
        return 


def findCamel(sentence):
    l = ''
    occurances = []
    for i, c in enumerate(sentence):
        if l and (c.isupper() and l.islower()):
            occurances.append(i)
        l = c
    return occurances

def findJunk(sentence):
    l = ''
    occurances = []
    for i, c in enumerate(sentence):
        if l and (c.isnumeric() and l.isalpha()):
            occurances.append(i)
        l = c
    return occurances
 
   
def pre_process_lyrics(lyrics):
   lines = lyrics.split('\n')[1:]
   pattern = re.compile(r'\d+Embed$')
   new_lyrics = []
   
   for line in lines:
      toAdd = line
      
      # remove any weird numeric characters
      junk_index = findJunk(line)
      if junk_index:
         junk_index = junk_index[0]
         toAdd = toAdd[:junk_index] #+ ' ' + toAdd[junk_index:]
      
      # remove any appended words
      junk_index = findCamel(line)
      if junk_index:
         junk_index = junk_index[0]
         toAdd = toAdd[:junk_index] #+ ' ' + toAdd[junk_index:]
      
      new_lyrics.append(toAdd + '\n')
   new_lyrics = [x for x in new_lyrics if x]
   word = ' '.join(new_lyrics)
   return word 