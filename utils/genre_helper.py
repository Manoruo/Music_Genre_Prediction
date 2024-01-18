import spacy 

# Load the spaCy model
nlp = spacy.load("en_core_web_md")


def get_closest_genres(tag, genres, threshold=0.6):
    # Calculate similarity between the tag and each genre discription 
    similarities = {}
    
    for genre, discs in genres.items():
        # get the best similarity between the tag and current list of discriptors 
        
        scores = []
        # go through all discriptors for the genre and see if any of them are similar to the current tag 
        for disc in discs:
            m1 = nlp(tag.lower())
            m2 = nlp(disc.lower())
            if m1.has_vector and m2.has_vector:
                score = m1.similarity(m2)
            else:
                score = 0 
                
            scores.append(score)
        best_score = max(scores)
        #best_score = max([nlp(tag.lower()).similarity(nlp(disc.lower())) for disc in discs if nlp(tag.lower()).has_vector and nlp(disc.lower()).has_vector])
        similarities[genre] = best_score # the best similarity score represents how well the tag matches the current genre

    # Filter genres with similarity above the threshold
    similar_genres = [{'genre': genre, 'sim': similarity} for genre, similarity in similarities.items() if similarity > threshold]
    sorted(similar_genres, key=lambda x: x['sim'], reverse=True) # sort genre's according to how well they match the tag 
    return similar_genres

def tags_to_genre(tags, genres, threshold=.6):
 
    genre_scores = {}   
    for tag in tags:
        scores = get_closest_genres(tag, genres)
        
        for score in scores:
            genre = score['genre']
            sim = score['sim']
            
            if genre not in genre_scores:
                genre_scores[genre] = {'score': 0, 'count': 0}
            
            genre_scores[genre]['score'] += sim 
            genre_scores[genre]['count'] += 1 
                
    # calc score per genre 
    avg_scores = []
    for genre in genre_scores.keys():
        score = genre_scores[genre]['score'] / genre_scores[genre]['count']
        avg_scores.append({'genre': genre, 'score': score})
    
    
    top_genres = sorted(avg_scores, key= lambda x: x['score'], reverse=True)
    top_genres = [x['genre'] for i, x in enumerate(top_genres) if i == 0 or x['score'] > threshold]
    
    return top_genres
    