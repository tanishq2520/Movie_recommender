import string
import pickle
import pandas as pd
import ast
import requests
import nltk
import aiohttp
import asyncio
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def get_genres(obj):
    try:
        lista = ast.literal_eval(obj)
        l1 = []
        for i in lista:
            l1.append(i['name'])
        return l1
    except:
        return []


def get_cast(obj):
    try:
        a = ast.literal_eval(obj)
        l_ = []
        len_ = len(a)
        for i in range(0, 10):
            if i < len_:
                l_.append(a[i]['name'])
        return l_
    except:
        return []


def get_crew(obj):
    try:
        l1 = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                l1.append(i['name'])
                break
        return l1
    except:
        return []


def read_csv_to_df():
    # Reading the CSV files
    credit_ = pd.read_csv(r'Files/tmdb_5000_credits.csv')
    movies = pd.read_csv(r'Files/tmdb_5000_movies.csv')

    # Merging
    movies = movies.merge(credit_, on='title')

    # Create movies2 for the Details page (contains full info)
    movies2 = movies.copy()
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    # Create movies dataframe for Analysis (Math)
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies', 'release_date']]
    
    # Critical: Drop NA and reset index to align matrices
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)

    # Extracting items from JSON strings
    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    # 1. Split Overview
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # 2. Remove Spaces (CRITICAL for Director/Cast logic)
    # "John Lasseter" -> "JohnLasseter"
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    # 3. Create the 'tags' column (Bag of Words)
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    # 4. Create the new_df for vectorization
    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']].copy()

    # 5. Join lists back to strings for CountVectorizer
    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))
    new_df['tcrew'] = new_df['tcrew'].apply(lambda x: " ".join(x))

    # 6. Lowercase everything
    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())
    new_df['tcrew'] = new_df['tcrew'].apply(lambda x: x.lower())

    # 7. Apply Stemming to tags and keywords
    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def stemming_stopwords(li):
    ans = []
    for i in li:
        ans.append(ps.stem(i))

    # Remove Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in ans:
        w = w.lower()
        if w not in stop_words:
            filtered_sentence.append(w)

    str_ = ''
    for i in filtered_sentence:
        if len(i) > 2:
            str_ = str_ + i + ' '

    # Remove Punctuation
    punc = string.punctuation
    str_ = str_.translate(str_.maketrans('', '', punc))
    return str_


# --- ASYNC FUNCTIONS FOR SPEED ---

async def fetch_single_poster_async(session, movie_id):
    try:
        url = 'https://api.themoviedb.org/3/movie/{}?api_key=6177b4297dff132d300422e0343471fb'.format(movie_id)
        async with session.get(url) as response:
            data = await response.json()
            if 'poster_path' in data and data['poster_path']:
                # Using w342 for grid (faster load)
                return "https://image.tmdb.org/t/p/w342/" + data['poster_path']
            else:
                return "https://via.placeholder.com/342x513?text=No+Image"
    except:
        return "https://via.placeholder.com/342x513?text=No+Image"

async def fetch_posters_async(movie_ids):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for movie_id in movie_ids:
            tasks.append(fetch_single_poster_async(session, movie_id))
        return await asyncio.gather(*tasks)

async def fetch_person_details_async(session, person_id):
    try:
        url = 'https://api.themoviedb.org/3/person/{}?api_key=6177b4297dff132d300422e0343471fb'.format(person_id)
        async with session.get(url) as response:
            data = await response.json()
            
            if 'profile_path' in data and data['profile_path']:
                # w185 for cast circles (very fast)
                img_url = 'https://image.tmdb.org/t/p/w185/' + data['profile_path']
            else:
                img_url = "https://via.placeholder.com/185x278?text=No+Image"
                
            biography = data.get('biography', "No biography available.")
            return img_url, biography
    except:
        return "https://via.placeholder.com/185x278?text=Error", "Could not fetch details."

async def fetch_cast_details_async(cast_ids):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pid in cast_ids:
            tasks.append(fetch_person_details_async(session, pid))
        return await asyncio.gather(*tasks)


# --- SYNCHRONOUS HELPER FUNCTIONS ---

def fetch_posters(movie_id):
    # This is used for the Background Image (Higher Quality)
    try:
        url = 'https://api.themoviedb.org/3/movie/{}?api_key=6177b4297dff132d300422e0343471fb'.format(movie_id)
        response = requests.get(url, timeout=3)
        data = response.json()
        # w780 for background
        str_ = "https://image.tmdb.org/t/p/w780/" + data['poster_path']
    except:
        str_ = "https://via.placeholder.com/500x750?text=No+Image"
    return str_

def fetch_person_details(id_):
    # Fallback synchronous method
    try:
        url = 'https://api.themoviedb.org/3/person/{}?api_key=6177b4297dff132d300422e0343471fb'.format(id_)
        data = requests.get(url, timeout=3).json()

        if 'profile_path' in data and data['profile_path']:
            url = 'https://image.tmdb.org/t/p/w185/' + data['profile_path']
        else:
            url = "https://via.placeholder.com/185x278?text=No+Image"

        biography = data.get('biography', "No biography available.")
    except:
        url = "https://via.placeholder.com/185x278?text=Error"
        biography = "Could not fetch details."

    return url, biography


# --- MAIN RECOMMENDATION LOGIC ---

def vectorise(new_df, col_name):
    # This is the logic used to build the pickle files
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec_tags = cv.fit_transform(new_df[col_name]).toarray()
    sim_bt = cosine_similarity(vec_tags)
    return sim_bt

def recommend(new_df, movie, similarity_matrix):
    try:
        idx_series = new_df[new_df['title'] == movie].index
        
        if idx_series.empty:
            return [], []
            
        movie_idx = idx_series[0]

        distances = similarity_matrix[movie_idx]
        
        # Get top 5
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        rec_movie_names = []
        rec_movie_ids = []

        for i in movie_list:
            rec_movie_names.append(new_df.iloc[i[0]]['title'])
            rec_movie_ids.append(new_df.iloc[i[0]]['movie_id'])

        return rec_movie_names, rec_movie_ids

    except IndexError:
        return [], []
    except Exception as e:
        print(f"Error: {e}")
        return [], []

def get_details(selected_movie_name):
    # Loading Dictionaries specifically for details
    pickle_file_path = r'Files/movies_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    movies = pd.DataFrame.from_dict(loaded_dict)

    pickle_file_path = r'Files/movies2_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict_2 = pickle.load(pickle_file)
    movies2 = pd.DataFrame.from_dict(loaded_dict_2)

    a = movies2[movies2['title'] == selected_movie_name]
    b = movies[movies['title'] == selected_movie_name]

    if a.empty or b.empty:
        return None

    def get_val(df, col):
        if not df.empty:
            return df.iloc[0][col]
        return None

    budget = get_val(a, 'budget')
    overview = get_val(a, 'overview')
    release_date = get_val(a, 'release_date')
    revenue = get_val(a, 'revenue')
    runtime = get_val(a, 'runtime')
    
    try:
        available_lang = ast.literal_eval(get_val(a, 'spoken_languages'))
    except:
        available_lang = []

    vote_rating = get_val(a, 'vote_average')
    vote_count = get_val(a, 'vote_count')
    movie_id = get_val(a, 'movie_id')
    
    try:
        genres = get_val(b, 'genres') 
    except:
        genres = []
        
    # Note: We use fetch_posters (Sync) here for the big detail image
    this_poster = fetch_posters(movie_id)
    
    cast_per = get_val(b, 'cast')
    try:
        cast_obj = ast.literal_eval(cast_per)
        cast_id = [i['id'] for i in cast_obj]
    except:
        cast_id = []

    lang = [i['name'] for i in available_lang]
    
    director_data = get_val(b, 'crew')
    director_list = get_crew(director_data)

    info = [this_poster, budget, genres, overview, release_date, revenue, runtime, available_lang, vote_rating,
            vote_count, movie_id, None, director_list, lang, cast_id]

    return info