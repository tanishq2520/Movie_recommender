import string
import pickle
import pandas as pd
import ast
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Object for porterStemmer
ps = PorterStemmer()

# Ensure stopwords are downloaded safely
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
    # Reading both the csv files
    # Ensure your file paths match these exactly
    credit_ = pd.read_csv(r'Files/tmdb_5000_credits.csv')
    movies = pd.read_csv(r'Files/tmdb_5000_movies.csv')

    # Merging the dataframes
    movies = movies.merge(credit_, on='title')

    # Create movies2 for details usage later
    movies2 = movies.copy()
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    # Extracting important and relevant features
    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies', 'release_date']]

    # --- CRITICAL FIX: DROP NA AND RESET INDEX ---
    # This fixes the "IndexError: index out of bounds"
    movies.dropna(inplace=True)
    movies.reset_index(drop=True, inplace=True)
    # ---------------------------------------------

    # Applying functions to convert from list string to list of items
    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    # Removing spaces from between the words (e.g., "Sci Fi" -> "SciFi")
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Creating 'tags' where we have all the words together for analysis
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    # Creating new dataframe for the analysis part only
    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']].copy()

    # Join lists back into strings for vectorization
    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

    new_df['tcrew'] = new_df['tcrew'].apply(lambda x: " ".join(x))

    # Lowercase everything for consistency
    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())
    new_df['tcrew'] = new_df['tcrew'].apply(lambda x: x.lower())

    # Applying stemming
    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def stemming_stopwords(li):
    ans = []
    for i in li:
        ans.append(ps.stem(i))

    # Removing Stopwords
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

    # Removing Punctuations
    punc = string.punctuation
    str_ = str_.translate(str_.maketrans('', '', punc))
    return str_


def fetch_posters(movie_id):
    try:
        url = 'https://api.themoviedb.org/3/movie/{}?api_key=6177b4297dff132d300422e0343471fb'.format(movie_id)
        # Timeout added to prevent hanging if internet is slow
        response = requests.get(url, timeout=5)
        data = response.json()
        str_ = "https://image.tmdb.org/t/p/w780/" + data['poster_path']
    except:
        # Fallback image if API fails or no poster exists
        str_ = "https://via.placeholder.com/500x750?text=No+Image"
    return str_


def recommend(new_df, movie, pickle_file_path):
    with open(pickle_file_path, 'rb') as pickle_file:
        similarity_tags = pickle.load(pickle_file)

    try:
        # Find the index of the movie
        # Because we reset_index in read_csv_to_df, these indices now align perfectly
        idx_series = new_df[new_df['title'] == movie].index
        
        if idx_series.empty:
            return [], []
            
        movie_idx = idx_series[0]

        # Getting the top 25 movies based on similarity scores
        distances = similarity_tags[movie_idx]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:26]

        rec_movie_list = []
        rec_poster_list = []

        for i in movie_list:
            # Fetch title
            rec_movie_list.append(new_df.iloc[i[0]]['title'])
            # Fetch poster
            rec_poster_list.append(fetch_posters(new_df.iloc[i[0]]['movie_id']))

        return rec_movie_list, rec_poster_list

    except IndexError:
        return [], []
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return [], []


def vectorise(new_df, col_name):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec_tags = cv.fit_transform(new_df[col_name]).toarray()
    sim_bt = cosine_similarity(vec_tags)
    return sim_bt


def fetch_person_details(id_):
    try:
        url = 'https://api.themoviedb.org/3/person/{}?api_key=6177b4297dff132d300422e0343471fb'.format(id_)
        data = requests.get(url, timeout=5).json()

        if 'profile_path' in data and data['profile_path']:
            url = 'https://image.tmdb.org/t/p/w220_and_h330_face' + data['profile_path']
        else:
            url = "https://via.placeholder.com/220x330?text=No+Image"

        biography = data.get('biography', "No biography available.")
    except:
        url = "https://via.placeholder.com/220x330?text=Error"
        biography = "Could not fetch details."

    return url, biography


def get_details(selected_movie_name):
    # Load the dictionaries
    pickle_file_path = r'Files/movies_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    movies = pd.DataFrame.from_dict(loaded_dict)

    pickle_file_path = r'Files/movies2_dict.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_dict_2 = pickle.load(pickle_file)
    movies2 = pd.DataFrame.from_dict(loaded_dict_2)

    # Find the movie rows
    a = movies2[movies2['title'] == selected_movie_name]
    b = movies[movies['title'] == selected_movie_name]

    if a.empty or b.empty:
        return None

    # Helper to safely get value
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