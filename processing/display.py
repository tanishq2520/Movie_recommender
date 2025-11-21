import os
import pickle
from processing import preprocess

def check_and_build_files():
    required_files = [
        r'Files/movies_dict.pkl',
        r'Files/movies2_dict.pkl',
        r'Files/new_df_dict.pkl',
        r'Files/similarity_tags_tags.pkl',
        r'Files/similarity_tags_genres.pkl',
        r'Files/similarity_tags_tcast.pkl',
        r'Files/similarity_tags_tprduction_comp.pkl',
        r'Files/similarity_tags_keywords.pkl',
        r'Files/similarity_tags_tcrew.pkl'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if not missing_files:
        return

    if not os.path.exists('Files'):
        os.makedirs('Files')

    movies, new_df, movies2 = preprocess.read_csv_to_df()

    with open(r'Files/movies_dict.pkl', 'wb') as f:
        pickle.dump(movies.to_dict(), f)

    with open(r'Files/movies2_dict.pkl', 'wb') as f:
        pickle.dump(movies2.to_dict(), f)

    with open(r'Files/new_df_dict.pkl', 'wb') as f:
        pickle.dump(new_df.to_dict(), f)

    columns_mapping = [
        ('tags', 'similarity_tags_tags.pkl'),
        ('genres', 'similarity_tags_genres.pkl'),
        ('tcast', 'similarity_tags_tcast.pkl'),
        ('tprduction_comp', 'similarity_tags_tprduction_comp.pkl'),
        ('keywords', 'similarity_tags_keywords.pkl'),
        ('tcrew', 'similarity_tags_tcrew.pkl')
    ]

    for col, filename in columns_mapping:
        sim_matrix = preprocess.vectorise(new_df, col)
        with open(fr'Files/{filename}', 'wb') as f:
            pickle.dump(sim_matrix, f)

if __name__ == "__main__":
    check_and_build_files()