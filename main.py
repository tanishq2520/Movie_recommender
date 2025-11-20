import streamlit as st
from streamlit_extras.stoggle import stoggle
from processing import preprocess
from processing.display import Main
import base64

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

def set_background_and_style(poster_url=None):
    
    if poster_url:
        bg_image_style = f"""
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("{poster_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        """
    else:
        bg_image_style = """
            background-image: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
            background-size: cover;
        """

    st.markdown(f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        {bg_image_style}
        background-repeat: no-repeat !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 100vh !important;
        overflow: hidden !important;
        z-index: -5 !important;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        display: block;
        background: transparent;
        box-shadow:none !important;
        z-index: 0;
        opacity: 0.5;
        transform: scale(0.8); 
        pointer-events: none;
    }}

    div.block-container {{
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        padding-top: 2rem;
    }}

    .title-text {{
        font-size: 50px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        text-align: center !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        margin-bottom: 20px;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.8); 
    }}
    
    h1, h2, h3, h4, h5, h6, p, span, div {{
        color: #FFFFFF !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.8);
    }}

    div[data-baseweb="select"] > div {{
        background-color: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }}
    
    .stImage img {{
        transition: transform 0.3s ease;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }}
    .stImage img:hover {{
        transform: scale(1.05);
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        border: 2px solid rgba(255,255,255,0.5);
    }}

    .stButton button {{
        border-radius: 10px;
        font-weight: 700;
        height: 3rem;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}
    
    </style>
    """, unsafe_allow_html=True)



if 'view_state' not in st.session_state:
    st.session_state['view_state'] = 'idle'
if 'search_movie' not in st.session_state:
    st.session_state['search_movie'] = None
if 'search_basis' not in st.session_state:
    st.session_state['search_basis'] = 'General (Tags)'
if 'detail_movie' not in st.session_state:
    st.session_state['detail_movie'] = None
if 'cached_movies' not in st.session_state:
    st.session_state['cached_movies'] = []
if 'cached_posters' not in st.session_state:
    st.session_state['cached_posters'] = []
if 'cached_basis_str' not in st.session_state:
    st.session_state['cached_basis_str'] = ""

def main():
    with Main() as bot:
        bot.main_()
        new_df, movies, movies2 = bot.getter()
    

    st.markdown('<p class="title-text">🎬 🍿 Movie Recommender</p>', unsafe_allow_html=True)

    col_movie, col_basis = st.columns([7, 3])
    
    with col_movie:
        selected_movie = st.selectbox(
            '🔍 Select a Movie', 
            new_df['title'].values,
            label_visibility='visible'
        )
    
    with col_basis:
        basis = st.selectbox(
            '🎯 Recommendation Basis',
            ('General (Tags)', 'Genre', 'Cast', 'Production Company', 'Keywords','Director'),
            label_visibility='visible'
        )

    try:
        row = new_df[new_df['title'] == selected_movie]
        if not row.empty:
            mov_id = row.iloc[0]['movie_id']
            bg_poster = preprocess.fetch_posters(mov_id)
            if "via.placeholder" in bg_poster:
                set_background_and_style(None)
            else:
                set_background_and_style(bg_poster)
        else:
            set_background_and_style(None)
    except:
        set_background_and_style(None)

    st.write("") 
    btn_col1, btn_col2, spacer_col = st.columns([1, 1, 8])
    
    with btn_col1:
        rec_pressed = st.button('✨ Recommend', type="primary")
    with btn_col2:
        desc_pressed = st.button('📜 Describe')

    if rec_pressed:
        st.session_state['view_state'] = 'recommend'
        st.session_state['search_movie'] = selected_movie
        st.session_state['search_basis'] = basis
        st.session_state['detail_movie'] = None
        
        basis_mapping = {
            'General (Tags)': (r'Files/similarity_tags_tags.pkl', "are"),
            'Genre': (r'Files/similarity_tags_genres.pkl', "on the basis of genres are"),
            'Cast': (r'Files/similarity_tags_tcast.pkl', "on the basis of cast are"),
            'Production Company': (r'Files/similarity_tags_tprduction_comp.pkl', "from the same production company are"),
            'Keywords': (r'Files/similarity_tags_keywords.pkl', "on the basis of keywords are"),
            'Director': (r'Files/similarity_tags_tcrew.pkl', "directed by the same person are"),
        }
        
        selected_path, selected_str = basis_mapping[basis]
        
        
        with st.spinner('Analyzing content...'):
            movies, posters = preprocess.recommend(new_df, selected_movie, selected_path)
        
        st.session_state['cached_movies'] = movies
        st.session_state['cached_posters'] = posters
        st.session_state['cached_basis_str'] = selected_str

    if desc_pressed:
        st.session_state['view_state'] = 'describe'
        st.session_state['search_movie'] = selected_movie
        st.session_state['detail_movie'] = selected_movie

    st.divider()

    if st.session_state['view_state'] == 'recommend':
        render_recommendations()
    
    elif st.session_state['view_state'] == 'describe':
        render_description()

def render_recommendations():
    movies = st.session_state['cached_movies']
    posters = st.session_state['cached_posters']
    selected_str = st.session_state['cached_basis_str']
    active_movie = st.session_state['search_movie']

    if len(movies) > 0:
        st.markdown(f"### Results for \"{active_movie}\"")
        st.caption(f"Recommendations {selected_str}...")
        st.write("")

        cols = st.columns(5)
        for i in range(min(5, len(movies))):
            with cols[i]:
                st.image(posters[i], width="stretch")
                st.markdown(f"**{movies[i]}**")
                
                if st.button(f"View Details", key=f"btn_{i}", width="stretch"):
                    st.session_state['view_state'] = 'describe'
                    st.session_state['detail_movie'] = movies[i]
                    st.rerun()
    else:
        if st.session_state['search_movie']:
            st.info("Select a movie and click Recommend to start.")

def render_description():
    target_movie = st.session_state['detail_movie']
    
    if st.button("← Back to Results"):
        st.session_state['view_state'] = 'recommend'
        st.session_state['detail_movie'] = None
        st.rerun()

    with st.spinner('Loading details...'):
        info = preprocess.get_details(target_movie)

    with st.container():
        col1, col2 = st.columns([1, 2], gap="medium")
        with col1:
            st.image(info[0], width="stretch")
        
        with col2:
            st.markdown(f"## {target_movie}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Rating", f"{info[8]}/10")
            c2.metric("Votes", info[9])
            c3.metric("Runtime", f"{info[6]} min")
            
            st.write("#### Overview")
            st.info(info[3])

            st.write("#### Key Info")
            d1, d2 = st.columns(2)
            d1.text(f"Release: {info[4]}")
            d1.text(f"Budget: ${info[1]:,}")
            
            d2.text(f"Genres: {', '.join(info[2])}")
            if info[12]:
                d2.text(f"Director: {info[12][0]}")

    st.divider()
    st.subheader("Cast")
    
    cast_ids = info[14]
    c_cols = st.columns(5)
    for i in range(min(5, len(cast_ids))):
        person_id = cast_ids[i]
        url, biography = preprocess.fetch_person_details(person_id)
        with c_cols[i]:
            st.image(url, width="stretch")
            stoggle("Bio", biography)

if __name__ == '__main__':
    main()