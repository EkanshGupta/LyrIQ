import streamlit as st
import helper
from ann import getNN
import numpy as np
import pickle

with open("embedding_genre.pkl", "rb") as f:
    embed_list, genre_list, tid_list = pickle.load(f)
with open("track_artist.pkl", "rb") as pf:
    artist_info = pickle.load(pf)
with open("artist_genre.pkl", "rb") as pf:
    genre_info = pickle.load(pf)
with open("song_embeddings.pkl", "rb") as pf:
    embeds = pickle.load(pf)

st.set_page_config(layout="wide")
nn_model = getNN(np.array(embed_list))

st.title("Music Recommender")

st.header("Find Songs by Your Mood")

user_mood = st.text_input("How are you feeling right now?", placeholder="e.g., I feel nostalgic and calm...")
if user_mood:
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.write("Finding songs that match your mood...")
        faiss_reco, annoy_reco = helper.match_song_to_embed(user_mood, embed_list, genre_list, tid_list)

        for tid in faiss_reco:
            meta = artist_info.get(tid, {})
            st.markdown(f"- **{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
        # for tid in annoy_reco:
        #     meta = artist_info.get(tid, {})
        #     st.markdown(f"- **{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
    with col2:
        st.subheader("Visualizing your mood in the song space:")
        query_vec = helper.convert_str_to_embed(user_mood)
        fig = helper.visualize_query_embedding_umap(query_vec, embed_list, genre_list, tid_list, artist_info)
        st.plotly_chart(fig, use_container_width=True)

st.header("Find Similar Songs to a Song You Like")

song_name = st.text_input("Enter a song name (and optionally artist name)", placeholder="e.g., Shape of You by Ed Sheeran")
if song_name:
    col1, col2 = st.columns([1, 1.5])
    st.write("Searching for the song on Spotify...")
    tid = helper.find_spotify_id(song_name)
    if tid:
        # keys = list(artist_info.keys())
        # for i in range(5):
        #     st.write(keys[i])
        #     st.write(artist_info[keys[i]])
        # st.write(tid)
        if tid not in artist_info:
            st.warning("Sorry, this song is not present in the database. Try another song")
        else:
            faiss_reco, annoy_reco = helper.get_recommendations(tid, embeds, embed_list, genre_list, tid_list, artist_info, genre_info, 5)

            with col1:
                st.write(f"Songs similar to **{song_name}**:")
                # for tid in faiss_reco:
                #     meta = artist_info.get(tid, {})
                #     st.markdown(f"- **{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
                for tid in annoy_reco:
                    meta = artist_info.get(tid, {})
                    st.markdown(f"- **{meta.get('track_name', 'Unknown')}** by *{meta.get('artist_name', 'Unknown')}*")
            with col2:
                st.subheader("Visualizing your selected song in the song space:")
                query_vec = embeds[tid]['lyric_embed']
                fig = helper.visualize_query_embedding_umap(query_vec, embed_list, genre_list, tid_list, artist_info)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Couldn't find that song on Spotify. Please check the name.")
