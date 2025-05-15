import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
# from tqdm.notebook import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import os
from lyricsgenius import Genius
from sys import getsizeof
import requests
import globals
import pickle
import time
import re
from glob import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from canon import CANONICAL_GENRE_MAP
from sklearn.manifold import TSNE
import plotly.express as px
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import ann

SAVE_EVERY = 100  # Save every 100 playlists
DATA_DIR = '../datasets/spotify/data/'
SAVE_DIR = '../datasets/spotify/processed_data/'

FILE_STATE_PATH = os.path.join(SAVE_DIR, 'file_state.json')
FILE_STATE_PATH_GENRE = os.path.join(SAVE_DIR, 'genre_state.json')


def patch_dict(filename):
    genius = Genius(globals.genius_client_access_token, timeout=15, retries=3, sleep_time=1)
    part_save = os.path.basename(filename).split('.')[0]+'_patched.pkl'
    print(part_save)
    with open(filename, "rb") as pf:
        data = pickle.load(pf)

    song_new_id=0
    song_no_id=0
    song_id_no_lyric=0

    print(f"total entries in dict: {len(data)}")
    for i, key in enumerate(data):
        pairs = data[key]
        if pairs["genius_id"]==None:
            song_no_id+=1
#             print([pairs['track_name'], pairs['artist_name']])
            #Need to get ID and lyrics both
            genius_song_id, title, url, comp = get_genius_song_id(pairs['track_name'], pairs['artist_name'])
            if comp == 1:
                song_new_id += 1
                data[key]['genius_id'] = genius_song_id
                data[key]['lyric'] = get_lyric(genius, genius_song_id)
        else:
            if pairs["lyric"]==None:
                song_id_no_lyric+=1
                data[key]['lyric'] = get_lyric(genius, pairs["genius_id"])

        if (i+1)%100==0:
            print(f"Iteration {i+1}/{len(data)}")
            print(f"Songs with no IDs: {song_no_id}")
            print(f"Songs with new IDs: {song_new_id}")
            print(f"Songs with IDs but no lyric: {song_id_no_lyric}")
            print(f"Unsuccessful lyric attempts: {globals.unsuccessful_lyric}")


    with open(part_save, "wb") as f:
        pickle.dump(data, f)


def merge_dicts():
    files = glob("track_metadata_*.pkl")
    print(files)

    merged_dict = {}
    max_range = (0, 0)
    final_filename = ""

    for f in files:
        match = re.search(r"track_metadata_(\d+)-(\d+)\.pkl", f)
        if match:
            x, y = map(int, match.groups())
            if y > max_range[1]:
                max_range = (x, y)
                final_filename = f

            with open(f, "rb") as pf:
                data = pickle.load(pf)
                merged_dict.update(data)
        print(f"Updated dict has {len(merged_dict)} entries")

    output_filename = f"track_metadata_{max_range[0]}-{max_range[1]}_merged.pkl"
    with open(output_filename, "wb") as out:
        pickle.dump(merged_dict, out)
    print(f"Merged {len(files)} files into: {output_filename}")


def get_genius_song_id(song_name, artist_name):
    base_url = "https://api.genius.com/search"
    headers = {'Authorization': f'Bearer {globals.genius_client_access_token}'}
    params = {'q': f"{song_name} {artist_name}"}

    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()

    try:
        for hit in data["response"]["hits"]:
            if artist_name.lower() in hit["result"]["primary_artist"]["name"].lower():
                return hit["result"]["id"], hit["result"]["title"], hit["result"]["url"], 1
    except Exception as e:
        print(data)
        print(f"ID not found: {song_name} by {artist_name} ")

    return None, None, None, 0


def get_lyric(genius_object, song_id):
    try:
        lyric = genius_object.lyrics(song_id)
    except Exception as e:
        globals.unsuccessful_lyric+=1
        lyric = None
    return lyric

def load_state():
    if os.path.exists(FILE_STATE_PATH):
        with open(FILE_STATE_PATH, 'r') as f:
            return json.load(f)
    else:
        all_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.json')])
        return {"current_file": None, "remaining_files": all_files, "part_index": 0, "playlist_index": 0}

def load_state_genre():
    if os.path.exists(FILE_STATE_PATH_GENRE):
        with open(FILE_STATE_PATH_GENRE, 'r') as f:
            return json.load(f)
    else:
        all_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.json')])
        return {"current_file": None, "remaining_files": all_files}


def save_state(current_file, remaining_files, part_index, playlist_index):
    with open(FILE_STATE_PATH, 'w') as f:
        json.dump({
            "current_file": current_file,
            "remaining_files": remaining_files,
            "part_index": part_index,
            "playlist_index": playlist_index
        }, f)

def save_state_genre(current_file, remaining_files):
    with open(FILE_STATE_PATH_GENRE, 'w') as f:
        json.dump({
            "current_file": current_file,
            "remaining_files": remaining_files,
        }, f)


def get_artist_genre():
    auth = SpotifyClientCredentials(client_id=globals.client_id, client_secret=globals.client_secret)
    sp = spotipy.Spotify(auth_manager=auth)
    token_info = auth.get_access_token(as_dict=True)
    print(token_info)
    artist_uri_list=[]
#     artist_name_list=[]
#     song_name_list=[]
    state = load_state_genre()
    remaining_files = state["remaining_files"]
    current_file = state["current_file"]
    if current_file!=None:
        with open(f"artist_genre.pkl", 'rb') as f:
            artist_genre = pickle.load(f)
    else:
        artist_genre={}

    for idx, file in enumerate(remaining_files):
        start_time = time.time()
        with open(file, 'r') as f:
            data = json.load(f)
        playlists = data['playlists']
        for i, playlist in enumerate(playlists):
            playlist_track_pairs = []
            pid = playlist['pid']
            for track in playlist['tracks']:
                artist_uri = track['artist_uri'].split(":")[-1]
                if artist_uri not in artist_genre and artist_uri not in artist_uri_list:
                    artist_uri_list.append(artist_uri)
#                     artist_name_list.append(track['artist_name'])
#                     song_name_list.append(track['track_name'])
                    if len(artist_uri_list)==49:
                        results = sp.artists(artist_uri_list)
                        for artist in results['artists']:
                            artist_genre[artist['id']] = {
                                'name': artist['name'],
                                'genre':    artist['genres'],
                                'popularity': artist['popularity'],
                                'followers': artist['followers']['total']
                            }
                        artist_uri_list=[]
                        time.sleep(0.3)

        if len(artist_uri_list)>0:
            results = sp.artists(artist_uri_list)
            for artist in results['artists']:
                            artist_genre[artist['id']] = {
                                'name': artist['name'],
                                'genre':    artist['genres'],
                                'popularity': artist['popularity'],
                                'followers': artist['followers']['total']
                            }
            artist_uri_list=[]
            time.sleep(0.3)

        with open(f"artist_genre.pkl", "wb") as f:
            pickle.dump(artist_genre, f)
        remaining_files = remaining_files[1:]
        save_state_genre(file, remaining_files)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"File number {idx+1}")
        print(f"Number of artists cached: {len(artist_genre)}")
        print(f"Elapsed time: {elapsed:.4f} seconds")


def get_playlist_data():
    playlist_data=[]
    state = load_state()
    remaining_files = state["remaining_files"]
    current_file = state["current_file"]
    for idx, file in enumerate(remaining_files):
#         print(f"\nProcessing file: {file}")
        with open(file, 'r') as f:
            data = json.load(f)
        playlists = data['playlists']
        for i, playlist in enumerate(playlists):
            playlist_track_pairs = []
            pid = playlist['pid']
            for track in playlist['tracks']:
                tid = track['track_uri'].split(":")[-1]
                playlist_track_pairs.append(tid)
            playlist_data.append(playlist_track_pairs)
        if (idx+1)%100==0:
            print(idx)
    with open(f"playlist_data.pkl", "wb") as f:
        pickle.dump(playlist_data, f)
    print("Done")


def patch_track_dict(filename):
    with open(filename, "rb") as pf:
        data = pickle.load(pf)
    song_id=0
    song_no_id=0
    song_lyrics=0
    song_no_lyrics=0
    song_patched=0
    lyrics_patched=0
    print(f"total entries in dict: {len(data)}")
    for key in data:
        pairs = data[key]
        if pairs["genius_id"]==None:
            song_no_id+=1
            if pairs["lyric"]==None:
                song_no_lyrics+=1
            else:
                song_lyrics+=1
        else:
            song_id+=1

    print(f"song_id: {song_id}")
    print(f"song_no_id: {song_no_id}")
    print(f"song_lyrics: {song_lyrics}")
    print(f"song_no_lyrics: {song_no_lyrics}")


def get_track_data(current_file, part_load):
    genius = Genius(globals.genius_client_access_token, timeout=15, retries=3, sleep_time=1)
    part_save = os.path.basename(current_file).split(".")[2]
    with open(f"track_metadata_{part_load}.pkl", 'rb') as f:
        track_metadata = pickle.load(f)

    print(f"\nProcessing file: {current_file}")
    with open(current_file, 'r') as f:
        data = json.load(f)

    playlists = data['playlists']

    for i, playlist in enumerate(tqdm(playlists, desc="Playlist loop", leave=False)):
        successful_id = 0
        unsuccessful_id = 0
        pid = playlist['pid']

        for track in playlist['tracks']:
            tid = track['track_uri'].split(":")[-1]

            if tid not in track_metadata:
                genius_song_id, title, url, comp = get_genius_song_id(track['track_name'], track['artist_name'])
                track_metadata[tid] = {
                    'track_name': track['track_name'],
                    'artist_name': track['artist_name'],
                    'album_name': track['album_name'],
                    'artist_uri': track['artist_uri'].split(":")[-1],
                    'album_uri': track['album_uri'].split(":")[-1],
                    'genius_id': genius_song_id,
                    'lyric': None
                }
                if comp == 1:
                    successful_id += 1
                    track_metadata[tid]['lyric'] = get_lyric(genius, genius_song_id)
                else:
                    unsuccessful_id += 1

        print(f"Number of tracks in dict: {len(track_metadata)}")
        print(f"Successful ID: {successful_id}, Unsuccesful ID {unsuccessful_id}")
        print(f"Unsuccesful lyric {globals.unsuccessful_lyric}")
        print(f"track metadata keys: {len(track_metadata)} and track metadata memory: {getsizeof(track_metadata)} bytes")

        if (i+1) % SAVE_EVERY == 0:
            with open(f"track_metadata_{part_save}.pkl", "wb") as f:
                pickle.dump(track_metadata, f)
    with open(f"track_metadata_{part_save}.pkl", "wb") as f:
        pickle.dump(track_metadata, f)

def split_lyrics_and_description(full_text):
    match = re.search(r"\[\s*(Intro|Verse|Chorus|Bridge)[^\]]*\]", full_text)
    if match:
        split_index = match.start()
        description = full_text[:split_index].strip()
        lyrics = full_text[split_index:].strip()
        return description, lyrics
    else:
        # Fallback if no tags found
        return full_text.strip(), ""


def embed_lyrics(lines):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim output
    sentence_embeddings = model.encode(lines, normalize_embeddings=True)
    return np.mean(sentence_embeddings, axis=0)

def save_embeddings(filename="track_metadata_20k_playlists.pkl"):
    with open(filename, "rb") as pf:
        data = pickle.load(pf)

    song_embeddings = {}
    for tid in tqdm(data):
        song_info = data[tid]
        lyric_text = song_info['lyric']
        if lyric_text!=None:
            song_embeddings[tid] = {'song_brief':None, 'lyric_embed':None, 'brief_embed':None}
            description, lyric = split_lyrics_and_description(lyric_text)
            if lyric=="":
                lyric = description
            else:
                song_embeddings[tid]['song_brief'] = description
                song_embeddings[tid]['brief_embed'] = embed_lyrics(description.split("\n"))
            lyric_list = lyric.split("\n")
            song_embeddings[tid]['lyric_embed'] = embed_lyrics(lyric_list)


    with open("song_embeddings.pkl", "wb") as f:
        pickle.dump(song_embeddings, f)

def create_track_artist_dict():
    with open("track_metadata_20k_playlists.pkl", "rb") as pf:
        track_data = pickle.load(pf)

    tid_artist_dict={}
    #This dict will have key= tid and store artist id, artist genre, lyric embeddings

    for key in tqdm(track_data):
        info = data[key]
        tid_artist_dict[key] = {'track_name':info['track_name'],'artist_name':info['artist_name'],\
                                'artist_uri':info['artist_uri']}

    with open("track_artist.pkl", "wb") as f:
        pickle.dump(tid_artist_dict, f)

def fix_artist_genre():
    with open("artist_genre.pkl", "rb") as pf:
        genre_info = pickle.load(pf)
    for key in genre_info:
        info = genre_info[key]
        if isinstance(info, list):
            genre_info[key] = {
                                'name': None,
                                'genre':    info,
                                'popularity': None,
                                'followers': None
            }
    for key in genre_info:
        info = genre_info[key]
        if isinstance(info, dict):
            pass
        else:
            print("Error")
    with open("artist_genre_patched.pkl", "wb") as f:
        pickle.dump(genre_info, f)

def save_embed_genre_lists():
    with open("song_embeddings.pkl", "rb") as pf:
        embeds = pickle.load(pf)
    with open("track_artist.pkl", "rb") as pf:
        artist_info = pickle.load(pf)
    with open("artist_genre.pkl", "rb") as pf:
        genre_info = pickle.load(pf)

    genre_keys = list(genre_info.keys())
    # for i in range(5):
    #     print(genre_info[genre_keys[i]])

    embed_list=[]
    genre_list=[]
    tid_list=[]
    artist_genre_absent=0
    for tid in tqdm(embeds):
        info = embeds[tid]
        artist_id = artist_info[tid]['artist_uri']
        if artist_id in genre_info:
            genre = genre_info[artist_id]['genre']
            genre_list.append(genre)
            embed_list.append(info['lyric_embed'])
            tid_list.append(tid)
        else:
            artist_genre_absent+=1

    with open("embedding_genre.pkl", "wb") as f:
        pickle.dump([embed_list, genre_list, tid_list], f)

    print(artist_genre_absent)
    print(len(embed_list))

def normalize_genre(genre_entry):
    if isinstance(genre_entry, list):
        return "|".join(sorted(genre_entry))
    return genre_entry  # Already a string

def map_to_broad_genre(genre_str):
    if not genre_str or genre_str.strip() == "":
        return "unknown"

    subgenres = genre_str.lower().split("|")
    for sub in subgenres:
        if sub in CANONICAL_GENRE_MAP:
            return CANONICAL_GENRE_MAP[sub]
    return "other"

def plot_embeds_tsne(embed_list, genre_list, tid_list, num_pts=1000):
    for i in range(len(genre_list)):
        genre_list[i] = normalize_genre(genre_list[i])
        genre_list[i] = map_to_broad_genre(genre_list[i])
    embed_list = embed_list[:num_pts]
    genre_list = genre_list[:num_pts]
    print("Unique genres:", list(set(genre_list)))
    reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedding_2d = reducer.fit_transform(np.array(embed_list))
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'genre': genre_list
    })
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='genre',
        title="2D Song Embeddings (TSNE) Colored by Genre",
        hover_data=['genre'],
        width=1000,
        height=700
    )
    fig.update_layout(legend_title_text='Genre')
    fig.show()


def plot_embeds_umap(embed_list, genre_list, tid_list, artist_info, num_pts=1000):
    color_seq = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Plotly
    color_seq = color_seq[:20]
    color_seq = px.colors.qualitative.Dark24
    for i in range(len(genre_list)):
        genre_list[i] = normalize_genre(genre_list[i])
        genre_list[i] = map_to_broad_genre(genre_list[i])
    embed_list = embed_list[:num_pts]
    genre_list = genre_list[:num_pts]
    track_list = [artist_info[tid]['track_name'] for tid in tid_list[:num_pts]]
    artist_list = [artist_info[tid]['artist_name'] for tid in tid_list[:num_pts]]

    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embed_list)
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'genre': genre_list,
        'track': track_list,
        'artist': artist_list
    })
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='genre',
        title="2D Song Embeddings Colored by Genre",
        hover_data=['track', 'artist', 'genre'],
        color_discrete_sequence=color_seq,
        width=1000,
        height=700
    )
    fig.update_layout(legend_title_text='Genre')
    fig.show()


def visualize_query_embedding_umap(query_vec, embed_list, genre_list, tid_list, artist_info, num_pts=1000):
    embed_list = np.array(embed_list[:num_pts])
    genre_list = genre_list[:num_pts]
    tid_list = tid_list[:num_pts]

    all_embeds = np.vstack([embed_list, query_vec])
    labels = genre_list + ["__QUERY__"]
    track_names = [artist_info.get(tid, {}).get("track_name", "") for tid in tid_list] + ["Query Song"]
    artist_names = [artist_info.get(tid, {}).get("artist_name", "") for tid in tid_list] + ["You"]

    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(all_embeds)

    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': labels,
        'track': track_names,
        'artist': artist_names,
        'highlight': ['Query' if l == '__QUERY__' else 'Song' for l in labels]
    })

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='highlight',
        symbol='highlight',
        hover_data=['track', 'artist', 'label'],
        title="2D Projection of Songs with Query",
        width=1000,
        height=700
    )

    return fig

#Taked TID and recommends nearest TIDs. ANN returns indices so this function needs embed_list, genre_list, tid_list
def get_recommendations(query_vector, embeds, embed_list, genre_list, tid_list, artist_info, genre_info, num_recs):
    getnn = ann.getNN(np.array(embed_list))
    query_track_name = artist_info[query_vector]['track_name']
    # print(f"\n\nQuery song is {artist_info[query_vector]['track_name']} by {artist_info[query_vector]['artist_name']} \
    # of genre {genre_info[artist_info[query_vector]['artist_uri']]['genre']}")
    query_embedding = embeds[query_vector]['lyric_embed']
    faiss_reco=[]
    annoy_reco=[]
    k=100
    idx_faiss = getnn.getNN_faiss(query_embedding, k=k)
    for j in range(k):
        tid = tid_list[idx_faiss[j]]
        tid_name = artist_info[tid]['track_name']
        if query_track_name in tid_name:
            continue
        faiss_reco.append(tid)
        if len(faiss_reco)==num_recs:
            break
    idx_annoy = getnn.getNN_annoy(query_embedding, k=k)
    for j in range(k):
        tid = tid_list[idx_annoy[j]]
        tid_name = artist_info[tid]['track_name']
        if query_track_name in tid_name:
            continue
        annoy_reco.append(tid)
        if len(annoy_reco)==num_recs:
            break
    return faiss_reco, annoy_reco

def convert_str_to_embed(embed_string):
    lyric_list = embed_string.split("\n")
    embed = embed_lyrics(lyric_list)
    return embed

def match_song_to_embed(embed_string, embed_list, genre_list, tid_list):
    embed = convert_str_to_embed(embed_string)
    getnn = ann.getNN(np.array(embed_list))
    idx_faiss = getnn.getNN_faiss(embed)
    idx_annoy = getnn.getNN_annoy(embed)
    faiss_reco=[]
    annoy_reco=[]
    for j in range(5):
        tid = tid_list[idx_faiss[j]]
        faiss_reco.append(tid)
    for j in range(5):
        tid = tid_list[idx_annoy[j]]
        annoy_reco.append(tid)
    return faiss_reco, annoy_reco

def find_spotify_id(track_name, limit=1):
    try:
        results = globals.sp.search(q=track_name, type='track', limit=limit)
        items = results['tracks']['items']
        if items:
            track = items[0]
            return track['id']  # Or return a dict if needed
        else:
            return None
    except Exception as e:
        print("Error in Spotify search:", e)
        return None
