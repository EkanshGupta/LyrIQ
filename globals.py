import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import os
from lyricsgenius import Genius
from sys import getsizeof
import requests

client_id = ""
client_secret = ""

scope = "user-library-read"
genius_client_id = ""
genius_client_secret = ""
genius_client_access_token = ""

unsuccessful_lyric = 0
auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth)
