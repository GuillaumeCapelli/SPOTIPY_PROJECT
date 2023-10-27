import numpy as np
import pandas as pd
import spotipy
import pickle
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

df = pd.read_csv('master.csv')

song_name_input = st.text_input('Please enter the song title: ')

if song_name_input:
        results = sp.search(q=song_name_input,limit=20,market="GB")

        name_list = []
        artist_list = []
        album_list = []
        id_list = []
        song_df = pd.DataFrame()
        for item in results['tracks']['items']:
                name_list.append(item['name'])
                artist_list.append(item['album']['artists'][0]['name'])
                album_list.append(item['album']['name'])
                id_list.append(item['id'])
        song_df['name'] = name_list
        song_df['artist'] = artist_list
        song_df['album'] = album_list
        song_df['id'] = id_list

        st.write(song_df)

        song_id_index = st.text_input('Please pick the index for the correct song: ')

        if song_id_index:
                song_id = song_df.iloc[int(song_id_index)]['id']

                song_audio_features = sp.audio_features(song_id)

                acousticness = []
                danceability = []
                energy = []
                instrumentalness = []
                speechiness = []
                tempo =[]
                valence =[]
                key = []
                time_signature = []
                audio_features_df = pd.DataFrame()
                for item in song_audio_features:
                        acousticness.append(item['acousticness'])
                        danceability.append(item['danceability'])
                        energy.append(item['energy'])
                        instrumentalness.append(item['instrumentalness'])
                        speechiness.append(item['speechiness'])
                        tempo.append(item['tempo'])
                        valence.append(item['valence'])
                        key.append(item['key'])
                        time_signature.append(item['time_signature'])
                audio_features_df['acousticness'] = acousticness
                audio_features_df['danceability'] = danceability
                audio_features_df['energy'] = energy
                audio_features_df['instrumentalness'] = instrumentalness
                audio_features_df['speechiness'] = speechiness
                audio_features_df['tempo'] = tempo
                audio_features_df['key'] = key
                audio_features_df['valence'] = valence
                audio_features_df['time_signature'] = time_signature

                with open("scaler.pickle", "rb") as f:
                        scaler = pickle.load(f)
                        
                with open("kmeans75.pickle", "rb") as f:
                        kmeans75 = pickle.load(f)
                        
                with open("kmeans100.pickle", "rb") as f:
                        kmeans100 = pickle.load(f)
                        
                with open("kmeans150.pickle", "rb") as f:
                        kmeans150 = pickle.load(f)
                        
                with open("kmeans200.pickle", "rb") as f:
                        kmeans200 = pickle.load(f)

                df_scaled = scaler.transform(audio_features_df)
                kmeans75_song = kmeans75.predict(df_scaled)
                kmeans100_song = kmeans100.predict(df_scaled)
                kmeans150_song = kmeans150.predict(df_scaled)
                kmeans200_song = kmeans200.predict(df_scaled)
                
                st.write(kmeans75_song)
                st.write(kmeans100_song)
                st.write(kmeans150_song)
                st.write(kmeans200_song)
                
                num_recommendations = st.slider('Select the number of recommendations you want', 1, 20, 5)

                st.write(df[['name','artist','album']][(df['cluster75'] == kmeans75_song[0]) & (df['cluster100'] == kmeans100_song[0]) & (df['cluster150'] == kmeans150_song[0]) & (df['cluster200'] == kmeans200_song[0])].sample(num_recommendations))