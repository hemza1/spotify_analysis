from dotenv import load_dotenv
import os
import base64
from requests import post 
import json 
import requests
load_dotenv()

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")

print(client_id,client_secret)


def get_token():
    auth_string = client_id +":"+ client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url ="https://accounts.spotify.com/api/token"
    header = {
        "Authorization": "Basic " + auth_base64 , 
        "Content-Type": "application/x-www-form-urlencoded"

    }
    data = {"grant_type" : "client_credentials"}
    result = post(url , headers=header , data=data)
    json_result = json.loads(result.content)
    token= json_result["access_token"]
    return token 


def get_playlist_tracks(token: str, playlist_id: str):
    """
    Récupère les morceaux d'une playlist Spotify à partir de son ID.

    Args:
        token (str): Spotify API access token.
        playlist_id (str): ID unique de la playlist.

    Returns:
        list: Une liste de dictionnaires contenant track_name, artist et genre.
    """
    playlist_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(playlist_url, headers=headers)

    if response.status_code != 200:
        print("❌ Erreur API Spotify:", response.json())
        return {"error": f"Impossible de récupérer la playlist: {response.json()}"}

    tracks = response.json().get("items", [])
    track_details = []

    for item in tracks:
        track = item.get("track")
        if track:
            track_name = track["name"]
            artist_name = track["artists"][0]["name"]
            artist_id = track["artists"][0]["id"]

            # Récupérer les genres de l'artiste
            artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
            artist_response = requests.get(artist_url, headers=headers)

            if artist_response.status_code == 200:
                genre = artist_response.json().get("genres", ["Unknown"])
            else:
                genre = ["Unknown"]

            track_details.append({
                "track_name": track_name,
                "artist": artist_name,
                "genre": genre
            })

    return track_details

token = get_token()

tracks = get_playlist_tracks(token, "23psvx6vUY6pmJHxE5yagM")


for track in tracks:
    print(track)


   

print(token)