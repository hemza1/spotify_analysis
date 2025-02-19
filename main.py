from dotenv import load_dotenv
import os
import pandas as pd
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
    df = pd.DataFrame(track_details)
    return df
    

token = get_token()

df_2025 = get_playlist_tracks(token, "6P5uEfKaqzr3eVvdIMhRDX")
df_2024 = get_playlist_tracks(token, "6tu5BRo8Q6L8Sok9No2MpL")
df_2023 = get_playlist_tracks(token, "47p068fujLh1Ku0NOY9E0z")
df_2022 = get_playlist_tracks(token, "6CtatNDDA4F0xyV23btSaF")
df_2021 = get_playlist_tracks(token, "4s2mBJV1iwonpDM5A85YWp")

print(token)

print(df_2025.head())
print("****************")

print(df_2024.head())
print("****************")

print(df_2023.head())
print("****************")

print(df_2022.head())
print("****************")
print(df_2021.head())


df_2025['year'] = 2025
df_2024['year'] = 2024
df_2023['year'] = 2023
df_2022['year'] = 2022
df_2021['year'] = 2021

# Fusionner les DataFrames
df_combined = pd.concat([df_2025, df_2024, df_2023, df_2022 , df_2021])

# Nettoyage des données : Suppression des lignes sans genre
df_clean = df_combined.dropna(subset=['genre'])


df_clean = df_combined.dropna(subset=['genre'])

df_clean = df_combined[df_combined['genre'].apply(lambda x: len(x) > 0)]  # Garde les lignes où 'genre' n'est pas vide
df_clean['genre'] = df_clean['genre'].apply(lambda x: x[0] if isinstance(x, list) else x)

plt.figure(figsize=(10, 6))
sns.countplot(data=df_clean, x='year', hue='genre', palette='Set2')
plt.title("Distribution des genres au fil des années")
plt.xlabel("Année")
plt.ylabel("Nombre de morceaux")
plt.xticks(rotation=45)
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

artist_counts = df_clean['artist'].value_counts().head(20)  # Top 20 des artistes les plus populaires
plt.figure(figsize=(10, 6))
sns.barplot(x=artist_counts.values, y=artist_counts.index, palette='viridis')
plt.title("Top 20 des artistes les plus populaires")
plt.xlabel("Nombre d'apparitions")
plt.ylabel("Artistes")
plt.tight_layout()
plt.show()


artist_longevity = df_clean.groupby('artist')['year'].nunique().sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=artist_longevity.values, y=artist_longevity.index, palette='coolwarm')
plt.title("Longévité des artistes (nombre d'années où l'artiste est présent)")
plt.xlabel("Nombre d'années")
plt.ylabel("Artistes")
plt.tight_layout()
plt.show()


genre_trends = df_clean.groupby(['year', 'genre']).size().unstack().fillna(0)
genre_trends.plot(kind='line', figsize=(12, 6), title="Évolution de la popularité des genres")
plt.ylabel("Nombre de morceaux")
plt.xlabel("Année")
plt.tight_layout()
plt.show()



# Calcul de la diversité des genres par année
genre_diversity = df_clean.groupby('year')['genre'].nunique()

# Visualisation de la diversité des genres au fil du temps
plt.figure(figsize=(12, 6))
genre_diversity.plot(kind='line', marker='o', color='b')
plt.title("Diversité des genres musicaux au fil des années")
plt.xlabel("Année")
plt.ylabel("Nombre de genres distincts")
plt.tight_layout()
plt.show()


# Trouver les morceaux les plus récurrents dans les playlists
top_tracks = df_clean['track_name'].value_counts().head(20)

# Affichage des morceaux les plus populaires
plt.figure(figsize=(10, 6))
sns.barplot(x=top_tracks.values, y=top_tracks.index, palette='viridis')
plt.title("Top 20 des morceaux qui apparraissent dans plusieurs années")
plt.xlabel("Nombre d'apparitions")
plt.ylabel("Morceaux")
plt.tight_layout()
plt.show()


# Nettoyage des données pour obtenir une structure correcte pour la régression
genre_popularity = df_clean.groupby(['year', 'genre']).size().unstack(fill_value=0)

# Structure de X : années comme features
X = genre_popularity.index.values.reshape(-1, 1)  # Utiliser les années comme feature

# Structure de y : nombre de morceaux pour chaque genre
y = genre_popularity.values  # Nombre de morceaux pour chaque genre (un vecteur par genre)

# Séparation en ensemble d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression pour chaque genre
from sklearn.linear_model import LinearRegression

# Initialiser un modèle pour chaque genre
models = {genre: LinearRegression() for genre in genre_popularity.columns}

# Entraînement des modèles pour chaque genre
for genre in genre_popularity.columns:
    model = models[genre]
    model.fit(X_train, y_train[:, genre_popularity.columns.get_loc(genre)])  # Entraînement avec les données du genre

# Prédictions
y_pred = {genre: model.predict(X_test) for genre, model in models.items()}

# Calcul de l'erreur pour chaque genre
from sklearn.metrics import mean_squared_error

for genre in genre_popularity.columns:
    mse = mean_squared_error(y_test[:, genre_popularity.columns.get_loc(genre)], y_pred[genre])
    print(f'Mean Squared Error for {genre}: {mse}')

    


# Nettoyage des données pour obtenir une structure correcte pour la régression
genre_popularity = df_clean.groupby(['year', 'genre']).size().unstack(fill_value=0)

# Structure de X : années comme features
X = genre_popularity.index.values.reshape(-1, 1)  # Utiliser les années comme feature

# Structure de y : nombre de morceaux pour chaque genre
y = genre_popularity.values  # Nombre de morceaux pour chaque genre (un vecteur par genre)

# Séparation en ensemble d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle avec Random Forest Regressor pour chaque genre
from sklearn.ensemble import RandomForestRegressor

# Initialiser un modèle Random Forest pour chaque genre
models = {genre: RandomForestRegressor(n_estimators=100, random_state=42) for genre in genre_popularity.columns}

# Entraînement des modèles pour chaque genre
for genre in genre_popularity.columns:
    model = models[genre]
    model.fit(X_train, y_train[:, genre_popularity.columns.get_loc(genre)])  # Entraînement avec les données du genre

# Prédictions
y_pred = {genre: model.predict(X_test) for genre, model in models.items()}

# Calcul de l'erreur pour chaque genre
from sklearn.metrics import mean_squared_error

for genre in genre_popularity.columns:
    mse = mean_squared_error(y_test[:, genre_popularity.columns.get_loc(genre)], y_pred[genre])
    print(f'Mean Squared Error for {genre}: {mse}')

# Prévoir la popularité des genres pour les années futures (exemple : 2024 à 2027)
future_years = np.array([2024, 2025, 2026, 2027]).reshape(-1, 1)

# Prédire la popularité des genres pour les années futures
future_predictions = {}

for genre in genre_popularity.columns:
    model = models[genre]
    future_predictions[genre] = model.predict(future_years)  # Prédictions pour les années futures

# Créer un DataFrame pour afficher les prédictions futures
future_predictions_df = pd.DataFrame(future_predictions, index=[2024, 2025, 2026, 2027])

# Afficher les prédictions pour les années futures
print("\nPrédictions pour les années futures :")
print(future_predictions_df)


"""on peut voir french pop est dominante dans y dans la majorité des années , ceci est probablement liée à sa volatilité et à l'impact de certains événements spécifiques

Les prédictions pour 2024 à 2027 montrent des tendances intéressantes :

Genres populaires comme "french pop" : La popularité de "french pop" semble diminuer dans les années à venir, avec une forte chute en 2025 (de 29.61 à 14.35), avant de se stabiliser.

Genres émergents ou stables comme "pop urbaine" et "lullaby" : Pop urbaine montre une très forte hausse de popularité prévue (de 0.99 en 2024 à 3.01 en 2025), et cette tendance est stable pendant plusieurs années.

Lullaby reste assez stable, avec une légère augmentation prévue.

Genres à faible popularité comme "downtempo" ou "zouk" : Ces genres ne semblent pas avoir beaucoup de croissance prévue dans les années à venir, avec des prévisions de valeurs assez faibles (proches de 0 pour 2024-2027).

genre
"""