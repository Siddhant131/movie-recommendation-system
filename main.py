from fuzzywuzzy import process
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings1.csv")

print(movies.head())
ratings.head()
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.head()
final_dataset.fillna(0,inplace=True)
final_dataset.head()
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
final_dataset
#removing sparsity
sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
print(sparsity)
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
# make the system using knn
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]  
    if len(movie_list):        
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
        
        rec_movie_indices = sorted(
            list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
            key=lambda x: x[1]
        )[1:]  
        
        recommend_frame = []
        for val in rec_movie_indices:
            movie_id = final_dataset.iloc[val[0]]['movieId']
            # Fetch the movie title using the movieId
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            recommend_frame.append({'Title': movie_title, 'Distance': val[1]})
        
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        return df
    else:
        return "No movies found. Please check your input"

recommendations = get_movie_recommendation('Gone In 60 Seconds')
print(recommendations)