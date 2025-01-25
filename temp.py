from fuzzywuzzy import process
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

movies = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/60876/118283/movies.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250125%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250125T141931Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=73e8c7d984acc1976b9a18f8543e17a241a44b1e1ad0041f6134720c4e444ed08328b9a42da15d506eff7baebbcf2395df4eb875a3cd55cb4e16330634a757b2666fa4473215ee6eb1cd2daf2ed66a34846e9dcf5958ebf89000531c4cba4ee41bc84dc469a54a71115b389d83935387322cc102bbf1a8157747d3687bf3c884347c5c2b79ab5b18c224f2fafe109c467e5b1d1767722182aeac30b947544b874a7414f1819e9ea63aa8b7ef68c4b1d2b7e86e93b0fd76c38f1c69f85364593134858b1723aac503e8c0392e78a1519e81a1db57e20a152336b5578df28d87457e3d978466fcece6a2a27ac8c98ee64c4459b71fc7f1f1c04add57391ae5dca2")
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

recommendations = get_movie_recommendation('Memento')
print(recommendations)