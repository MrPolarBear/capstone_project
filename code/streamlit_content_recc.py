import streamlit as st
import numpy as np 
import os 
import pandas as pd 
import re
import string
import seaborn as sns
from rake_nltk import Rake

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix, hstack
from scipy.sparse import vstack

from fuzzywuzzy import fuzz



df_anime_nomovie = pd.read_csv("../data/df_anime_nomovie.csv")
df_anime_all = pd.read_csv("../data/df_anime_all_cleaned.csv")

df_anime_nomovie.dropna(inplace = True)
df_anime_all.dropna(inplace = True)

df_anime_t100 = df_anime_nomovie[df_anime_nomovie['popularity'] <= 100]

########################################################
#Functions to use
########################################################
# Recc all sim, but for top 100
def recommend_all_sim_t100(show_title, n_recom, vectorized_bag_of_words, title, df):
    show_indices = np.where(title == show_title)[0]
    if len(show_indices) == 0:
        print(f"Show title '{show_title}' not found in the DataFrame.")
        return []  # Return an empty list if the show title is not found

    similarity_matrix = cosine_similarity(vectorized_bag_of_words, vectorized_bag_of_words[list(np.where(title == show_title)[0]), :])
    similarity_dataframe = pd.DataFrame(similarity_matrix)
    similarity_dataframe.index = title

    similarity_dataframe = similarity_dataframe.iloc[:, [0]]

    # Calculate the popularity weights
    max_popularity = df['popularity_normalized'].max()
    
    popularity_weights = (1 - df['popularity_normalized']) / max_popularity

    #print(popularity_weights)

    # Multiply the similarity scores by popularity weights
    #similarity_dataframe *= popularity_weights

    similarity_dataframe = similarity_dataframe.sort_values(by = 0, ascending=False)
    #similarity_dataframe = similarity_dataframe.drop_duplicates()

    # Find similar titles using fuzzy matching
    similar_titles = []
    for title in similarity_dataframe.index:
        if fuzz.partial_ratio(show_title, title) >= 0:  # Set a threshold for similarity
            if show_title not in title:
                similar_titles.append(title)

    return similar_titles[:n_recom]
# Considers ALL titles similar to input + popularity
def recommend_all_sim(show_title, n_recom, vectorized_bag_of_words, title, df):
    show_indices = np.where(title == show_title)[0]
    if len(show_indices) == 0:
        print(f"Show title '{show_title}' not found in the DataFrame.")
        return []  # Return an empty list if the show title is not found

    similarity_matrix = cosine_similarity(vectorized_bag_of_words, vectorized_bag_of_words[list(np.where(title == show_title)[0]), :])
    similarity_dataframe = pd.DataFrame(similarity_matrix)
    similarity_dataframe.index = title

    similarity_dataframe = similarity_dataframe.iloc[:, [0]]

    # Calculate the popularity weights
    max_popularity = df['popularity_normalized'].max()
    
    popularity_weights = (1 - df['popularity_normalized']) / max_popularity

    #print(popularity_weights)

    # Multiply the similarity scores by popularity weights
    similarity_dataframe *= popularity_weights

    similarity_dataframe = similarity_dataframe.sort_values(by = 0, ascending=False)
    similarity_dataframe = similarity_dataframe.drop_duplicates()

    # Find similar titles using fuzzy matching
    similar_titles = []
    for title in similarity_dataframe.index:
        if fuzz.partial_ratio(show_title, title) >= 30:  # Set a threshold for similarity
            if show_title not in title:
                similar_titles.append(title)

    return similar_titles[:n_recom]
# Vectorizer of genre and words
def vectorize_genre_and_words(genre_column, synopsis, genre_weight=2.0, words_weight=1.0):
    #extracting keywords for recommender
    rake = Rake()
    words = []
    for plot in synopsis:
        rake.extract_keywords_from_text(str(plot))
        keywords_i = rake.get_ranked_phrases()
        keywords_i_string = ""
        for keyword in keywords_i:
            keywords_i_string = keywords_i_string + " " + keyword
        words.append(keywords_i_string)
    #temp_df['words'] = words
    # Ended up not needing above
    
    #Adjust weight as needed
    # Combine genre and words into a single column
    #combined_column = genre_column + words
    # This ^ ultimately ended up not being used, will leave in for legacy

    # Create a TF-IDF vectorizer
    vectorizer_words = TfidfVectorizer()
    vectorized_words = vectorizer_words.fit_transform(words)
                                                    #try temp_df['words'] if words dont work

    # Create a TF-IDF vectorizer
    vectorizer_genre = TfidfVectorizer()
    vectorized_genre = vectorizer_genre.fit_transform(genre_column)

    # Get the number of genres
    num_genre_features = vectorized_genre.shape[1]

    # Apply the weights to the genre and words vectors
    weighted_vectorized_genre = vectorized_genre.multiply(genre_weight)
    weighted_vectorized_words = vectorized_words.multiply(words_weight)

    # Combine the genre and words vectors
    vectorized_combined = hstack([weighted_vectorized_genre, weighted_vectorized_words])

    # Convert to csr_matrix
    vectorized_combined = csr_matrix(vectorized_combined)

    # Convert to array
    vectorized_combined = vectorized_combined.toarray()

    return vectorized_combined


#Process before running rec:
nomovie_title = df_anime_nomovie['title']
vectorized_no_movie = vectorize_genre_and_words(df_anime_nomovie['genre'], df_anime_nomovie['synopsis_processed'], genre_weight=1.5, words_weight=0.9)

##################################################
# Show Search before using (need EXACT title)
##################################################
#take input
title = st.text_input('Type in show title')
#display
st.write("The show you're searching for is: ", title)

# Filter the DataFrame based on partial match
filtered_shows = df_anime_all[df_anime_all['title'].str.contains(title, case=False)]
filtered_shows = filtered_shows.sort_values(by='popularity', ascending=True)

# Display the filtered shows with 'title' and 'popularity' columns
st.write("Filtered Shows:")
st.write(filtered_shows[['title', 'uid','popularity','episodes']])

###################################
#Content Recc 1 input and use below
#This is the more volatile recc one
###################################
content_recc_input = st.text_input("For content recc example 1, copy and paste in from above which show you'd like a recommendation on based on: ")
st.write("The list below is our recommendation for the show you searched for: ", content_recc_input)
st.write(recommend_all_sim(content_recc_input,20, vectorized_no_movie, nomovie_title, df_anime_nomovie))



###################################
#Content Recc 2 input and use below
#This is the top 100 use only
###################################
#Process before running rec:
t100_title = df_anime_t100['title']
vectorized_t1000 = vectorize_genre_and_words(df_anime_t100['genre'], df_anime_t100['synopsis_processed'], genre_weight=1.5, words_weight=0.9)


content_recc_input_2 = st.text_input("For content recc example 2, copy and paste in from above which show you'd like a recommendation on based on: ")
st.write("The list below is our recommendation for the show you searched for: ", content_recc_input_2)
st.write(recommend_all_sim_t100(content_recc_input_2 ,20, vectorized_t1000, t100_title, df_anime_t100))

#st.write(recommend_all_sim("Fullmetal Alchemist",10, vectorized_no_movie, nomovie_title, df_anime_nomovie))
