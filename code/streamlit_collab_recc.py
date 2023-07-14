import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD


#files to use
df_main = pd.read_csv("../data/df_prof_rev_anime_clean.csv")
df_anime = pd.read_csv("../data/df_anime_export.csv")

df_touse = df_main[["profile", "uid", "review_score"]]
df_anime_touse = df_anime[["uid", "title"]]

##################################################
# Show Search before using (need EXACT title)
##################################################
#take input
title = st.text_input('Type in show title')
#display
st.write("The show you're searching for is: ", title)

# Filter the DataFrame based on partial match
filtered_shows = df_anime[df_anime['title'].str.contains(title, case=False)]
filtered_shows = filtered_shows.sort_values(by='popularity', ascending=True)

# Display the filtered shows with 'title' and 'popularity' columns
st.write("Filtered Shows:")
st.write(filtered_shows[['title', 'uid','popularity','episodes']])

###############################
#Personal Ratings
###############################
new_rows = [
    {'profile': 'DargeLildo', 'uid': '2,001', 'review_score': '9'},  #Gurren Lagann
    {'profile': 'DargeLildo', 'uid': '30', 'review_score': '10'},   #NGEv
    {'profile': 'DargeLildo', 'uid': '11,061', 'review_score': '10'},    #hxh
    {'profile': 'DargeLildo', 'uid': '226', 'review_score': '1'},  #elfen
    {'profile': 'DargeLildo', 'uid': '1,535', 'review_score': '7'},  #deathnote
    {'profile': 'DargeLildo', 'uid': '918', 'review_score': '5'},    #gintama
    {'profile': 'DargeLildo', 'uid': '28,977', 'review_score': '5'}    #gintama*
]

df_touse = pd.concat([df_touse, pd.DataFrame(new_rows)], ignore_index=True)

#############################
#begin collab recc
#############################
df = Dataset.load_from_df(df_touse, Reader(rating_scale=(1, 10)))
trainset = df.build_full_trainset()
model = SVD()
model.fit(trainset)

# Take user input
user_type = st.radio("Are you a new user or a returning user?", ("New User", "Returning User"))
st.write("IN THE NEW USER, A RETURNING USER MAY ALSO UES THIS OPTION TO PUT IN A NEW REVIEW AND GET RECOMMENDATIONS BASED ON THAT")

if user_type == "New User":
    # New user 
    user_id = st.text_input("Enter your user ID")
    show_id = st.text_input("Enter the show ID")
    rating = st.slider("Rate the show (1-10)", 1, 10)
    
    # Update the training dataset with new user stuff
    new_row = pd.DataFrame({'profile': [user_id], 'uid': [show_id], 'review_score': [rating]})
    df_touse_updated = pd.concat([df_touse, new_row], ignore_index=True)

    # Update the model with the updated user info
    df_updated = Dataset.load_from_df(df_touse_updated, Reader(rating_scale=(1, 10)))
    trainset_updated = df_updated.build_full_trainset()
    model_updated = SVD()
    model_updated.fit(trainset_updated)

    # Predict the rating for the show
    predicted_rating = model_updated.predict(user_id, show_id, r_ui=rating)

    # Get the top rated shows for the user
    all_anime = df_touse_updated['uid'].unique()
    watched = df_touse_updated[df_touse_updated['profile'] == user_id].uid
    not_watched = [anime for anime in all_anime if anime not in watched]

    score = [model_updated.predict(user_id, anime_id).est for anime_id in not_watched]
    df_pred = pd.DataFrame({'uid': not_watched, 'pred_score': score})
    df_pred_real = df_pred.sort_values('pred_score', ascending=False).head(10)
    recommended_shows = df_pred_real.merge(df_anime_touse, how='left', on='uid')

    # Display the predicted rating and recommended shows
    #st.write("Predicted Rating:")
    #st.write(predicted_rating.est)
    st.write("Recommended Shows:")
    st.write(recommended_shows[['title', 'pred_score']])

elif user_type == "Returning User":
    # Returning user scenario
    user_id = st.text_input("Enter your user ID")

    if user_id not in df_touse['profile'].unique():
        st.error("User ID does not exist. Please enter a valid user ID.")
    else:
        # Get the top rated shows for the user
        all_anime = df_touse['uid'].unique()
        watched = df_touse[df_touse['profile'] == user_id].uid
        not_watched = [anime for anime in all_anime if anime not in watched]

        score = [model.predict(user_id, anime_id).est for anime_id in not_watched]
        df_pred = pd.DataFrame({'uid': not_watched, 'pred_score': score})
        df_pred_real = df_pred.sort_values('pred_score', ascending=False).head(10)
        recommended_shows = df_pred_real.merge(df_anime_touse, how='left', on='uid')

        # Display the recommended shows
        st.write("Recommended Shows:")
        st.write(recommended_shows[['title', 'pred_score', 'uid']])