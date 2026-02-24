import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = {
    "title": ["Avengers", "Iron Man", "Batman", "Superman", "Spiderman"],
    "description": [
        "superhero team action",
        "technology superhero action",
        "dark hero crime action",
        "alien superhero action",
        "young superhero adventure"
    ]
}

df = pd.DataFrame(movies)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"])
similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_name):
    if movie_name not in df["title"].values:
        print("Movie not found!")
        return

    index = df[df["title"] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]

    print("\nRecommended Movies:")
    print("--------------------")
    for i in scores:
        print(df.iloc[i[0]]["title"])

if __name__ == "__main__":
    name = input("Enter movie name: ")
    recommend(name)
