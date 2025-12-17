import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(
        self,
        movies_path,
        genres_path,
        movies_genre_path,
        directors_path,
        movies_director_path,
        actors_path,
        movies_cast_path,
        keywords_path,
        collection_path=None,
        cache_dir="./cache_recommender/cache_cb"
    ):
        self.movies_path = movies_path
        self.genres_path = genres_path
        self.movies_genre_path = movies_genre_path
        self.directors_path = directors_path
        self.movies_director_path = movies_director_path
        self.actors_path = actors_path
        self.movies_cast_path = movies_cast_path
        self.keywords_path = keywords_path
        self.collection_path = collection_path

        self.cache_dir = cache_dir
        self.cache_movies = f"{cache_dir}/movies_cache.csv"
        self.cache_sim = f"{cache_dir}/similarity_matrix.npy"
        
        # Tạo thư mục cache
        os.makedirs(cache_dir, exist_ok=True)

        self.movies = None
        self.similarity_matrix = None

    def _sanitize(self, x):
        """Hàm làm sạch text cơ bản"""
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        return ""

    # ===============================
    # 1. LOAD & MERGE
    # ===============================
    def load_data(self):
        # Ưu tiên load từ Cache
        if os.path.exists(self.cache_movies):
            print("✅ [Content-Based] Loading metadata from cache...")
            self.movies = pd.read_csv(self.cache_movies)
            
            # --- FIX LỖI POSTER_PATH NẾU BỊ TÁCH ---
            if 'poster_path' not in self.movies.columns:
                if 'poster_path_x' in self.movies.columns:
                    self.movies['poster_path'] = self.movies['poster_path_x']
                    if 'poster_path_y' in self.movies.columns:
                         self.movies['poster_path'] = self.movies['poster_path'].fillna(self.movies['poster_path_y'])
                elif 'poster_path_y' in self.movies.columns:
                    self.movies['poster_path'] = self.movies['poster_path_y']
                else:
                    self.movies['poster_path'] = ""
            return
        
        print("⚙️ [Content-Based] Building metadata from source CSVs...")
        
        movies = pd.read_csv(self.movies_path)
        
        # Load các bảng phụ
        genres = pd.read_csv(self.genres_path)
        movies_genre = pd.read_csv(self.movies_genre_path)
        directors = pd.read_csv(self.directors_path)
        movies_director = pd.read_csv(self.movies_director_path)
        actors = pd.read_csv(self.actors_path)
        movies_cast = pd.read_csv(self.movies_cast_path)
        keywords = pd.read_csv(self.keywords_path)

        # --- A. MERGE GENRES ---
        mg = movies_genre.merge(genres, on="genre_id")
        genre_group = mg.groupby("movie_id")["name"].apply(list).reset_index()
        genre_group["genres_str"] = genre_group["name"].apply(lambda x: " ".join(self._sanitize(x)))
        movies = movies.merge(genre_group[["movie_id", "genres_str"]], on="movie_id", how="left")

        # --- B. MERGE DIRECTORS ---
        md = movies_director.merge(directors, on="director_id")
        director_group = md.groupby("movie_id")["name"].apply(list).reset_index()
        director_group["director_str"] = director_group["name"].apply(lambda x: " ".join(self._sanitize(x)))
        movies = movies.merge(director_group[["movie_id", "director_str"]], on="movie_id", how="left")

        # --- C. MERGE ACTORS ---
        cast_full = movies_cast.merge(actors[["actor_id", "name"]], on="actor_id", how="left")
        actor_group = cast_full.groupby("movie_id")["name"].apply(list).reset_index()
        actor_group["actors_str"] = actor_group["name"].apply(lambda x: " ".join(self._sanitize(x)))
        movies = movies.merge(actor_group[["movie_id", "actors_str"]], on="movie_id", how="left")

        # --- D. MERGE KEYWORDS ---
        keyword_group = keywords.groupby("movie_id")["name"].apply(list).reset_index()
        keyword_group["keywords_str"] = keyword_group["name"].apply(lambda x: " ".join(self._sanitize(x)))
        movies = movies.merge(keyword_group[["movie_id", "keywords_str"]], on="movie_id", how="left")
        
        # --- E. MERGE COLLECTION (ĐÃ FIX LỖI KEYERROR) ---
        if self.collection_path and os.path.exists(self.collection_path):
            collection_df = pd.read_csv(self.collection_path)
            
            # Kiểm tra xem file movies có collection_id không
            if 'collection_id' in movies.columns:
                # [FIX]: Đổi tên cột 'name' thành 'collection_name' TRƯỚC khi merge để tránh nhầm lẫn
                coll_subset = collection_df[['collection_id', 'name']].rename(columns={'name': 'collection_name'})
                
                # Merge an toàn
                movies = movies.merge(coll_subset, on="collection_id", how="left")
                
                # Xử lý text
                movies["collection_str"] = movies["collection_name"].fillna("").apply(self._sanitize)
            else:
                movies["collection_str"] = ""
        else:
             movies["collection_str"] = ""

        # --- F. TẠO "SOUP" ---
        cols_to_fill = ["overview", "tagline", "genres_str", "director_str", "actors_str", "keywords_str", "collection_str"]
        for col in cols_to_fill:
            if col in movies.columns:
                movies[col] = movies[col].fillna("")
            else:
                movies[col] = ""

        # Công thức trọng số (Collection * 6 để ưu tiên phim bộ)
        movies["content"] = (
            movies["overview"] + " " +
            movies["genres_str"] * 2 + " " +
            movies["collection_str"] * 6 + " " + 
            movies["director_str"] * 2 + " " +
            movies["actors_str"] + " " +
            movies["keywords_str"]
        )

        movies.to_csv(self.cache_movies, index=False)
        print("✅ [Content-Based] Metadata built & cached (Fixed Collection).")
        self.movies = movies.reset_index(drop=True)

    # ===============================
    # 2. VECTORIZE
    # ===============================
    def vectorize(self):
        if os.path.exists(self.cache_sim):
            print("✅ [Content-Based] Loading similarity matrix from cache...")
            self.similarity_matrix = np.load(self.cache_sim)
            return

        print("⚙️ [Content-Based] Computing TF-IDF & Similarity Matrix...")
        
        tfidf = TfidfVectorizer(
            max_features=10000, 
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        tfidf_matrix = tfidf.fit_transform(self.movies["content"])

        sim = cosine_similarity(tfidf_matrix).astype(np.float32)
        
        self.similarity_matrix = sim
        np.save(self.cache_sim, sim)
        print("✅ [Content-Based] Similarity matrix computed & saved.")

    # ===============================
    # 3. SEARCH (Nâng cấp)
    # ===============================
    def search_movies(self, keyword, limit=10):
        if self.movies is None: return pd.DataFrame()
        
        mask_title = self.movies['title'].str.contains(keyword, case=False, na=False)
        
        keyword_sanitized = keyword.lower().replace(" ", "")
        mask_meta = (
            self.movies['actors_str'].str.contains(keyword_sanitized, case=False, na=False) |
            self.movies['director_str'].str.contains(keyword_sanitized, case=False, na=False) |
            self.movies['genres_str'].str.contains(keyword_sanitized, case=False, na=False)
        )
        
        results = self.movies[mask_title | mask_meta].copy()
        
        results['priority'] = 0
        results.loc[mask_title, 'priority'] = 10
        results.loc[mask_meta & ~mask_title, 'priority'] = 5
        
        results = results.sort_values(by=['priority', 'vote_count'], ascending=[False, False])
        
        cols = ["movie_id", "title", "vote_count", "vote_average"]
        if "poster_path" in self.movies.columns: cols.append("poster_path")

        return results.head(limit)[cols]

    # ===============================
    # 4. RECOMMEND
    # ===============================
    def recommend_by_movie(self, movie_id, top_k=10):
        if self.movies is None or self.similarity_matrix is None: return pd.DataFrame()
        if movie_id not in self.movies["movie_id"].values: return pd.DataFrame()
    
        try:
            idx = self.movies.index[self.movies["movie_id"] == movie_id][0]
        except IndexError: return pd.DataFrame()

        scores = self.similarity_matrix[idx]
        top_indices = scores.argsort()[::-1][1 : top_k + 1]
        
        cols = ["movie_id", "title", "vote_count", "vote_average"]
        if "poster_path" in self.movies.columns: cols.append("poster_path")

        return self.movies.iloc[top_indices][cols]

    def recommend_by_title(self, title, top_k=10):
        match = self.movies[self.movies["title"].str.contains(title, case=False, na=False)]
        if match.empty: return pd.DataFrame()
        first_id = match.iloc[0]["movie_id"]
        return self.recommend_by_movie(first_id, top_k)