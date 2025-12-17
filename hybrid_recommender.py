import pandas as pd
import numpy as np
import os
import time
from collections import Counter

# Import các class con (giả sử bạn để trong folder recommender)
from recommender.svd_recommender import SVDRecommender
from recommender.contentbase_recommender import ContentBasedRecommender
from recommender.knn_recommender import KNNRecommender

class HybridRecommender:
    def __init__(self, ratings_df, data_dir="./data_clean", cache_dir="./cache_recommender"):
        print("🚀 Initializing Hybrid System (Final - Fixed Collection Logic)...")
        
        self.data_dir = data_dir
        # Định nghĩa lại ratings_path để dùng cho việc lưu file khi rate
        self.ratings_path = f"{data_dir}/ratings_clean.csv" 
        
        self.movies_path = f"{data_dir}/movies_clean.csv"
        self.users_path = f"{data_dir}/users_clean.csv"
        self.collection_path = f"{data_dir}/collection_clean.csv"
        
        # --- NHẬN DATAFRAME TỪ BÊN NGOÀI ---
        self.ratings = ratings_df 
        
        # File Log hành vi
        self.search_logs_path = f"{data_dir}/search_history.csv"
        self.view_logs_path = f"{data_dir}/view_history.csv"
        
        self._init_log_files()

        # --- KHỞI TẠO 3 MODEL CON ---
        
        # 1. Content Based
        self.cb = ContentBasedRecommender(
            movies_path=self.movies_path,
            genres_path=f"{data_dir}/genres_clean.csv",
            movies_genre_path=f"{data_dir}/movies_genre_clean.csv",
            directors_path=f"{data_dir}/directors_clean.csv",
            movies_director_path=f"{data_dir}/movies_director_clean.csv",
            actors_path=f"{data_dir}/actors_clean.csv",
            movies_cast_path=f"{data_dir}/movies_cast_clean.csv",
            keywords_path=f"{data_dir}/keywords_clean.csv",
            collection_path=self.collection_path,
            cache_dir=f"{cache_dir}/cache_cb"
        )

        # 2. KNN Recommender
        self.knn = KNNRecommender(
            ratings_df=self.ratings,   # Truyền DF
            movies_path=self.movies_path,
            cache_dir=f"{cache_dir}/cache_knn"
        )

        # 3. SVD Recommender
        self.svd = SVDRecommender(
            ratings_df=self.ratings,   # Truyền DF
            movies_path=self.movies_path,
            cache_dir=f"{cache_dir}/cache_svd"
        )

        self._load_all_models()

    def _init_log_files(self):
        # Tạo file log rỗng nếu chưa có
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.users_path):
            pd.DataFrame(columns=["user_id", "username"]).to_csv(self.users_path, index=False)
        if not os.path.exists(self.search_logs_path):
            pd.DataFrame(columns=["user_id", "query", "type", "timestamp"]).to_csv(self.search_logs_path, index=False)
        if not os.path.exists(self.view_logs_path):
            pd.DataFrame(columns=["user_id", "movie_id", "timestamp"]).to_csv(self.view_logs_path, index=False)

    def _load_all_models(self):
        print("--- Loading Sub-models ---")
        # Content Based
        self.cb.load_data()
        self.cb.vectorize()
        
        # KNN
        self.knn.load_data()
        if not self.knn.load_cache():
            self.knn.build_matrix()
            self.knn.train()
            self.knn.save_cache()
            
        # SVD
        self.svd.load_data()
        self.svd.check_and_train()
        print("✅ Hybrid System Ready!\n")

    # ====================================================
    # 1. LOGGING
    # ====================================================
    def log_search(self, user_id, query, search_type='keyword'):
        new_row = {"user_id": user_id, "query": query, "type": search_type, "timestamp": int(time.time())}
        pd.DataFrame([new_row]).to_csv(self.search_logs_path, mode='a', header=False, index=False)
        print(f"📝 Logged: User {user_id} -> {search_type}: '{query}'")

    def log_view(self, user_id, movie_id):
        new_row = {"user_id": user_id, "movie_id": movie_id, "timestamp": int(time.time())}
        pd.DataFrame([new_row]).to_csv(self.view_logs_path, mode='a', header=False, index=False)
        print(f"👁️ Logged View: User {user_id} -> Movie {movie_id}")

    # ====================================================
    # 2. SEARCH
    # ====================================================
    def search_hybrid(self, keyword, top_k=20):
        return self.cb.search_movies(keyword, limit=top_k)

    def get_movies_of_person(self, name, role='actor', top_k=50):
        name_sanitized = str(name).lower().replace(" ", "")
        column = 'actors_str' if role == 'actor' else 'director_str'
        if column in self.cb.movies.columns:
            mask = self.cb.movies[column].str.contains(name_sanitized, case=False, na=False)
            return self.cb.movies[mask].sort_values('vote_count', ascending=False).head(top_k)
        return pd.DataFrame()

    # ====================================================
    # 3. ANALYZE & RECOMMEND
    # ====================================================
    def analyze_user_interest(self, user_id):
        interests = {"genres": [], "keywords": [], "actors": [], "directors": []}
        try:
            if os.path.exists(self.view_logs_path):
                df_view = pd.read_csv(self.view_logs_path)
                if 'timestamp' in df_view.columns and not df_view.empty:
                    user_views = df_view[df_view['user_id'] == user_id].sort_values('timestamp', ascending=False).head(20)
                    if not user_views.empty:
                        viewed_ids = user_views['movie_id'].tolist()
                        viewed_meta = self.cb.movies[self.cb.movies['movie_id'].isin(viewed_ids)]
                        
                        def get_top_attr(col_name):
                            items = []
                            for s in viewed_meta[col_name].dropna(): items.extend(s.split())
                            return [x[0] for x in Counter(items).most_common(3)]

                        interests['genres'] = get_top_attr('genres_str')
                        interests['actors'] = get_top_attr('actors_str')
                        interests['directors'] = get_top_attr('director_str')

            if os.path.exists(self.search_logs_path):
                df_search = pd.read_csv(self.search_logs_path)
                if 'timestamp' in df_search.columns and not df_search.empty:
                    user_logs = df_search[df_search['user_id'] == user_id].sort_values('timestamp', ascending=False).head(10)
                    if not user_logs.empty:
                        if 'type' not in user_logs.columns: user_logs['type'] = 'keyword'
                        for _, row in user_logs.iterrows():
                            q = str(row['query'])
                            t = row['type']
                            if t == 'genre': interests['genres'].insert(0, q.lower().replace(" ", ""))
                            elif t == 'actor': interests['actors'].insert(0, q.lower().replace(" ", ""))
                            elif t == 'director': interests['directors'].insert(0, q.lower().replace(" ", ""))
                            else: interests['keywords'].append(q)
        except Exception as e:
            print(f"⚠️ Error analyzing interest: {e}")
        return interests

    def recommend_based_on_behavior(self, user_id, top_k=12):
        print(f"🧠 Generating Behavior-Based Recs for User {user_id}...")
        
        profile = self.analyze_user_interest(user_id)
        candidates = {} 
        
        for g in list(set(profile['genres'])):
            mask = self.cb.movies['genres_str'].str.contains(g, na=False)
            for _, row in self.cb.movies[mask].sort_values('vote_count', ascending=False).head(10).iterrows():
                candidates[row['movie_id']] = candidates.get(row['movie_id'], 0) + 2.5

        for p in list(set(profile['actors'])):
            mask = self.cb.movies['actors_str'].str.contains(p, na=False)
            for mid in self.cb.movies[mask].head(10)['movie_id']: candidates[mid] = candidates.get(mid, 0) + 2.0
            
        for d in list(set(profile['directors'])):
            mask = self.cb.movies['director_str'].str.contains(d, na=False)
            for mid in self.cb.movies[mask].head(10)['movie_id']: candidates[mid] = candidates.get(mid, 0) + 2.0

        for kw in list(set(profile['keywords'])):
            res = self.cb.search_movies(kw, limit=5)
            for mid in res['movie_id']: candidates[mid] = candidates.get(mid, 0) + 1.5

        # SVD Bonus
        if user_id in self.svd.ratings['user_id'].values:
            svd_recs = self.svd.recommend_for_user(user_id, top_k=20)
            if not svd_recs.empty:
                for _, row in svd_recs.iterrows():
                    candidates[row['movie_id']] = candidates.get(row['movie_id'], 0) + 1.0

        if not candidates:
            return self.cb.movies.sort_values(by=["vote_count", "vote_average"], ascending=[False, False]).head(top_k)

        final_scores = [{"movie_id": k, "score": v} for k, v in candidates.items()]
        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_ids = [x["movie_id"] for x in final_scores[:top_k]]
        return self.cb.movies[self.cb.movies['movie_id'].isin(top_ids)]

    # ====================================================
    # 4. GỢI Ý KHI XEM PHIM (CHI TIẾT)
    # ====================================================
    def get_recommendations_for_viewing(self, user_id, current_movie_id, top_k=10, current_rating=None):
        item_weight = 1.0 
        if current_rating is not None:
            if current_rating <= 2.5: item_weight = 0.1 
            elif current_rating >= 4.0: item_weight = 1.2

        candidates_knn = self._get_knn_candidates(current_movie_id)
        candidates_cb = self._get_cb_candidates(current_movie_id)
        
        # Series Boost
        collection_bonus = {}
        if 'collection_id' in self.cb.movies.columns:
            try:
                curr_row = self.cb.movies[self.cb.movies['movie_id'] == current_movie_id]
                if not curr_row.empty:
                    coll_val = curr_row.iloc[0]['collection_id']
                    if pd.notna(coll_val) and coll_val != 0 and str(coll_val) != 'nan':
                        siblings_mask = self.cb.movies['collection_id'] == coll_val
                        siblings = self.cb.movies[siblings_mask]['movie_id'].tolist()
                        for sib_id in siblings:
                            if sib_id != current_movie_id:
                                collection_bonus[sib_id] = 5.0 
            except Exception as e:
                print(f"⚠️ Error in Series Boost logic: {e}")

        all_ids = set(candidates_knn.keys()) | set(candidates_cb.keys()) | set(collection_bonus.keys())
        if current_movie_id in all_ids: all_ids.remove(current_movie_id)

        final_scores = []
        for mid in all_ids:
            score_knn = candidates_knn.get(mid, 0.0) 
            score_cb = candidates_cb.get(mid, 0.0)
            score_series = collection_bonus.get(mid, 0.0)
            
            try: 
                if user_id in self.svd.ratings['user_id'].values:
                    pred = self.svd.model.predict(user_id, mid).est
                    score_svd = pred / 10.0
                else: score_svd = 0.0
            except: score_svd = 0.0
            
            final_score = (score_knn * 0.4 * item_weight) + \
                          (score_cb * 0.3 * item_weight) + \
                          (score_svd * 0.3) + \
                          score_series 
            
            final_scores.append({"movie_id": mid, "score": final_score})

        final_scores.sort(key=lambda x: x["score"], reverse=True)
        top_ids = [x["movie_id"] for x in final_scores[:top_k]]
        
        return self.cb.movies[self.cb.movies['movie_id'].isin(top_ids)]

    def _get_cb_candidates(self, movie_id, k=30):
        try:
            rec_df = self.cb.recommend_by_movie(movie_id, top_k=k)
            if rec_df is None or rec_df.empty: return {}
            return {row['movie_id']: (1.0 - i*0.01) for i, row in rec_df.iterrows()}
        except: return {}

    def _get_knn_candidates(self, movie_id, k=30):
        if self.knn.sparse_matrix is None or movie_id not in self.knn.movie_to_idx: return {}
        idx = self.knn.movie_to_idx[movie_id]
        dists, indices = self.knn.knn_model.kneighbors(self.knn.sparse_matrix[idx], n_neighbors=k+1)
        res = {}
        for i, d in zip(indices.flatten(), dists.flatten()):
            mid = self.knn.idx_to_movie[i]
            if mid != movie_id: res[mid] = 1.0 - d
        return res

    # ====================================================
    # 5. RATE & UPDATE (ĐÃ SỬA: Cập nhật RAM)
    # ====================================================
    def rate_movie(self, user_id, movie_id, rating_val):
        # 1. Check User File
        try:
            if os.path.exists(self.users_path):
                users_df = pd.read_csv(self.users_path)
                if user_id not in users_df['user_id'].values:
                    pd.DataFrame([{"user_id": user_id, "username": "anonymous"}]).to_csv(self.users_path, mode='a', header=False, index=False)
            else:
                pd.DataFrame([{"user_id": user_id}]).to_csv(self.users_path, index=False)
        except Exception as e: print(f"⚠️ User check error: {e}")

        # 2. Tạo dòng dữ liệu mới
        rating_norm = rating_val * 2.0 
        new_data = {
            "id": int(time.time()), 
            "user_id": user_id, 
            "movie_id": movie_id, 
            "rating": rating_val, 
            "timestamp": int(time.time()), 
            "rating_norm": rating_norm
        }
        
        # 3. Ghi vào FILE CSV (Để lưu trữ lâu dài)
        if os.path.exists(self.ratings_path):
             pd.DataFrame([new_data]).to_csv(self.ratings_path, mode='a', header=False, index=False)
        else:
             print("⚠️ Warning: ratings_path not found, skipped writing to disk.")

        print(f"⭐ Rated: User {user_id} -> Movie {movie_id} ({rating_val}*)")

        # 4. QUAN TRỌNG: Cập nhật vào RAM (Để có tác dụng ngay lập tức)
        # Cập nhật self.ratings của Hybrid
        new_row_df = pd.DataFrame([new_data])
        self.ratings = pd.concat([self.ratings, new_row_df], ignore_index=True)
        
        # Cập nhật cho SVD (Vì SVD tham chiếu đến biến ratings cũ)
        if self.svd:
            self.svd.ratings = self.ratings
            
        # Cập nhật cho KNN (KNN có hàm update riêng để build lại matrix)
        if self.knn:
            self.knn.update_model_realtime(user_id, movie_id, rating_norm)
