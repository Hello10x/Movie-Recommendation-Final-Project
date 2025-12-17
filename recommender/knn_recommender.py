import pandas as pd
import numpy as np
import os
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class KNNRecommender:
    """
    Item-based CF using Scipy Sparse Matrix
    FIX: Xử lý trùng lặp đánh giá (Dedup logic)
    """

    def __init__(self, ratings_path, movies_path, cache_dir="./cache_recommender/cache_knn"):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.cache_dir = cache_dir

        self.model_path = os.path.join(cache_dir, 'knn_model.pkl')
        self.matrix_path = os.path.join(cache_dir, 'sparse_matrix.pkl')
        self.mappings_path = os.path.join(cache_dir, 'mappings.pkl')

        self.ratings = None
        self.movies = None
        
        self.sparse_matrix = None
        self.knn_model = None
        
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.user_to_idx = {}
        self.idx_to_user = {}

    def load_data(self):
        if not os.path.exists(self.movies_path) or not os.path.exists(self.ratings_path):
             raise FileNotFoundError("❌ Data files not found!")
        
        self.movies = pd.read_csv(self.movies_path)
        
        # --- FIX 1: LOAD DỮ LIỆU THÔNG MINH ---
        print("⚙️ Loading and Deduplicating Ratings...")
        raw_ratings = pd.read_csv(self.ratings_path)
        
        # Sắp xếp theo timestamp giảm dần (cái mới nhất lên đầu)
        # Giả sử file có cột 'timestamp', nếu không có thì nó lấy dòng cuối cùng (mặc định của drop_duplicates keep='last')
        if 'timestamp' in raw_ratings.columns:
            raw_ratings = raw_ratings.sort_values('timestamp', ascending=False)
            
        # Xóa các dòng trùng (User + Movie), chỉ giữ lại dòng đầu tiên (là dòng mới nhất do đã sort)
        self.ratings = raw_ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='first')
        
        print(f"✅ Ratings loaded. Raw: {len(raw_ratings)} -> Clean: {len(self.ratings)} (Removed {len(raw_ratings) - len(self.ratings)} old duplicates)")

    def build_matrix(self):
        print("⚙️ Building Sparse Matrix...")
        
        # Tạo Mapping
        unique_movies = self.ratings['movie_id'].unique()
        unique_users = self.ratings['user_id'].unique()

        self.movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self.idx_to_movie = {i: mid for i, mid in enumerate(unique_movies)}
        
        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for i, uid in enumerate(unique_users)}

        # Map dữ liệu
        # LƯU Ý: self.ratings lúc này đã sạch (không trùng), nên tạo matrix sẽ không bị cộng dồn
        row_users = self.ratings['user_id'].map(self.user_to_idx).values
        col_movies = self.ratings['movie_id'].map(self.movie_to_idx).values
        data_ratings = self.ratings['rating_norm'].values

        self.sparse_matrix = csr_matrix(
            (data_ratings, (col_movies, row_users)), 
            shape=(len(unique_movies), len(unique_users))
        )
        print(f"✅ Sparse Matrix Created. Shape: {self.sparse_matrix.shape}")

    def train(self, k=20):
        print("⏳ Training KNN...")
        self.knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=-1)
        self.knn_model.fit(self.sparse_matrix)
        print("✅ KNN Training Completed.")

    def save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.model_path, "wb") as f: pickle.dump(self.knn_model, f)
        with open(self.matrix_path, "wb") as f: pickle.dump(self.sparse_matrix, f)
        with open(self.mappings_path, "wb") as f: 
            pickle.dump({
                'm2i': self.movie_to_idx, 'i2m': self.idx_to_movie,
                'u2i': self.user_to_idx, 'i2u': self.idx_to_user
            }, f)
        print("💾 KNN Cache Saved.")

    def load_cache(self):
        if os.path.exists(self.model_path) and os.path.exists(self.matrix_path):
            try:
                with open(self.model_path, "rb") as f: self.knn_model = pickle.load(f)
                with open(self.matrix_path, "rb") as f: self.sparse_matrix = pickle.load(f)
                with open(self.mappings_path, "rb") as f:
                    maps = pickle.load(f)
                    self.movie_to_idx = maps['m2i']
                    self.idx_to_movie = maps['i2m']
                    self.user_to_idx = maps['u2i']
                    self.idx_to_user = maps['i2u']
                
                # Load luôn ratings sạch vào RAM để phục vụ update realtime
                # (Nếu không load lại thì update sẽ bị thiếu dữ liệu cũ)
                self.load_data() 
                
                print("✅ KNN Cache Loaded.")
                return True
            except Exception as e:
                print(f"⚠️ Cache Error: {e}")
                return False
        return False

    def recommend_by_movie(self, movie_id, top_k=10):
        if self.sparse_matrix is None: return None
        if movie_id not in self.movie_to_idx: return None
            
        movie_idx = self.movie_to_idx[movie_id]
        dists, indices = self.knn_model.kneighbors(self.sparse_matrix[movie_idx], n_neighbors=top_k+1)
        
        similar_ids = [self.idx_to_movie[i] for i in indices.flatten() if self.idx_to_movie[i] != movie_id]
        return self.movies[self.movies["movie_id"].isin(similar_ids)][["movie_id", "title"]].head(top_k)

    # ===============================
    # FIX 2: UPDATE REALTIME THÔNG MINH
    # ===============================
    def update_model_realtime(self, user_id, movie_id, rating_norm):
        print(f"⚡ Updating KNN (Smart): U={user_id}, M={movie_id}, Score={rating_norm}")
        
        # 1. Kiểm tra xem User này đã từng chấm phim này chưa
        # Tạo mask (bộ lọc)
        mask = (self.ratings['user_id'] == user_id) & (self.ratings['movie_id'] == movie_id)
        
        if self.ratings[mask].empty:
            # Case A: Chưa từng chấm -> Thêm dòng mới
            new_row = pd.DataFrame([{
                "user_id": user_id, "movie_id": movie_id, "rating_norm": rating_norm, "timestamp": 0
            }]) # Timestamp ko quan trọng vì mình vừa thêm vào cuối
            self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)
            print("   -> Inserted new rating.")
        else:
            # Case B: Đã chấm rồi -> CẬP NHẬT giá trị dòng cũ
            self.ratings.loc[mask, 'rating_norm'] = rating_norm
            print("   -> Overwrote existing rating.")

        # 2. Rebuild Matrix (Giờ đây self.ratings đã sạch, không bị trùng)
        self.build_matrix()
        self.knn_model.fit(self.sparse_matrix)
        self.save_cache()