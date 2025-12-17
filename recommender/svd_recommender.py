import pandas as pd
import datetime
import os
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class SVDRecommender:
    """
    User-based Collaborative Filtering using SVD (Matrix Factorization)
    UPDATED: Nhận DataFrame trực tiếp từ bên ngoài
    """

    # --- SỬA ĐỔI 1: Nhận ratings_df thay vì ratings_path ---
    def __init__(self, ratings_df, movies_path, cache_dir="./cache_recommender/cache_svd", collection_boost=0.3):
        self.ratings = ratings_df  # <--- Gán trực tiếp dữ liệu
        self.movies_path = movies_path
        self.cache_dir = cache_dir
        self.collection_boost = collection_boost

        # Tạo đường dẫn cache
        self.model_path = os.path.join(self.cache_dir, 'svd_model.pkl')
        self.timestamp_path = os.path.join(self.cache_dir, 'last_train_time.txt')

        self.movies = None
        self.model = None

    # ===============================
    # 1. Load data (Chỉ load Movies)
    # ===============================
    def load_data(self):
        # --- SỬA ĐỔI 2: Chỉ load Movies, không load Ratings từ file nữa ---
        if not os.path.exists(self.movies_path):
             raise FileNotFoundError(f"❌ Movies file not found at {self.movies_path}")

        self.movies = pd.read_csv(self.movies_path)

        # Xử lý collection_id để tránh lỗi NaN
        if "collection_id" in self.movies.columns:
            self.movies["collection_id"] = (
                self.movies["collection_id"]
                .fillna(-1)
                .astype(int)
            )
        
        # Kiểm tra dữ liệu ratings
        if self.ratings is None or self.ratings.empty:
            print("⚠️ Warning: Ratings DataFrame is empty inside SVD!")
        else:
            print(f"✅ Data Ready: Movies({len(self.movies)}), Ratings({len(self.ratings)})")

    # ===============================
    # 2. Core SVD Training
    # ===============================
    def train(self, n_factors=50, n_epochs=20):
        print("⏳ Starting SVD Training (This may take a while)...")
        
        # Đảm bảo movies đã load (để logic sau này không lỗi)
        if self.movies is None:
            self.load_data()

        # Lưu ý: Kiểm tra file rating của bạn max là bao nhiêu để set rating_scale
        reader = Reader(rating_scale=(0, 5)) 

        # Load dữ liệu từ DataFrame (self.ratings có sẵn)
        data = Dataset.load_from_df(
            self.ratings[["user_id", "movie_id", "rating_norm"]], # Dùng cột rating_norm
            reader
        )

        # Train trên toàn bộ dữ liệu (để gợi ý tốt nhất)
        trainset = data.build_full_trainset()

        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=0.005,
            reg_all=0.1
        )
        self.model.fit(trainset)
        print("✅ SVD Training Completed.")

    # ===============================
    # 3. Cache Management (Lưu/Load Model)
    # ===============================
    def save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 1. Lưu Model SVD (Dạng binary)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
            
        # 2. Lưu Timestamp (Dạng text ngày tháng)
        with open(self.timestamp_path, "w") as f:
            f.write(str(datetime.date.today()))
            
        print(f"💾 SVD Cache saved to: {self.cache_dir}")

    def load_cache(self):
        """Chỉ load model lên RAM"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                
                # Đừng quên load movies info nếu lấy từ cache
                self.load_data()
                return True
            except Exception as e:
                print(f"⚠️ Load Cache Error: {e}")
                return False
        return False

    # ===============================
    # 4. Smart Train Scheduler (Logic 7 Ngày)
    # ===============================
    def check_and_train(self, force=False):
        """
        Hàm quan trọng nhất: Quyết định xem nên Load Cache hay Train mới
        """
        # Luôn đảm bảo load movies trước
        if self.movies is None:
            self.load_data()

        should_train = False
        reason = ""

        # Case 1: Nếu chưa có file model -> Bắt buộc train
        if not os.path.exists(self.model_path) or not os.path.exists(self.timestamp_path):
            should_train = True
            reason = "Cache not found"
        
        # Case 2: Nếu có file, kiểm tra ngày tháng
        else:
            with open(self.timestamp_path, "r") as f:
                last_date_str = f.read().strip()
            
            try:
                last_date = datetime.date.fromisoformat(last_date_str)
                days_diff = (datetime.date.today() - last_date).days
                
                print(f"📅 Last trained: {last_date} ({days_diff} days ago)")
                
                if days_diff >= 7:
                    should_train = True
                    reason = "Expired (> 7 days)"
            except ValueError:
                should_train = True
                reason = "Invalid Timestamp format"

        # Quyết định cuối cùng
        if force:
            print("💪 Force Training requested.")
            self.train()
            self.save_cache()
        elif should_train:
            print(f"🔄 Retraining SVD... (Reason: {reason})")
            self.train()
            self.save_cache()
        else:
            print("✅ Model is fresh. Loading from cache...")
            if not self.load_cache():
                # Phòng hờ trường hợp file pkl bị lỗi dù timestamp đúng
                print("⚠️ Cache load failed. Retraining fallback...")
                self.train()
                self.save_cache()

    # ===============================
    # 5. Recommendation Logic (Giữ nguyên)
    # ===============================
    def recommend_for_user(self, user_id, top_k=10):
        # Kiểm tra model
        if self.model is None:
            return pd.DataFrame() # Trả về DF rỗng thay vì string để tránh lỗi code bên ngoài

        # Lấy danh sách phim user đã xem
        user_watched = self.ratings[self.ratings["user_id"] == user_id]["movie_id"].tolist()
        
        # Lọc ra danh sách ứng viên (Candidate): Tất cả phim TRỪ phim đã xem
        candidates = self.movies[~self.movies["movie_id"].isin(user_watched)].copy()

        # Tìm các collection mà user thích (để boost điểm)
        watched_meta = self.movies[self.movies["movie_id"].isin(user_watched)]
        liked_collections = set(watched_meta["collection_id"].dropna().unique())
        if -1 in liked_collections: liked_collections.remove(-1) 

        predictions = []

        # Predict điểm cho từng ứng viên
        for _, row in candidates.iterrows():
            movie_id = row["movie_id"]
            col_id = row["collection_id"]
            
            # 1. SVD dự đoán rating gốc (est)
            pred = self.model.predict(uid=user_id, iid=movie_id)
            score = pred.est
            
            # 2. Cộng điểm thưởng nếu cùng Collection
            if col_id in liked_collections:
                score += self.collection_boost
            
            predictions.append((movie_id, score))

        # Sort lấy top cao nhất
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:top_k]

        # Map lại ra tên phim
        top_ids = [x[0] for x in top_movies]
        result = self.movies[self.movies["movie_id"].isin(top_ids)].copy()
        
        # Sort lại result theo đúng thứ tự điểm số
        result["temp_score"] = result["movie_id"].apply(lambda x: dict(top_movies)[x])
        result = result.sort_values("temp_score", ascending=False).drop(columns="temp_score")
        
        return result[["movie_id", "title", "collection_id"]]
