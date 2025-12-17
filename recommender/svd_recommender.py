import pandas as pd
import datetime
import os
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class SVDRecommender:
    """
    User-based Collaborative Filtering using SVD (Matrix Factorization)
    + Logic: Retrain định kỳ (7 ngày/lần) + Boost Collection
    """

    def __init__(self, ratings_path, movies_path, cache_dir="./cache_recommender/cache_svd", collection_boost=0.3):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.cache_dir = cache_dir
        self.collection_boost = collection_boost

        # Tạo đường dẫn cache
        self.model_path = os.path.join(self.cache_dir, 'svd_model.pkl')
        self.timestamp_path = os.path.join(self.cache_dir, 'last_train_time.txt')

        self.ratings = None
        self.movies = None
        self.model = None

    # ===============================
    # 1. Load data (CSV)
    # ===============================
    def load_data(self):
        if not os.path.exists(self.movies_path) or not os.path.exists(self.ratings_path):
             raise FileNotFoundError("❌ Không tìm thấy file CSV ratings hoặc movies!")

        self.ratings = pd.read_csv(self.ratings_path)
        self.movies = pd.read_csv(self.movies_path)

        # Xử lý collection_id để tránh lỗi NaN
        if "collection_id" in self.movies.columns:
            self.movies["collection_id"] = (
                self.movies["collection_id"]
                .fillna(-1)
                .astype(int)
            )
        print("✅ Data Loaded (CSV).")

    # ===============================
    # 2. Core SVD Training
    # ===============================
    def train(self, n_factors=50, n_epochs=20):
        print("⏳ Starting SVD Training (This may take a while)...")
        
        # Lưu ý: Kiểm tra file rating của bạn max là bao nhiêu để set rating_scale
        reader = Reader(rating_scale=(0, 5)) 

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
            
        print(f"💾 SVD Cache saved (Model + Timestamp) to: {self.cache_dir}")

    def load_cache(self):
        """Chỉ load model lên RAM, không quan tâm ngày tháng"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
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
    # 5. Recommendation Logic
    # ===============================
    def recommend_for_user(self, user_id, top_k=10):
        # Kiểm tra model
        if self.model is None:
            return "❌ Model chưa sẵn sàng (chưa train/load)"

        # Lấy danh sách phim user đã xem
        user_watched = self.ratings[self.ratings["user_id"] == user_id]["movie_id"].tolist()
        
        # Lọc ra danh sách ứng viên (Candidate): Tất cả phim TRỪ phim đã xem
        candidates = self.movies[~self.movies["movie_id"].isin(user_watched)].copy()

        # Tìm các collection mà user thích (để boost điểm)
        # Logic: Nếu user đã xem phim A thuộc collection X, ta sẽ ưu tiên phim B cũng thuộc collection X
        watched_meta = self.movies[self.movies["movie_id"].isin(user_watched)]
        liked_collections = set(watched_meta["collection_id"].dropna().unique())
        if -1 in liked_collections: liked_collections.remove(-1) # Bỏ collection rác

        predictions = []

        # Predict điểm cho từng ứng viên
        # (Surprise predict rất nhanh nên loop này chạy ổn với <100k items)
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
        
        # Mẹo: Sort lại result theo đúng thứ tự điểm số (vì lệnh isin làm mất thứ tự)
        result["temp_score"] = result["movie_id"].apply(lambda x: dict(top_movies)[x])
        result = result.sort_values("temp_score", ascending=False).drop(columns="temp_score")
        
        return result[["movie_id", "title", "collection_id"]]