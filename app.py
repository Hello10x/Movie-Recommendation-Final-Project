import streamlit as st
import pandas as pd
import os
import gdown  

from hybrid_recommender import HybridRecommender

st.set_page_config(page_title="Movie Recommendation GK - Ken", page_icon="🍿", layout="wide", initial_sidebar_state="expanded")

# ============================================
# 0. CSS TÙY CHỈNH
# ============================================
st.markdown("""
<style>
    /* Chỉ giữ lại style cho Button và Phân trang */
    .stButton button { 
        width: 100%; 
        border: 1px solid #444; 
    }
    .pagination-cnt { 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        gap: 20px; 
        margin-top: 20px; 
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 1. LOAD DỮ LIỆU TỪ DRIVE
# ============================================
@st.cache_data(show_spinner="Đang tải dữ liệu Rating từ Google Drive...")
def load_ratings_from_drive():
    file_path = 'ratings_clean.csv'  # Tên file sẽ lưu trên server
    
    # Kiểm tra nếu file chưa tồn tại thì mới tải
    if not os.path.exists(file_path):
        # ID file của bạn
        file_id = '1ZLWgsnkcsJ3ktvtit3t-tS3MXaInuMQC'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Tải file về (quiet=False để xem log nếu cần)
        try:
            gdown.download(url, file_path, quiet=False)
        except Exception as e:
            st.error(f"Lỗi tải file từ Drive: {e}")
            return pd.DataFrame() # Trả về rỗng để không crash app
    
    # Đọc file CSV
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return pd.DataFrame()

# ============================================
# 2. KHỞI TẠO HỆ THỐNG
# ============================================
@st.cache_resource(show_spinner="Đang khởi động AI & Load Metadata...")
def load_system():
    # 1. Tải Ratings trước
    df_ratings = load_ratings_from_drive()
    
    if df_ratings.empty:
        st.error("Không tải được dữ liệu Ratings. Vui lòng kiểm tra kết nối mạng hoặc Link Drive.")
        return None, None

    # 2. Kiểm tra thư mục dữ liệu nhỏ (movies, actors...)
    if not os.path.exists("./data_clean"): 
        st.error("Không tìm thấy thư mục ./data_clean. Bạn đã upload các file nhỏ lên GitHub chưa?")
        return None, None

    # 3. Khởi tạo HybridRecommender và TRUYỀN DATAFRAME VÀO
    # [QUAN TRỌNG]: Phải truyền df_ratings vào đây
    hybrid_sys = HybridRecommender(ratings_df=df_ratings, data_dir="./data_clean")
    
    # 4. Load Metadata (để hiển thị giao diện)
    meta = {}
    meta['movies'] = pd.read_csv("./data_clean/movies_clean.csv")
    
    def load_csv(name):
        p = f"./data_clean/{name}"
        return pd.read_csv(p) if os.path.exists(p) else None
        
    meta['actors'] = load_csv("actors_clean.csv")
    meta['directors'] = load_csv("directors_clean.csv")
    meta['mov_act'] = load_csv("movies_cast_clean.csv")
    meta['mov_dir'] = load_csv("movies_director_clean.csv")
    meta['genres'] = load_csv("genres_clean.csv")
    meta['mov_gen'] = load_csv("movies_genre_clean.csv")
    
    return hybrid_sys, meta

# --- GỌI HÀM LOAD SYSTEM ---
hybrid, metadata = load_system()

if not hybrid: 
    st.stop() # Dừng app nếu load lỗi

# ============================================
# 3. SESSION STATE
# ============================================
if 'page' not in st.session_state: st.session_state['page'] = 'home'
if 'selected_movie_id' not in st.session_state: st.session_state['selected_movie_id'] = None
if 'selected_person_id' not in st.session_state: st.session_state['selected_person_id'] = None
if 'selected_person_type' not in st.session_state: st.session_state['selected_person_type'] = None
if 'user_id' not in st.session_state: st.session_state['user_id'] = 1

if 'search_query' not in st.session_state: st.session_state['search_query'] = ""
if 'search_genre' not in st.session_state: st.session_state['search_genre'] = "Tất cả"
if 'current_page_num' not in st.session_state: st.session_state['current_page_num'] = 1

# ============================================
# 4. HELPER FUNCTIONS
# ============================================
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/500x750.png?text=No+Image"

def get_image_url(path):
    if pd.isna(path) or str(path).strip() == "" or str(path).lower() == "nan": return PLACEHOLDER
    path_str = str(path).strip()
    if len(path_str) < 5: return PLACEHOLDER
    if "http" in path_str: return path_str
    return f"{POSTER_BASE}{path_str}"

def get_movie_details(mid):
    row = metadata['movies'][metadata['movies']['movie_id'] == mid]
    if row.empty: return None
    info = row.iloc[0].to_dict()
    
    info['directors'], info['actors'], info['genres_list'] = [], [], []
    
    if metadata['mov_dir'] is not None:
        d_ids = metadata['mov_dir'][metadata['mov_dir']['movie_id'] == mid]['director_id']
        info['directors'] = metadata['directors'][metadata['directors']['director_id'].isin(d_ids)].to_dict('records')
        
    if metadata['mov_act'] is not None:
        a_ids = metadata['mov_act'][metadata['mov_act']['movie_id'] == mid]['actor_id']
        info['actors'] = metadata['actors'][metadata['actors']['actor_id'].isin(a_ids)].head(6).to_dict('records')
        
    if metadata['mov_gen'] is not None and metadata['genres'] is not None:
        g_ids = metadata['mov_gen'][metadata['mov_gen']['movie_id'] == mid]['genre_id']
        info['genres_list'] = metadata['genres'][metadata['genres']['genre_id'].isin(g_ids)]['name'].tolist()
        
    return info

def get_person_details(pid, role):
    df = metadata['actors'] if role == 'actor' else metadata['directors']
    col_id = 'actor_id' if role == 'actor' else 'director_id'
    row = df[df[col_id] == pid]
    if row.empty: return None
    return row.iloc[0].to_dict()

# --- ACTION FUNCTIONS ---
def go_to_movie(mid):
    st.session_state['selected_movie_id'] = mid
    st.session_state['page'] = 'detail'
    st.rerun()

def go_to_person(pid, role):
    st.session_state['selected_person_id'] = pid
    st.session_state['selected_person_type'] = role
    st.session_state['page'] = 'person_detail'
    st.rerun()

def click_genre(genre_name):
    hybrid.log_search(st.session_state['user_id'], genre_name, search_type='genre')
    st.session_state['search_genre'] = genre_name
    st.session_state['search_query'] = ""
    st.session_state['current_page_num'] = 1 
    st.session_state['page'] = 'home'
    st.rerun()

# Hàm render_grid
def render_grid(df, cols=4, key_prefix="grid"):
    if df is None or df.empty: 
        st.warning("Không tìm thấy phim phù hợp.")
        return
    
    df_show = df[['movie_id']].merge(
        metadata['movies'][['movie_id', 'title', 'poster_path', 'vote_average']], 
        on='movie_id', how='left'
    )
    
    rows = [st.columns(cols) for _ in range((len(df_show)+cols-1)//cols)]
    for i, row in enumerate(rows):
        for j, col in enumerate(row):
            idx = i*cols + j
            if idx < len(df_show):
                m = df_show.iloc[idx]
                with col:
                    img_url = get_image_url(m['poster_path'])
                    st.markdown(
                        f"""
                        <div style="height: 350px; border-radius: 8px; overflow: hidden; margin-bottom: 10px;">
                            <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;">
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.write(f"**{m['title']}**")
                    st.caption(f"⭐ {m['vote_average']:.1f}")
                    
                    unique_key = f"{key_prefix}_{m['movie_id']}"
                    if st.button("Chi tiết", key=unique_key):
                        go_to_movie(m['movie_id'])

def render_chart(name, cap, col=None):
    path = f"./charts/{name}"
    if os.path.exists(path):
        if col: col.image(path, caption=cap, use_container_width=True)
        else: st.image(path, caption=cap, use_container_width=True)
    else:
        msg = f"⚠️ Chưa có biểu đồ: {name}."
        if col: col.caption(msg)
        else: st.caption(msg)

# ============================================
# 5. SIDEBAR
# ============================================
with st.sidebar:
    st.title("🍿 MovieFlix")
    mode = st.radio("Chế độ:", ["Đăng nhập", "Đăng ký"])
    
    if mode == "Đăng nhập":
        uid = st.number_input("User ID:", min_value=1, value=st.session_state['user_id'])
        st.session_state['user_id'] = uid
    else:
        # Lấy ID mới dựa trên dữ liệu ratings đã tải
        if not hybrid.ratings.empty:
            new_id = int(hybrid.ratings['user_id'].max()) + 1
        else:
            new_id = 1
        st.info(f"🆕 ID mới của bạn: **{new_id}**")
        st.session_state['user_id'] = new_id
    
    st.markdown("---")
    if st.button("🏠 Trang Chủ", use_container_width=True): 
        st.session_state['search_query'] = ""
        st.session_state['search_genre'] = "Tất cả"
        st.session_state['current_page_num'] = 1
        st.session_state['page'] = 'home'
        st.rerun()
        
    if st.button("📊 Visualization", use_container_width=True): 
        st.session_state['page'] = 'viz'
        st.rerun()

# ============================================
# PAGE 1: TRANG CHỦ
# ============================================
if st.session_state['page'] == 'home':
    st.title("🏠 Trang Chủ")
    
    with st.expander("🔍 Tìm kiếm & Lọc", expanded=True):
        c1, c2 = st.columns([3, 1])
        
        query_input = c1.text_input("Nhập tên phim, diễn viên...", value=st.session_state['search_query'])
        
        all_genres = set()
        if 'genres_str' in hybrid.cb.movies.columns:
            for g in hybrid.cb.movies['genres_str'].dropna(): all_genres.update(g.split())
        sorted_genres = ["Tất cả"] + sorted(list(all_genres))
        
        try: gen_idx = sorted_genres.index(st.session_state['search_genre'])
        except: gen_idx = 0
        genre_select = c2.selectbox("Thể loại:", sorted_genres, index=gen_idx)

    uid = st.session_state['user_id']
    
    # Logic Input
    if query_input != st.session_state['search_query']:
        st.session_state['search_query'] = query_input
        if query_input: hybrid.log_search(uid, query_input, search_type='keyword')
        st.session_state['current_page_num'] = 1
    
    if genre_select != st.session_state['search_genre']:
        st.session_state['search_genre'] = genre_select
        if genre_select != "Tất cả": hybrid.log_search(uid, genre_select, search_type='genre')
        st.session_state['current_page_num'] = 1

    # HIỂN THỊ KẾT QUẢ
    
    # 1. Đang Tìm kiếm
    if st.session_state['search_query']:
        st.subheader(f"🔍 Kết quả cho: '{st.session_state['search_query']}'")
        render_grid(hybrid.search_hybrid(st.session_state['search_query'], top_k=50), key_prefix="search")
        
    # 2. Đang Lọc Thể loại
    elif st.session_state['search_genre'] != "Tất cả":
        st.subheader(f"🎭 Thể loại: {st.session_state['search_genre']}")
        mask = hybrid.cb.movies['genres_str'].str.contains(st.session_state['search_genre'], na=False)
        render_grid(hybrid.cb.movies[mask].sort_values('vote_count', ascending=False).head(40), key_prefix="genre")
        
    # 3. Mặc định: Gợi ý + Kho Phim
    else:
        # A. Gợi ý
        st.subheader(f"✨ Gợi ý dành riêng cho User {uid}")
        recs = hybrid.recommend_based_on_behavior(uid, top_k=8)
        render_grid(recs, cols=4, key_prefix="rec")
        
        st.markdown("---")
        
        # B. Kho Phim
        st.subheader("🎬 Kho Phim (Tất cả)")
        
        all_movies_sorted = metadata['movies'].sort_values(by='popularity', ascending=False)
        ITEMS_PER_PAGE = 40
        total_items = len(all_movies_sorted)
        total_pages = max(1, (total_items // ITEMS_PER_PAGE) + (1 if total_items % ITEMS_PER_PAGE > 0 else 0))
        
        # Pagination Controls
        col_prev, col_info, col_next = st.columns([1, 8, 1])
        with col_prev:
            if st.button("⬅️ Trước"):
                if st.session_state['current_page_num'] > 1:
                    st.session_state['current_page_num'] -= 1
                    st.rerun()
        with col_next:
            if st.button("Sau ➡️"):
                if st.session_state['current_page_num'] < total_pages:
                    st.session_state['current_page_num'] += 1
                    st.rerun()
        with col_info:
            st.markdown(f"<div style='text-align: center; padding-top: 10px;'>Trang <b>{st.session_state['current_page_num']}</b> / {total_pages}</div>", unsafe_allow_html=True)
        
        start_idx = (st.session_state['current_page_num'] - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        page_data = all_movies_sorted.iloc[start_idx:end_idx]
        
        render_grid(page_data, cols=5, key_prefix="all")

# ============================================
# PAGE 2: CHI TIẾT PHIM
# ============================================
elif st.session_state['page'] == 'detail':
    mid = st.session_state['selected_movie_id']
    if st.button("⬅️ Quay lại"): st.session_state['page'] = 'home'; st.rerun()
    
    m = get_movie_details(mid)
    if m:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(get_image_url(m.get('poster_path')), use_container_width=True)
            if st.button("▶️ XEM PHIM", type="primary"):
                hybrid.log_view(st.session_state['user_id'], mid)
                st.success("Đang phát phim... 🍿")
                st.image(get_image_url(m.get('backdrop_path')), caption="Màn hình đang phát...", use_container_width=True)
        with c2:
            st.title(m['title'])
            st.markdown(f"**Năm:** {str(m.get('release_date'))[:4]} | **Điểm:** {m.get('vote_average')}/10")
            
            if m.get('genres_list'):
                st.write("**Thể loại:**")
                cols_g = st.columns(len(m['genres_list']) + 1)
                for i, g_name in enumerate(m['genres_list']):
                    with cols_g[i]:
                        if st.button(g_name, key=f"gen_{mid}_{i}"): click_genre(g_name)
            
            st.write(m.get('overview'))
            st.markdown("---")
            
            if m['directors']:
                st.markdown("#### 🎬 Đạo diễn")
                cols = st.columns(len(m['directors']) + 2)
                for i, d in enumerate(m['directors']):
                    with cols[i]:
                        img_url = get_image_url(d['profile_path'])
                        st.markdown(f'<img src="{img_url}" style="width:80px; height:80px; border-radius:50%; object-fit:cover; margin-bottom:5px;">', unsafe_allow_html=True)
                        if st.button(d['name'], key=f"dir_{d['director_id']}"):
                            hybrid.log_search(st.session_state['user_id'], d['name'], search_type='director')
                            go_to_person(d['director_id'], 'director')
            
            if m['actors']:
                st.markdown("#### 👥 Diễn viên")
                cols = st.columns(6)
                for i, a in enumerate(m['actors']):
                    with cols[i]:
                        img_url = get_image_url(a['profile_path'])
                        st.markdown(f'<img src="{img_url}" style="width:100%; height:120px; border-radius:8px; object-fit:cover; margin-bottom:5px;">', unsafe_allow_html=True)
                        if st.button(a['name'], key=f"act_{a['actor_id']}"):
                            hybrid.log_search(st.session_state['user_id'], a['name'], search_type='actor')
                            go_to_person(a['actor_id'], 'actor')
            
            st.markdown("---")
            val = st.slider("Đánh giá:", 0.0, 5.0, 0.0, 0.5)
            if st.button("Gửi đánh giá"): 
                hybrid.rate_movie(st.session_state['user_id'], mid, val)
                st.success("Đã lưu!")
        
        st.markdown("---")
        st.subheader("Có thể bạn cũng thích:")
        render_grid(hybrid.get_recommendations_for_viewing(st.session_state['user_id'], mid), key_prefix="detail_rec")

# ============================================
# PAGE 3: CHI TIẾT NGƯỜI
# ============================================
elif st.session_state['page'] == 'person_detail':
    pid = st.session_state['selected_person_id']
    role = st.session_state['selected_person_type']
    
    if st.button("⬅️ Quay lại"): st.session_state['page'] = 'home'; st.rerun()

    person = get_person_details(pid, role)
    if person:
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image(get_image_url(person['profile_path']), use_container_width=True)
        with c2:
            role_vn = "Đạo diễn" if role == 'director' else "Diễn viên"
            st.title(person['name'])
            st.caption(f"Vai trò: {role_vn}")
            gender = "Nam" if person.get('gender') == 2 else ("Nữ" if person.get('gender') == 1 else "N/A")
            st.write(f"**Giới tính:** {gender}")
        
        st.markdown("---")
        st.subheader(f"🎬 Các phim tham gia:")
        
        person_movies = hybrid.get_movies_of_person(person['name'], role)
        if not person_movies.empty:
            render_grid(person_movies, key_prefix="person")
        else:
            st.info("Chưa tìm thấy phim nào.")

# ============================================
# PAGE 4: VIZUALIZATION
# ============================================
elif st.session_state['page'] == 'viz':
    st.title("📊 Data Visualization")
    
    if st.button("⬅️ Quay lại Trang chủ"):
        st.session_state['page'] = 'home'
        st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["⭐ Phân Tích Rating", "🎭 Thể Loại & Xu Hướng", "📈 Tương Quan"])

    with tab1:
        st.header("1. Phân bố Điểm số")
        c1, c2 = st.columns(2)
        render_chart("rating_distribution.png", "Phân bố điểm rating (User chấm)", c1)
        render_chart("vote_average_hist.png", "Phân bố điểm trung bình của Phim", c2)
        st.info("Nhận xét: Người dùng có xu hướng chấm điểm hào phóng (3-4 sao), trong khi điểm IMDB của phim tuân theo phân phối chuẩn.")

    with tab2:
        st.header("2. Thể loại & Phim Nổi bật")
        render_chart("genre_frequency.png", "Top 15 Thể loại phổ biến nhất")
        st.markdown("---")
        c1, c2 = st.columns(2)
        render_chart("top_movies_popularity.png", "Top phim theo Độ phổ biến (Trending)", c1)
        render_chart("top_movies_vote_count.png", "Top phim theo Lượt Vote", c2)

    with tab3:
        st.header("3. Tương quan giữa các chỉ số")
        c1, c2 = st.columns([2, 1])
        render_chart("correlation_heatmap.png", "Heatmap tương quan", c1)
        with c2:
            st.markdown("### Insight:")
            st.write("- **Vote Count** và **Popularity** có tương quan dương mạnh (màu đỏ đậm).")
            
        st.markdown("---")
        st.subheader("Phân bố độ phổ biến")

        render_chart("popularity_hist.png", "Histogram Popularity")
