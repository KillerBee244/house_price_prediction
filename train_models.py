import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# 0. PATH & TẠO FOLDER LƯU ẢNH
# ==============================

DATA_PATH = Path("data") / "houses.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

IMG_DIR = Path("static") / "img"
os.makedirs(IMG_DIR, exist_ok=True)

# ==============================
# 1. ĐỌC & CHUẨN BỊ DỮ LIỆU
# ==============================

df = pd.read_csv(DATA_PATH)

# TÁCH ZIPCODE TỪ statezip (vd: "WA 98136" -> 98136)
df["zipcode"] = (
    df["statezip"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)
df["zipcode"] = pd.to_numeric(df["zipcode"], errors="coerce")

# BỘ THUỘC TÍNH ĐANG DÙNG
feature_cols = [
    "bedrooms",        # số phòng ngủ
    "bathrooms",       # số phòng tắm
    "sqft_living",     # diện tích sinh hoạt
    "sqft_lot",        # diện tích đất
    "floors",          # số tầng
    "waterfront",      # có view mặt nước không (0/1)
    "view",            # chất lượng view (0-4)
    "condition",       # tình trạng nhà (1-5)
    "sqft_above",      # diện tích trên mặt đất
    "sqft_basement",   # diện tích tầng hầm
    "yr_built",        # năm xây
    "yr_renovated",    # năm cải tạo
    "zipcode",         # mã vùng
    "city",            # thành phố
]

target_col = "price"   # giá nhà (USD) - biến mục tiêu

df = df[feature_cols + [target_col]].copy()

numeric_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition",
    "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "zipcode",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_cols + [target_col])

X = df[feature_cols]
y = df[target_col]

# ==============================
# 2. TIỀN XỬ LÝ
# ==============================

numeric_features = numeric_cols
categorical_features = ["city"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ==============================
# 3. MÔ HÌNH RANDOM FOREST
# ==============================

rf_regressor = RandomForestRegressor(
    n_estimators=500,
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf_regressor),
])

# ==============================
# 4. CHIA TRAIN / TEST
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. TRAIN & ĐÁNH GIÁ
# ==============================

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("=== Random Forest Regression (tối ưu với zipcode) ===")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")

# ==============================
# 6. LƯU MODEL
# ==============================

model_path = MODELS_DIR / "house_price_rf.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Đã lưu model Random Forest tại: {model_path}")

# ==============================
# 7. VẼ CÁC BIỂU ĐỒ ĐỂ VIẾT BÁO CÁO
# ==============================

# ----- 7.1. Scatter: Giá thực tế vs. Giá dự đoán -----
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())
plt.plot([min_price, max_price], [min_price, max_price], "--")
plt.xlabel("Giá thực tế (USD)")
plt.ylabel("Giá dự đoán (USD)")
plt.title("So sánh giá thực tế vs giá dự đoán")
plt.tight_layout()
scatter_path = IMG_DIR / "y_true_vs_pred.png"
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"📊 Đã lưu biểu đồ: {scatter_path}")

# ----- 7.2. Histogram: Phân bố sai số (residuals) -----
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=50)
plt.xlabel("Sai số (Giá thực tế - Giá dự đoán) [USD]")
plt.ylabel("Tần suất")
plt.title("Phân bố sai số dự đoán (Residuals)")
plt.tight_layout()
residuals_path = IMG_DIR / "residuals_hist.png"
plt.savefig(residuals_path, dpi=150)
plt.close()
print(f"📊 Đã lưu biểu đồ: {residuals_path}")

# ----- 7.3. Feature Importance (tổng hợp theo feature gốc) -----
rf = model.named_steps["rf"]
pre = model.named_steps["preprocess"]

# tên cột numeric giữ nguyên
num_names = np.array(numeric_features)

# tên cột sau one-hot cho city
cat_transformer = pre.named_transformers_["cat"]
city_feature_names = cat_transformer.get_feature_names_out(["city"])

all_feature_names = np.concatenate([num_names, city_feature_names])
importances = rf.feature_importances_

# Lấy top 15 feature quan trọng nhất
idx = np.argsort(importances)[::-1][:15]
top_features = all_feature_names[idx]
top_importances = importances[idx]

plt.figure(figsize=(8, 5))
plt.barh(range(len(top_features)), top_importances[::-1])
plt.yticks(range(len(top_features)), top_features[::-1])
plt.xlabel("Độ quan trọng tương đối")
plt.title("Top 15 thuộc tính quan trọng nhất (Random Forest)")
plt.tight_layout()
fi_path = IMG_DIR / "feature_importance.png"
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"📊 Đã lưu biểu đồ: {fi_path}")
