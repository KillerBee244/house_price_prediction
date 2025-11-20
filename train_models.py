import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path

# ==============================
# 1. ĐỌC DỮ LIỆU
# ==============================

DATA_PATH = Path("data") / "houses.csv"

df = pd.read_csv(DATA_PATH)

# TÁCH ZIPCODE TỪ statezip (ví dụ "WA 98136" -> 98136)
df["zipcode"] = (
    df["statezip"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

df["zipcode"] = pd.to_numeric(df["zipcode"], errors="coerce")

# BỘ THUỘC TÍNH TỐI ƯU (bỏ street, statezip, country, date)
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
    "zipcode",         # mã vùng bưu điện (feature rất mạnh)
    "city",            # thành phố/khu vực
]

target_col = "price"   # giá nhà (USD) - biến cần dự đoán

# Chỉ giữ cột cần thiết
df = df[feature_cols + [target_col]].copy()

# Chuyển các cột số về dạng numeric (nếu có ký tự lạ, '?' -> NaN)
numeric_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition",
    "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "zipcode",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Bỏ các dòng bị thiếu (NaN) ở feature hoặc target
df = df.dropna(subset=feature_cols + [target_col])

X = df[feature_cols]
y = df[target_col]

# ==============================
# 2. TIỀN XỬ LÝ (NUMERIC + CATEGORICAL)
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
# 3. MÔ HÌNH RANDOM FOREST (MẠNH)
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

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "house_price_rf.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Đã lưu model Random Forest tại: {model_path}")
