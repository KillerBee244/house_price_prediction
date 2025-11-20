import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from pathlib import Path

# ==============================
# 1. ĐỌC DỮ LIỆU
# ==============================

DATA_PATH = Path("data") / "houses.csv"

df = pd.read_csv(DATA_PATH)

# Chỉ dùng các cột:
# sqft_living – diện tích sàn
# sqft_lot    – diện tích đất
# bedrooms    – số phòng ngủ
# bathrooms   – số phòng tắm
# city        – thành phố
# country     – quốc gia
feature_cols = [
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "city",
    "country",
]
target_col = "price"

# Giữ đúng các cột cần thiết
df = df[feature_cols + [target_col]].copy()

# Bỏ dòng bị thiếu dữ liệu
df = df.dropna()

X = df[feature_cols]
y = df[target_col]

# ==============================
# 2. TIỀN XỬ LÝ
# ==============================

numeric_features = ["sqft_living", "sqft_lot", "bedrooms", "bathrooms"]
categorical_features = ["city", "country"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ==============================
# 3. TẠO PIPELINE CHO 2 MÔ HÌNH
# ==============================

linreg_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

lasso_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Lasso(alpha=1.0, max_iter=10000))
])

# ==============================
# 4. CHIA TRAIN / TEST
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. TRAIN VÀ ĐÁNH GIÁ
# ==============================

# Linear Regression
linreg_pipeline.fit(X_train, y_train)
y_pred_lin = linreg_pipeline.predict(X_test)

print("=== Linear Regression ===")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R2 :", r2_score(y_test, y_pred_lin))

# Lasso Regression
lasso_pipeline.fit(X_train, y_train)
y_pred_lasso = lasso_pipeline.predict(X_test)

print("\n=== Lasso Regression ===")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R2 :", r2_score(y_test, y_pred_lasso))

# ==============================
# 6. LƯU FILE .PKL
# ==============================

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

with open(models_dir / "linear_regression_model.pkl", "wb") as f:
    pickle.dump(linreg_pipeline, f)

with open(models_dir / "lasso_regression_model.pkl", "wb") as f:
    pickle.dump(lasso_pipeline, f)

print("\n✅ Đã lưu models/linear_regression_model.pkl")
print("✅ Đã lưu models/lasso_regression_model.pkl")
