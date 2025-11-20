from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Đường dẫn tới thư mục models
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "house_price_rf.pkl"

# Load mô hình Random Forest đã train
with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

# Danh sách cột phải TRÙNG với train_models.py
FEATURE_COLS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "city",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_msg = None

    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            bedrooms = float(request.form["bedrooms"])
            bathrooms = float(request.form["bathrooms"])
            sqft_living = float(request.form["sqft_living"])
            sqft_lot = float(request.form["sqft_lot"])
            floors = float(request.form["floors"])
            waterfront = float(request.form["waterfront"])
            view = float(request.form["view"])
            condition = float(request.form["condition"])
            sqft_above = float(request.form["sqft_above"])
            sqft_basement = float(request.form["sqft_basement"])
            yr_built = float(request.form["yr_built"])
            yr_renovated = float(request.form["yr_renovated"])
            zipcode = float(request.form["zipcode"])
            city = request.form["city"]

            # Tạo DataFrame 1 dòng
            input_dict = {
                "bedrooms": [bedrooms],
                "bathrooms": [bathrooms],
                "sqft_living": [sqft_living],
                "sqft_lot": [sqft_lot],
                "floors": [floors],
                "waterfront": [waterfront],
                "view": [view],
                "condition": [condition],
                "sqft_above": [sqft_above],
                "sqft_basement": [sqft_basement],
                "yr_built": [yr_built],
                "yr_renovated": [yr_renovated],
                "zipcode": [zipcode],
                "city": [city],
            }

            input_df = pd.DataFrame(input_dict, columns=FEATURE_COLS)

            # Dự đoán với Random Forest
            prediction = float(rf_model.predict(input_df)[0])

        except Exception as e:
            error_msg = f"Lỗi khi xử lý dữ liệu nhập: {e}"
            print(error_msg)

    return render_template(
        "index.html",
        prediction=prediction,
        error_msg=error_msg,
    )


if __name__ == "__main__":
    app.run(debug=True)
