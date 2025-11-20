from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Đường dẫn tới thư mục models
MODELS_DIR = Path("models")

# Load 2 mô hình đã train
with open(MODELS_DIR / "linear_regression_model.pkl", "rb") as f:
    linreg_model = pickle.load(f)

with open(MODELS_DIR / "lasso_regression_model.pkl", "rb") as f:
    lasso_model = pickle.load(f)

# Cột feature phải đúng thứ tự như khi train
FEATURE_COLS = [
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "city",
    "country",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_lin = None
    prediction_lasso = None
    error_msg = None

    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            sqft_living = float(request.form["sqft_living"])
            sqft_lot = float(request.form["sqft_lot"])
            bedrooms = float(request.form["bedrooms"])
            bathrooms = float(request.form["bathrooms"])
            city = request.form["city"]

            # Country cố định là USA
            country = "USA"

            # Tạo DataFrame 1 dòng
            input_dict = {
                "sqft_living": [sqft_living],
                "sqft_lot": [sqft_lot],
                "bedrooms": [bedrooms],
                "bathrooms": [bathrooms],
                "city": [city],
                "country": [country],
            }

            input_df = pd.DataFrame(input_dict, columns=FEATURE_COLS)

            # Dự đoán
            prediction_lin = float(linreg_model.predict(input_df)[0])
            prediction_lasso = float(lasso_model.predict(input_df)[0])

        except Exception as e:
            error_msg = f"Lỗi khi xử lý dữ liệu nhập: {e}"
            print(error_msg)

    return render_template(
        "index.html",
        prediction_lin=prediction_lin,
        prediction_lasso=prediction_lasso,
        error_msg=error_msg,
    )


if __name__ == "__main__":
    app.run(debug=True)
