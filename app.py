# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# # Load model, scaler and encoder
# model = load_model("model.h5")

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("onehot_encoder_geo.pkl", "rb") as f:
#     encoder = pickle.load(f)

# # Title
# st.title("Customer Churn Prediction")

# # Input Fields
# geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
# age = st.slider("Age", 18, 92)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
# balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
# tenure = st.slider("Tenure", 0, 10)
# num_of_products = st.slider("Number of Products", 1, 4)
# has_crcard = st.selectbox("Has Credit Card", ["Yes", "No"])
# is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# # Convert categorical to numeric
# has_crcard = 1 if has_crcard == "Yes" else 0
# is_active = 1 if is_active == "Yes" else 0

# # Prepare numeric features array (order MUST match training)
# numeric_features = np.array([[credit_score, age, tenure, balance, num_of_products, has_crcard, is_active]])

# # One-hot encode geography using loaded encoder
# geo_encoded = encoder.transform([[geography]])  # Returns array shape (1, 3)

# # Concatenate numeric features + one-hot encoded geography
# final_input = np.concatenate([numeric_features, geo_encoded], axis=1)

# # Scale numeric part only (first 7 features)
# final_input[:, :7] = scaler.transform(final_input[:, :7])

# # Predict churn probability on final input
# if st.button("Predict Churn"):
#     prediction_prob = model.predict(final_input)[0][0]
#     if prediction_prob >= 0.5:
#         st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction_prob:.2f})")
#     else:
#         st.success(f"✅ This customer is likely to stay. (Probability: {prediction_prob:.2f})")
# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# # Load model, scaler and encoder
# model = load_model("model.h5")

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("onehot_encoder_geo.pkl", "rb") as f:
#     encoder = pickle.load(f)

# # Title
# st.title("Customer Churn Prediction")

# # Input Fields
# geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
# age = st.slider("Age", 18, 92)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
# balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
# tenure = st.slider("Tenure", 0, 10)
# num_of_products = st.slider("Number of Products", 1, 4)
# has_crcard = st.selectbox("Has Credit Card", ["Yes", "No"])
# is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# # Convert categorical to numeric
# has_crcard = 1 if has_crcard == "Yes" else 0
# is_active = 1 if is_active == "Yes" else 0

# if geography not in encoder.categories_[0]:
#     st.error(f"Error: '{geography}' is not a recognized Geography.")
# else:
#     # Prepare numeric features array (order MUST match training)
#     numeric_features = np.array([[credit_score, age, tenure, balance, num_of_products, has_crcard, is_active]])

#     # One-hot encode geography using loaded encoder
#     geo_encoded = encoder.transform([[geography]])  # shape (1, 3)

#     # Concatenate numeric features + one-hot encoded geography
#     final_input = np.concatenate([numeric_features, geo_encoded], axis=1)

#     # Scale numeric part only (first 7 features)
#     final_input[:, :7] = scaler.transform(final_input[:, :7])

#     # Predict churn probability on final input
#     if st.button("Predict Churn"):
#         prediction_prob = model.predict(final_input)[0][0]
#         if prediction_prob >= 0.5:
#             st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction_prob:.2f})")
#         else:
#             st.success(f"✅ This customer is likely to stay. (Probability: {prediction_prob:.2f})")
# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# # Load model, scaler and encoder
# model = load_model("model.h5")

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("onehot_encoder_geo.pkl", "rb") as f:
#     encoder = pickle.load(f)  # This is actually a LabelEncoder

# # Title
# st.title("Customer Churn Prediction")

# # Input Fields
# geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
# age = st.slider("Age", 18, 92)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
# balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
# tenure = st.slider("Tenure", 0, 10)
# num_of_products = st.slider("Number of Products", 1, 4)
# has_crcard = st.selectbox("Has Credit Card", ["Yes", "No"])
# is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# # Convert categorical to numeric
# has_crcard = 1 if has_crcard == "Yes" else 0
# is_active = 1 if is_active == "Yes" else 0

# # Check if geography is known to encoder
# if geography not in encoder.classes_:
#     st.error(f"Error: '{geography}' is not a recognized Geography.")
# else:
#     # Encode geography label (LabelEncoder returns an integer)
#     geo_encoded_label = encoder.transform([geography])  # shape (1,)

#     # Convert label to one-hot encoding
#     geo_onehot = np.zeros(len(encoder.classes_))
#     geo_onehot[geo_encoded_label[0]] = 1

#     # Prepare numeric features array (order MUST match training)
#     numeric_features = np.array([[credit_score, age, tenure, balance, num_of_products, has_crcard, is_active]])

#     # Concatenate numeric features + one-hot encoded geography
#     final_input = np.concatenate([numeric_features, geo_onehot.reshape(1, -1)], axis=1)

#     # Scale numeric part only (first 7 features)
#     final_input[:, :7] = scaler.transform(final_input[:, :7])

#     # Predict churn probability on final input
#     if st.button("Predict Churn"):
#         prediction_prob = model.predict(final_input)[0][0]
#         if prediction_prob >= 0.5:
#             st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction_prob:.2f})")
#         else:
#             st.success(f"✅ This customer is likely to stay. (Probability: {prediction_prob:.2f})")
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load model, scaler and encoders
model = load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

# Title
st.title("Customer Churn Prediction")

# Input Fields
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
age = st.slider("Age", 18, 92)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
balance = st.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_crcard = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])

# Convert categorical to numeric
has_crcard = 1 if has_crcard == "Yes" else 0
is_active = 1 if is_active == "Yes" else 0

# Prepare numeric features array (order MUST match training)
numeric_features = np.array([[credit_score, age, tenure, balance, num_of_products, has_crcard, is_active]])

try:
    # One-hot encode geography using the loaded encoder
    # geo_encoded = onehot_encoder_geo.transform([[geography]])
    # geo_encoded = onehot_encoder_geo.transform([[geography]])
    # geo_encoded = geo_encoded.toarray()  # Ensure it's a 2D numpy array

    # final_input = np.concatenate([numeric_features, geo_encoded], axis=1)
    final_input = numeric_features  # No geography at all

    
    # Scale numeric part only (first 7 features)
    final_input[:, :7] = scaler.transform(final_input[:, :7])
    
    # Predict churn probability on final input
    if st.button("Predict Churn"):
        prediction_prob = model.predict(final_input)[0][0]
        if prediction_prob >= 0.5:
            st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction_prob:.2f})")
        else:
            st.success(f"✅ This customer is likely to stay. (Probability: {prediction_prob:.2f})")
except ValueError as e:
    # st.error(f"Error: {str(e)}. Please select a valid Geography from {onehot_encoder_geo.categories_[0].tolist()}")
    st.error(f"Error: {str(e)}. Geography encoding failed — please ensure the encoder is a OneHotEncoder.")
