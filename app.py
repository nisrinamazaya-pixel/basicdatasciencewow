
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- 1. Load the Saved Model and Scaler ---
model_filename = 'model_gradient_boosting.pkl'
saler_filename = 'feature_scaler.pkl'

try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure they are in the same directory as app.py.")
    st.stop() # Stop the app if essential files are missing

# --- 2. Define Preprocessing Components (based on training data) ---
# These are the unique values and order derived from your training data (df_bersih)
# They are crucial for consistent encoding during prediction.

unique_pendidikan = ['SMA', 'SMK', 'S1', 'D3'] # Ensure this matches df_bersih['Pendidikan'].unique()
unique_jurusan = ['administrasi', 'teknik las', 'desain grafis', 'teknik listrik', 'otomotif'] # Ensure this matches df_bersih['Jurusan'].unique() (after lowercasing)

le_pendidikan = LabelEncoder()
le_pendidikan.fit(unique_pendidikan)

le_jurusan = LabelEncoder()
le_jurusan.fit(unique_jurusan)

# Mapping for Jenis_Kelamin cleaning
mapping_gender = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'P': 'Wanita'}

# Feature columns and their order used during model training
feature_cols = ['Pendidikan', 'Jurusan', 'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 
                'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja', 
                'Usia', 'Durasi_Jam', 'Nilai_Ujian']

# --- 3. Prediction Function ---
def predict_gaji(usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja):
    # Create a DataFrame for the new input
    new_data_input = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }
    new_df = pd.DataFrame([new_data_input])

    # Apply categorical data cleaning
    new_df['Jenis_Kelamin'] = new_df['Jenis_Kelamin'].replace(mapping_gender)
    new_df['Jurusan'] = new_df['Jurusan'].str.lower()

    # Apply Label Encoding
    new_df_label_encoded = pd.DataFrame({
        'Pendidikan': le_pendidikan.transform(new_df['Pendidikan']),
        'Jurusan': le_jurusan.transform(new_df['Jurusan'])
    })

    # Apply One-Hot Encoding
    list_one_hot = ['Jenis_Kelamin', 'Status_Bekerja']
    new_df_one_hot = pd.get_dummies(new_df[list_one_hot], prefix=list_one_hot)
    new_df_one_hot = new_df_one_hot.astype(int)

    # Align one-hot encoded columns with training data
    one_hot_training_cols = [col for col in feature_cols if any(prefix in col for prefix in ['Jenis_Kelamin_', 'Status_Bekerja_'])]
    processed_one_hot_aligned = pd.DataFrame(0, index=new_df_one_hot.index, columns=one_hot_training_cols)
    for col in new_df_one_hot.columns:
        if col in processed_one_hot_aligned.columns:
            processed_one_hot_aligned[col] = new_df_one_hot[col]

    # Extract numerical columns
    numerical_cols_ordered = ['Usia', 'Durasi_Jam', 'Nilai_Ujian']
    new_df_numerical = new_df[numerical_cols_ordered].copy()

    # Combine all processed features
    processed_new_data = pd.concat([
        new_df_label_encoded['Pendidikan'],
        new_df_label_encoded['Jurusan'],
        processed_one_hot_aligned,
        new_df_numerical
    ], axis=1)

    # Ensure the order of columns matches the training features
    processed_new_data = processed_new_data[feature_cols]

    # Scale the processed data
    scaled_new_data = loaded_scaler.transform(processed_new_data)

    # Make prediction
    predicted_gaji = loaded_model.predict(scaled_new_data)
    return predicted_gaji[0]

# --- 4. Streamlit UI ---
st.title("Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.write("Masukkan informasi di bawah untuk memprediksi gaji awal.")

# Input fields
usia = st.number_input("Usia", min_value=18, max_value=60, value=25)
durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=20, max_value=100, value=50)
nilai_ujian = st.slider("Nilai Ujian", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
pendidikan = st.selectbox("Pendidikan", unique_pendidikan)
jurusan = st.selectbox("Jurusan", unique_jurusan.tolist())
jenis_kelamin = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Wanita'])
status_bekerja = st.selectbox("Status Bekerja", ['Sudah Bekerja', 'Belum Bekerja'])

if st.button("Prediksi Gaji Awal"):
    try:
        predicted_value = predict_gaji(usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja)
        st.success(f"Prediksi Gaji Awal (Juta): **{predicted_value:.2f}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

