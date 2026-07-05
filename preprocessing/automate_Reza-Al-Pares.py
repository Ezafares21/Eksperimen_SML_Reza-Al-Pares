import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_path):
    print(f"[*] Membaca data mentah dari: {input_path}")
    df = pd.read_csv(input_path, sep=';')
    
    # Pembersihan
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    # Outlier Capping
    outlier_cols = ['balance', 'duration']
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])
        df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col])

    # Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    # Standarisasi
    fitur_cols = df.columns.drop('y')
    scaler = StandardScaler()
    df[fitur_cols] = scaler.fit_transform(df[fitur_cols])
    
    # Simpan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    # Setup Jalur Folder Otomatis mengikuti instruksi Dicoding
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Ini adalah folder /preprocessing/
    ROOT_DIR = os.path.dirname(CURRENT_DIR) # Ini adalah root repo
    
    INPUT_FILE = os.path.join(ROOT_DIR, 'Credit_Scoring_Bank_raw', 'Credit_Scoring_Bank.csv')
    OUTPUT_FILE = os.path.join(CURRENT_DIR, 'Credit_Scoring_Bank_preprocessing', 'Credit_Scoring_Bank_clean.csv')
    
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)
