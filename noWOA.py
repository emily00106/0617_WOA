import numpy as np
import pickle
import time
from keras.applications import MobileNetV2
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 載入資料
def load_and_preprocess_data():
    with open("preprocessed_data.pkl", "rb") as file:
        data = pickle.load(file)
    
    images = data['X_train']   # 使用訓練集的圖片
    labels = data['y_train']   # 使用訓練集的標籤
    label_to_int = {label: i for i, label in enumerate(np.unique(labels))}        # 創建標籤到整數的映射
    int_labels = np.array([label_to_int[label] for label in labels])              # 使用映射將標籤轉換為整數
    one_hot_labels = to_categorical(int_labels, num_classes=len(label_to_int))    # 將整數標籤進行獨熱編碼
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)   # 分割數據集為訓練集和驗證集
    
    y_train = np.argmax(y_train_oh, axis=1)   #轉回整數
    y_val = np.argmax(y_val_oh, axis=1)       #轉回整數
    
    return X_train, X_val, y_train, y_val, label_to_int

# 提取特徵--使用MobileNetV2 提取特徵
def extract_features(X_train, X_val):
    #去除分類層，只保留捲積層作為特徵提取器
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #設置base_model所有層不可訓練
    for layer in base_model.layers:
        layer.trainable = False
    
    #創建特徵提取模型，將輸出改為GlobalAveragePooling2D
    feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    #使用特徵提取模型提取訓練集和驗證集特徵
    train_features = feature_extractor.predict(X_train)
    val_features = feature_extractor.predict(X_val)
    
    # 顯示特徵的形狀（通常是 [樣本數, 特徵維度]）
    print("Train features shape:", train_features.shape)  # (n_samples, 1280)
    print("Validation features shape:", val_features.shape)  # (n_samples, 1280)
    print('--------------')

    return train_features, val_features

# 匯出特徵
def save_results(selected_train, selected_val, best_features, best_score, history, y_train, y_val, label_to_int):
    results = {
        "selected_train": selected_train,
        "selected_val": selected_val,
        "best_features": best_features,
        "best_score": best_score,
        "history": history,
        "y_train": y_train,  
        "y_val": y_val, 
        "label_to_int": label_to_int   
    }
    
    with open("noWOA_selected_features.pkl", "wb") as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, label_to_int = load_and_preprocess_data()
    train_features, val_features = extract_features(X_train, X_val)

    # 不使用 WOA，直接選擇所有特徵
    selected_train = train_features  # 使用所有特徵
    selected_val = val_features  # 使用所有特徵

    # 顯示選擇的特徵數量
    print(f"選擇的特徵數量：{selected_train.shape[1]}")
    print('----------------------')

    # 匯出結果
    save_results(selected_train, selected_val, None, None, None, y_train, y_val, label_to_int)
