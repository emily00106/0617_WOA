import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(main_folder, target_size=(224, 224), test_size=0.5, output_file="preprocessed_data.pkl"):
    labels = []  # 用於儲存標籤
    images = []  # 用於儲存預處理後的圖片
    
    # 遍歷主資料夾
    for label in os.listdir(main_folder):
        label_folder = os.path.join(main_folder, label)
        
        # 確保子資料夾是目錄
        if not os.path.isdir(label_folder):
            continue
        
        # 對每個標籤資料夾中的jpg圖片進行預處理
        for image_filename in os.listdir(label_folder):
            # 檢查文件是否為jpg格式
            if image_filename.endswith(".jpg"):
                image_path = os.path.join(label_folder, image_filename)
                
                # 讀取圖片
                image = cv2.imread(image_path)
                
                # 調整大小
                image = cv2.resize(image, target_size)
                # 標準化圖片（將像素值縮放到0到1的範圍）
                image = image.astype("float32") / 255.0
                
                # 儲存標籤和圖片
                labels.append(label)
                images.append(image)
    
    # 將數據分成訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)

    
    print(f"訓練集：{len(X_train)}, 測試集：{len(X_test)}")
    print(f"圖片 shape: {X_train[0].shape}")
    print(f"類別數量: {len(np.unique(y_train))}")

    # 將預處理後的數據儲存為檔案
    data = {
        "X_train": np.array(X_train), "y_train": np.array(y_train),
        "X_test": np.array(X_test), "y_test": np.array(y_test)
    }
    
    with open(output_file, "wb") as file:
        pickle.dump(data, file)

# 指定主資料夾的路徑
main_folder = "afterccitus/Fruits"

# 執行預處理並分割數據集
preprocess_and_split_data(main_folder)
