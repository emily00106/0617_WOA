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
from model.Whale_Optimization_Algorithm import WOA

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

# 目標函數
def objective_function(binary_mask):
    """
    目標函數，用於WOA的特徵選擇
    :param binary_mask: 特徵子集的二進制掩碼
    :return: 目標函數值，越小越好（準確率越高，特徵越少）
    """
    # 將二進制掩碼轉換為布林型數組
    binary_mask = np.round(binary_mask).astype(bool)
    # 防止全為 0 的情況
    if np.sum(binary_mask) == 0:
        return 1e9  # 這表示選擇了 0 個特徵，返回一個很大的值
    # 選擇相應的特徵
    selected_train = train_features[:, binary_mask]
    selected_val = val_features[:, binary_mask]
    # 訓練邏輯回歸分類器
    start = time.time()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(selected_train, y_train)
    # 在驗證集上進行預測
    acc = accuracy_score(y_val, clf.predict(selected_val))
    duration = time.time() - start

    # 權重 alpha 用來平衡準確率和特徵數量
    alpha = 0.95  # 越高表示 越重視特徵數量少
    return (1 - alpha) * (1 - acc) + alpha * (np.sum(~binary_mask) / len(binary_mask))

# 匯出特徵提取WOA優化結果
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
    
    with open("selected_features.pkl", "wb") as file:
        pickle.dump(results, file)


# # 繪製訓練過程中的準確率曲線
# def plot_training_history(history):
#     """
#     繪製訓練過程中的準確率曲線
#     :param history: 訓練過程中的歷史紀錄，通常來自 model.fit() 的返回值
#     """
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.title('Training and Validation Accuracy Curve')
#     plt.show()


# # 繪製各分類器準確率的條形圖
# def plot_results(accuracies):
#     plt.figure(figsize=(10, 6))
#     plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
#     plt.xlabel('Classifier')
#     plt.ylabel('Accuracy')
#     plt.title('Comparison of Classifiers')
#     plt.show()


if __name__ == "__main__":
    X_train, X_val, y_train, y_val, label_to_int = load_and_preprocess_data()
    train_features, val_features = extract_features(X_train, X_val)

    # 設置 WOA 參數
    dim = train_features.shape[1]  # 特徵的維度
    bounds = np.array([[0, 1]] * dim)  # 每一個特徵都有 [0, 1] 的範圍
    woa = WOA(objective_function, dim=dim, pop_size=10, max_iter=30, bounds=bounds)
    best_features, best_score, history = woa.optimize()
    # 顯示最優特徵子集和最小目標函數值
    print("最佳特徵遮罩：", best_features)  #是一個二進制掩碼，它指示了選擇的特徵。True 表示選擇該特徵，False 表示不選擇。
    print("最小目標函數值：", best_score)   #這是最佳特徵子集對應的目標函數值，該值越小，表示該子集的分類準確度越高，且特徵數量越少。
    print("歷史最佳分數：", history)
    print('----------------------')

    # 過濾特徵
    selected_train = train_features[:, best_features.astype(bool)]   # 選擇訓練集的特徵
    selected_val = val_features[:, best_features.astype(bool)]       # 選擇驗證集的特徵
    # 顯示選擇的特徵數量
    print(f"選擇的特徵數量：{np.sum(best_features)}")
    print('----------------------')

    # 匯出特徵提取WOA優化結果
    save_results(selected_train, selected_val, best_features, best_score, history, y_train, y_val, label_to_int)
