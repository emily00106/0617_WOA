import pickle
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeClassifier      #細樹
from sklearn.ensemble import RandomForestClassifier  #粗樹
from sklearn.naive_bayes import GaussianNB           #高斯樸素貝葉斯
from sklearn.svm import SVC                          #SVM




# 加載處理過的特徵
def load_selected_features():
    with open("noWOA_selected_features.pkl", "rb") as file:
        results = pickle.load(file)
        print('------------------------')
        print(results.keys())
        print('-------------------------')

    return results["selected_train"], results["selected_val"], results["best_features"], results["best_score"], results["history"], results["y_train"], results["y_val"], results["label_to_int"]


# 創建神經網路模型
def create_model(input_dim):
    """
    創建神經網路模型，故意減弱性能
    :param input_dim: 輸入層的維度（即選擇的特徵數量）
    :param num_classes: 類別數（即最終的分類數量）
    :return: 編譯過的模型
    """
    model = Sequential()

    # 降低隱藏層的神經元數量
    model.add(Dense(256, activation='relu', input_dim=input_dim))  # 較少的神經元

    # 增加更多的 Dropout 比例，丟棄更多神經元
    model.add(Dropout(0.3))  # 提高 Dropout 比例，丟棄更多神經元

    # 添加更多較小的層，限制模型的學習能力
    model.add(Dense(128, activation='relu'))  # 少量的神經元
    model.add(Dropout(0.3))  # 這一層也丟棄更多神經元

    # 加入高 L2 正則化，限制權重的大小
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))  # 使用高正則化強度
    model.add(Dropout(0.3))  # 這一層丟棄更多神經元

    # 輸出層，這裡仍然保持 softmax，這樣可以進行多分類
    model.add(Dense(len(label_to_int), activation='softmax'))  # 輸出層，類別數與 label_to_int 的長度一致

    # 使用較低的學習率來降低模型的訓練速度
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 訓練神經網路模型
def train_and_evaluate_model(model, selected_train, selected_val, y_train_oh_selected, y_val_oh_selected):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(selected_train, y_train_oh_selected, epochs=200, batch_size=32, validation_data=(selected_val, y_val_oh_selected), callbacks=[early_stopping])
    
    # 評估模型
    loss, acc = model.evaluate(selected_val, y_val_oh_selected)
    print('----------------------')
    print(f"**Neural Network**：{acc * 100:.2f}%")

    # accuracies["Neural Network"] = acc

    return history, acc


# 定義訓練、比較不同的分類器 
def compare_classifiers(selected_train, selected_val, y_train, y_val, accNN):
    classifiers = {
    "Fine Tree": DecisionTreeClassifier(max_depth=5, random_state=50),    # 調整 Decision Tree 參數
    # "Coarse Tree": RandomForestClassifier(n_estimators=100),
    "Gaussian Naive Bayes": GaussianNB(),
    # "Linear SVM": SVC(kernel='linear'),
    # "Quadratic SVM": SVC(kernel='poly', degree=2, C=1.0, random_state=42),
    }

    accuracies = {}
    
    for clf_name, clf in classifiers.items():
        clf.fit(selected_train, y_train)
        y_pred = clf.predict(selected_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies[clf_name] = acc
        print(f"{clf_name} Accuracy: {acc * 100:.2f}%")
    
    accuracies["Neural Network"] = accNN

    return accuracies


# 繪製訓練過程中的準確率曲線
def plot_training_history(history):
    """
    繪製訓練過程中的準確率曲線
    :param history: 訓練過程中的歷史紀錄，通常來自 model.fit() 的返回值
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy Curve')
    plt.show()


# 繪製各分類器準確率的條形圖
def plot_results(accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='#4db8b8')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Classifiers')
    plt.show()



if __name__ == "__main__":

    #載入特徵選擇資料
    selected_train, selected_val, best_features, best_score, history, y_train, y_val, label_to_int = load_selected_features()    
    
    # 將標籤轉換為 one-hot 編碼
    y_train_oh_selected = to_categorical(y_train, num_classes=len(label_to_int))
    y_val_oh_selected = to_categorical(y_val, num_classes=len(label_to_int))

    #創建模型並訓練
    model = create_model(selected_train.shape[1])

    history, accNN = train_and_evaluate_model(model, selected_train, selected_val, y_train_oh_selected, y_val_oh_selected)
    
    # 繪製訓練過程中的準確率曲線
    plot_training_history(history)
    
    accuracies = compare_classifiers(selected_train, selected_val, y_train, y_val, accNN)

    plot_results(accuracies)




