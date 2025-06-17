import numpy as np
import cv2
import os
import scipy.special as special  #beta 轉換

#img = cv2.imread(r"C:\Users\emily\Downloads\0510.jpg",-1)

"""
# equalize(太黑)
def equalize_histogram_color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    v_equalized = cv2.equalizeHist(v)
    hsv_img_equalized = cv2.merge([h,s,v_equalized])
    img_color_equalized = cv2.cvtColor(hsv_img_equalized, cv2.COLOR_HSV2BGR)
    return img_color_equalized
"""

def clahe_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    clahe = cv2.createCLAHE(clipLimit= 1.5, tileGridSize=(8,8))  # 創建 CLAHE
    v_clahe = clahe.apply(v)

    hsv_img_clahe = cv2.merge([h,s,v_clahe])
    img_clahe = cv2.cvtColor(hsv_img_clahe, cv2.COLOR_HSV2BGR)
    return img_clahe

#beta 轉換
def beta_correction(f, a = 4.5, b = 4.5):
    g = f.copy()
    nr, nc = f.shape[:2]
    x = np.linspace(0, 1, 256)
    table = np.round(special.betainc(a, b, x) * 255, 0)
    if f.ndim != 3:  #灰階
        for x in range(nr):
            for y in range(nc):
                g[x, y] = table[f[x, y]]

    else:            #彩色
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    g[x, y, k] = table[f[x,y,k]]
    return g

#gmma 轉換
def gamma_correction(f, gamma = 0.5):
    g = f.copy()
    nr,nc = f.shape[:2]
    c = 255.0 / (255.0 ** gamma)
    table = np.zeros(256)
    for i in range(256):
        table[i] = round(i ** gamma * c, 0)
    if f.ndim != 3:
        for x in range(nr):
            for y in range(nc):
                g[x, y] = table[f[x, y]]
    else:
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    g[x, y, k] = table[f[x, y, k]]
    return g

# 進行四種資料增強處理
def process_image(img):
    #直方圖均衡化  改成converScaleAbs
    #img1 = equalize_histogram_color(img)
    img1 = cv2.convertScaleAbs(img, alpha=1.5, beta=10) 
    img2 = clahe_hsv(img)
    img3 = beta_correction(img) 
    img4 = gamma_correction(img)
    return img1, img2, img3, img4

def process_images_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                # 原圖路徑
                image_path = os.path.join(root, filename)

                # 計算相對路徑來保留子資料夾結構
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)

                # 建立對應的輸出資料夾
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # 讀取並處理圖片
                img = cv2.imread(image_path)
                processed_img1, processed_img2, processed_img3, processed_img4 = process_image(img)

                # 儲存處理後的圖片
                output_image_path = os.path.join(output_subfolder, filename)
                output_image_path_convert = os.path.join(output_subfolder, f"{filename.split('.')[0]}_convert.jpg")
                output_image_path_clahe = os.path.join(output_subfolder, f"{filename.split('.')[0]}_clahe.jpg")
                output_image_path_beta = os.path.join(output_subfolder, f"{filename.split('.')[0]}_beta.jpg")
                output_image_path_gamma = os.path.join(output_subfolder, f"{filename.split('.')[0]}_gamma.jpg")
                # output_image_path = os.path.join(output_subfolder, filename)

                cv2.imwrite(output_image_path, img)
                cv2.imwrite(output_image_path_convert, processed_img1)
                cv2.imwrite(output_image_path_clahe, processed_img2)
                cv2.imwrite(output_image_path_beta, processed_img3)
                cv2.imwrite(output_image_path_gamma, processed_img4)
                # print(f"處理並儲存圖片: {output_image_path}")

"""

# 遍歷資料夾中的圖片並進行資料增強
def process_images_in_folder(input_folder, output_folder, beta_a=2.0, beta_b=2.0, gamma_value=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍歷資料夾中的每張圖片
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(image_path) and filename.endswith(".jpg"):
            img = cv2.imread(image_path)

            # 處理圖片
            processed_img = process_image(img, beta_a=beta_a, beta_b=beta_b, gamma_value=gamma_value)

            # 儲存處理後的圖片到指定的資料夾
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, processed_img)
            print(f"處理並儲存圖片: {filename}")
"""

input_folder = "ccitus"
output_folder = "afterccitus"

# 呼叫函數進行資料增強處理
process_images_in_folder(input_folder, output_folder)
process_images_in_folder(input_folder = output_folder, output_folder = output_folder)
"""
img = cv2.convertScaleAbs(img, alpha=2.0, beta=30) 
#img1 = equalize_histogram_color(img)
img1 = clahe_hsv(img)
img2 = beta_correction(img, a = 0.5, b = 0.5)
img3 = gamma_correction(img)

cv2.imshow("DATA AUGMENTATION:", img1)
cv2.waitKey(0) 
"""


