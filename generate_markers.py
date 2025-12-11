import cv2
import numpy as np

# マーカーの設定（4x4のグリッド、50種類のIDを持つ辞書を使用）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 生成するマーカーのID（左上, 右上, 右下, 左下の順）
ids = [0, 1, 2, 3]
marker_size = 200  # ピクセルサイズ（印刷用）

for id_num in ids:
    # マーカー画像の生成
    img = cv2.aruco.generateImageMarker(aruco_dict, id_num, marker_size)
    
    # ファイルに保存
    filename = f"marker_id_{id_num}.png"
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

print("完了。印刷して切り取ってください。")