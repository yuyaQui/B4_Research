import os
import cv2
import numpy as numpy
import sys
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError

relative_dir = "example"
filename = input(f"'{relative_dir}'フォルダ内の画像ファイル名を入力してください: ")

if not filename:
    print("エラー: ファイル名が入力されていません")
    sys.exit(1)

full_relative_path = os.path.join(relative_dir, filename)

try:
    img = cv2.imread(full_relative_path)
    print(f"'{full_relative_path}' が正常に読み込みました")

except FileNotFoundError:
    print(f"エラー: ファイル '{full_relative_path}' が見つかりません。")
except UnidentifiedImageError:
    print(f"エラー: '{full_relative_path}' は画像ファイルとして認識できません。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")

laplacian = cv2.Laplacian(img,cv2.CV_64F)

plt.subplot(2,1,1), plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2), plt.imshow(laplacian)
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()