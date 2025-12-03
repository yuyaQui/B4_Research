from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from io import BytesIO
import torch
import cv2
import math
import numpy as np
from torchvision import transforms, utils, models
from utils.data_process import preprocess_img, postprocess_img
from PIL import Image
import os # os をインポート

def generate_image_from_quiz():
    try:
        client = genai.Client()

        print("クイズの情報を入力してください。")
        question = input("問題文: ")
        answer = input("解答: ")

        prompt = (
            f"""
            あなたが画像を生成するAIです。あなたの唯一のタスクは、以下の情報を基にイラストを生成することです。
            あなたの出力は画像データでなければなりません。説明、文章、その他のテキストは一切出力しないでください。

            # 指示
            - 以下の[問題文]と[解答]の内容を忠実に表現したイラストを生成してください。
            - [問題文]の文脈を正しく読み取り、文章がなくても画像だけで内容を人間が簡単に理解できるような画像の生成を目標としてください。
            - イラスト内には、いかなる文字、単語、数字を含まないでください。
            - イラストは、被写体が大きく描かれ、背景の空白が少なくなるように構成してください。
            - スタイル: 適したスタイルを適宜判断

            # 情報
            [問題文]
            {question}

            [解答]
            {answer}
            """
        )
        print("\n画像を生成中です...しばらくお待ちください。")

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        # 修正: response と response.candidates が有効かチェックする
        if response and response.candidates:
            image_found = False
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    # 画像データが見つかった場合の処理
                    image = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                    output_filename = r"example/generated_image.png"
                    image.save(output_filename)
                    print(f"Geminiが生成した画像を'{output_filename}'として保存しました")
                    image_found = True
                    return image, answer

            # ループが終了しても画像が見つからなかった場合
            if not image_found:
                print("❌ 応答に画像データが含まれていませんでした。")
                # テキスト応答があれば表示する（デバッグ用）
                if response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
                    print(f"モデルのテキスト応答: {response.candidates[0].content.parts[0].text}")
                return None, None
        else:
            # response自体が無効だった場合
            print("❌ モデルから有効な応答が得られませんでした。")
            return None, None

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None, None

def distance(x, y, i, j):
    return int(math.sqrt((x - i)**2 + (y - j)**2)) # 距離計算関数を修正

if __name__ == "__main__":

    generated_pil_image, answer_text = generate_image_from_quiz()

    if generated_pil_image is None:
        print("画像の生成に失敗したため、処理を終了します。")
    else:
        generated_image_path = "example/generated_image.png" # generate_image_from_quizの出力パスと一致させる
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"CUDA利用可能: {torch.cuda.is_available()}")

        flag = 1 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

        if flag:
            from TranSalNet_Res import TranSalNet
            model = TranSalNet()
            model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Res.pth'))
        else:
            from TranSalNet_Dense import TranSalNet
            model = TranSalNet()
            model.load_state_dict(torch.load(r'pretrained_models\TranSalNet_Dense.pth'))

        # --- 2-1. Saliency Map計算の準備 ---
        model = model.to(device) 
        model.eval()

        # --- 2-2. 入力画像の前処理 ---
        img = preprocess_img(generated_image_path) # 画像をクロップ&Numpyに変換
        img = np.array(img)/255.
        img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)

        # --- 2-3. PyTorchテンソルへの変換と推論 ---
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor).to(device)
        pred_saliency_tensor = model(img)

        # --- 2-4. 出力の後処理 ---
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency_tensor.squeeze())
        saliency_map_np = postprocess_img(pic, generated_image_path) # Numpy配列に変換
        
        # --- 2-5. 顕著性マップの保存 ---
        saliency_output_filename = r'example/result_saliency.png'
        cv2.imwrite(saliency_output_filename, saliency_map_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"Saliency Mapを '{saliency_output_filename}' として保存しました。")

        # --- 2-6. Saliency Mapの最大値（最も目立つ点）を取得 ---
        max_loc_flat = np.argmax(saliency_map_np)
        max_y, max_x = np.unravel_index(max_loc_flat, saliency_map_np.shape)
        print(f"💡 Saliency Mapで最も注視される座標 (x, y): ({max_x}, {max_y})")

        saliency_center_x = int(max_x)
        saliency_center_y = int(max_y)

        saliency_threshold = 50
        print(f"\n Saliencyの閾値を{saliency_threshold}として切り出し領域を計算します")

        y_coords, x_coords = np.where(saliency_map_np > saliency_threshold)

        top = float('inf')
        bottom = 0
        left = float('inf')
        right = 0

        if y_coords.size > 0: # バウンディングボックスの頂点
            top = np.min(y_coords)
            bottom = np.max(y_coords)
            left = np.min(x_coords)
            right = np.max(x_coords)
        else:
            print(f"閾値{saliency_threshold}を超えるSaliency領域が見つかりませんでした。")

        draw = ImageDraw.Draw(generated_pil_image)
        
        # フォントとテキストサイズの準備
        font_path = "C:/Windows/Fonts/meiryob.ttc"
        font_size = 36
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 指定されたフォント '{font_path}' が見つかりません。デフォルトフォントを使用します。")
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_half_width = text_width // 2
        text_half_height = text_height // 2

        img_width, img_height = generated_pil_image.size

        gray_std_threshold = 1
        gray_flag_var = float('inf')
        final_x, final_y = saliency_center_x, saliency_center_y

        for i in range(left, right):
            for j in range(top, bottom):
                crop_left = max(0, i - text_half_width)
                crop_top = max(0, j - text_half_height)
                crop_right = min(img_width, i + text_half_width)
                crop_bottom = min(img_height, j + text_half_height)

                if crop_left < crop_right and crop_top < crop_bottom:
                    patch_pil = generated_pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                    patch_gray = patch_pil.convert("L")
                    patch_np = np.array(patch_gray)

                    if patch_np.size == 0:
                        continue

                    gray_sum = np.sum(patch_np)
                    gray_std = np.std(patch_np)
                    factor_var = gray_sum + gray_std

                    if gray_std < gray_std_threshold:
                        continue

                    if factor_var < gray_flag_var:
                        gray_flag_var = factor_var
                        final_x, final_y = i, j
        
        print(f"✅ スキャン完了。")
        print(f"💡 最適なテキスト配置座標 (x, y): ({final_x}, {final_y})")

        text_x = final_x - text_half_width
        text_y = final_y - text_half_height

        if text_x < 0:
            text_x = 0
        if text_x > img_width - text_half_width * 2:
            text_x = img_width - text_half_width * 2
        if text_y < 0:
            text_y = 0
        if text_y > img_height + text_half_height * 2:
            text_y = img_height + text_half_height * 2
        
        fill_color = "#00ff00"
        stroke_color = "black"
        stroke_width = 3
        draw.text(
            (text_x, text_y),
            answer_text,
            font=font, 
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )

        final_output_filename = r"example/final_result_with_answer.png"
        generated_pil_image.save(final_output_filename)
        print(f"🎉 完成！解答テキスト付き画像を '{final_output_filename}' として保存しました。")