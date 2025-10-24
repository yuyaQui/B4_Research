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
        img = preprocess_img(generated_image_path)
        img = np.array(img)/255.
        img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)

        # --- 2-3. PyTorchテンソルへの変換と推論 ---
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor).to(device)
        pred_saliency_tensor = model(img)

        # --- 2-4. 出力の後処理 ---
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency_tensor.squeeze())
        saliency_map_np = postprocess_img(pic, generated_image_path)
        
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

        # --- テキスト描画の準備 ---
        square_size = 100
        half_size = square_size // 2

        # Saliency中心から探索する領域を定義
        left = saliency_center_x - half_size
        top = saliency_center_y - half_size
        right = saliency_center_x + half_size
        bottom = saliency_center_y + half_size

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
        
        img_width, img_height = generated_pil_image.size
        
        # --- 2-7. Saliency Mapの勾配が最大となる位置を探索 ---
        print(f"\n🔍 Saliency Mapの注目領域 ({left}, {top}) から ({right}, {bottom}) の勾配を計算します...")

        # Saliency Mapのサイズを取得 (NumPyは H, W の順)
        saliency_height, saliency_width = saliency_map_np.shape

        # 座標が画像境界をはみ出さないように調整 (クロップ座標を決定)
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(saliency_width, right)
        crop_bottom = min(saliency_height, bottom)

        # 注目領域（Saliencyの中心周辺）を切り出す
        # NumPyのスライス [y1:y2, x1:x2]
        saliency_crop = saliency_map_np[crop_top:crop_bottom, crop_left:crop_right]

        if saliency_crop.size == 0:
            print("警告: Saliency Mapのクロップ領域が空です。中心座標をそのまま使います。")
            final_x, final_y = saliency_center_x, saliency_center_y
        else:
            # Saliency MapのX方向とY方向の勾配を計算 (cv2.CV_64Fで負の値も考慮)
            grad_x_full = cv2.Sobel(saliency_map_np, cv2.CV_64F, 1, 0, ksize=5) # 全体で計算
            grad_y_full = cv2.Sobel(saliency_map_np, cv2.CV_64F, 0, 1, ksize=5) # ksizeを5に調整して平滑化

            # Saliencyが最も大きい点での勾配ベクトル
            main_grad_x = grad_x_full[saliency_center_y, saliency_center_x]
            main_grad_y = grad_y_full[saliency_center_y, saliency_center_x]
            
            # 勾配の大きさが0に近い場合は回避 (例: 完全に平坦なSaliencyの場合)
            grad_magnitude_at_center = math.sqrt(main_grad_x**2 + main_grad_y**2)

            # テキストボックスの大きさの目安
            text_box_diag_length = math.sqrt(text_width**2 + text_height**2)
            # 配置オフセット量を調整（テキストボックスの対角線の半分程度を基準に）
            # Saliencyが減る方向に移動させたいので、勾配の反対方向へ
            offset_factor = text_box_diag_length * 0.75 # 0.75は調整可能

            if grad_magnitude_at_center > 0.1: # 勾配が十分に大きい場合
                # 勾配の正規化ベクトル（Saliencyが増加する方向）
                normalized_grad_x = main_grad_x / grad_magnitude_at_center
                normalized_grad_y = main_grad_y / grad_magnitude_at_center

                # Saliencyが減少する方向にオフセット
                offset_x = -normalized_grad_x * offset_factor
                offset_y = -normalized_grad_y * offset_factor

                # 初期配置候補点 (saliency_center_x, saliency_center_y) にオフセットを加算
                final_x = int(saliency_center_x + offset_x)
                final_y = int(saliency_center_y + offset_y)
            else:
                # 勾配が小さい場合はSaliency中心から少しずらすなど、別のロジックを検討
                # 今回はSaliency中心から少し右下にオフセットする（例）
                print("警告: Saliency中心での勾配が小さいため、デフォルトのオフセットを使用します。")
                final_x = saliency_center_x + half_size // 2
                final_y = saliency_center_y + half_size // 2

            print(f"✅ 勾配計算とオフセット完了。")
            print(f"💡 最適なテキスト配置座標 (x, y): ({final_x}, {final_y})")

        # ----- 3. 元画像に解答テキストを描画 -----
        
        offset_height = 6 # フォントのベースライン微調整用
        # テキストを final_x, final_y の中心に配置するための左上座標を計算
        text_x = final_x - (text_width / 2)
        text_y = final_y - (text_height / 2) + offset_height
        
        # テキストが画像からはみ出さないように座標を最終調整
        
        # 左端のチェックと調整
        if text_x < 0:
            text_x = 0
        
        # 右端のチェックと調整
        if text_x + text_width > img_width:
            text_x = img_width - text_width
            
        # 上端のチェックと調整
        if text_y < 0:
            text_y = 0
            
        # 下端のチェックと調整
        if text_y + text_height > img_height:
            text_y = img_height - text_height
            
        # テキスト本体を描画
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

        # ----- 4. 最終画像を保存 -----
        final_output_filename = r"example/final_result_with_answer.png"
        generated_pil_image.save(final_output_filename)
        print(f"🎉 完成！解答テキスト付き画像を '{final_output_filename}' として保存しました。")