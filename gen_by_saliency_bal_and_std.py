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

# generate_image_from_quiz 関数を修正
# 保存パスを引数 (output_image_path) として受け取るように変更
def generate_image_from_quiz(output_image_path):
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
            - イラスト内には、いかなる文字、単語、数字を含ないでください。
            - イラスト内には、英語を配置しないでください。
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
                    
                    # ▼▼▼【変更点】▼▼▼
                    # ハードコーディングされていたパスを、引数で受け取ったパスに変更
                    output_filename = output_image_path
                    # ▲▲▲【変更点】▲▲▲

                    # 保存先ディレクトリが存在するか確認し、なければ作成
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    
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

    # ==========================================================
    # ⚙️ 設定（ハードコーディングされた変数）
    # ==========================================================
    
    # --- 1. パス設定 ---
    OUTPUT_DIR = "example" # メインの出力先ディレクトリ
    
    # TranSalNetモデルのパス
    MODEL_PATH_RES = r'pretrained_models\TranSalNet_Res.pth'
    MODEL_PATH_DENSE = r'pretrained_models\TranSalNet_Dense.pth'

    # Windowsのフォントパス（環境に合わせて変更してください）
    FONT_PATH = "C:/Windows/Fonts/meiryob.ttc" 
    
    # --- 2. モデル選択 ---
    # 0 = TranSalNet_Dense, 1 = TranSalNet_Res
    MODEL_FLAG = 1 

    # --- 3. Saliencyピーク検出設定 ---
    # このSaliency値を超えるピクセルを「顕著なピーク」として考慮する
    PEAK_SALIENCY_THRESHOLD = 180
    
    # 検出するピークの最大数
    MAX_PEAKS_TO_FIND = 5
    MIN_PEAKS_TO_FIND = 2
    
    # ピーク周辺をマスクする際のSaliencyしきい値
    # このしきい値を下回る部分までの距離でマスク半径を決定
    MASK_SALIENCY_THRESHOLD_FOR_RADIUS = 100 
    
    # マスク半径の最小値（ピーク周辺を確実にマスクするため）
    MIN_MASK_RADIUS = 50

    POS_OFFSET = 20

    # --- 4. テキスト描画設定 ---
    FONT_SIZE = 36
    FILL_COLOR = "#00ff00" # テキスト本体の色 (緑)
    STROKE_COLOR = "black"  # 縁取りの色 (黒)
    STROKE_WIDTH = 3        # 縁取りの太さ

    # ==========================================================
    # 処理開始
    # ==========================================================

    # 設定からファイルパスを構築
    generated_image_path = os.path.join(OUTPUT_DIR, "generated_image.png")
    saliency_output_filename = os.path.join(OUTPUT_DIR, "result_saliency.png")
    final_output_filename = os.path.join(OUTPUT_DIR, "final_result_with_answer.png")

    # 1. 画像生成（引数に保存パスを渡す）
    generated_pil_image, answer_text = generate_image_from_quiz(generated_image_path)

    if generated_pil_image is None:
        print("画像の生成に失敗したため、処理を終了します。")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"CUDA利用可能: {torch.cuda.is_available()}")

        if MODEL_FLAG == 1:
            from TranSalNet_Res import TranSalNet
            model = TranSalNet()
            model.load_state_dict(torch.load(MODEL_PATH_RES))
        else:
            from TranSalNet_Dense import TranSalNet
            model = TranSalNet()
            model.load_state_dict(torch.load(MODEL_PATH_DENSE))

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
        cv2.imwrite(saliency_output_filename, saliency_map_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"Saliency Mapを '{saliency_output_filename}' として保存しました。")
        
        # 元画像のサイズをここで取得
        img_width, img_height = generated_pil_image.size
        h, w = saliency_map_np.shape

        # ▼▼▼▼▼【ロジック修正箇所】▼▼▼▼▼
        # --- 2-6. 反復的なピーク検出とマスク処理 ---
        print(f"\n🔍 反復的なピーク検出を開始 (最小{MIN_PEAKS_TO_FIND}個、最大{MAX_PEAKS_TO_FIND}個、しきい値 > {PEAK_SALIENCY_THRESHOLD})")

        # ピークを格納するリスト
        all_x_peaks = []
        all_y_peaks = []
        all_vals_peaks = []

        # マスク処理用のSaliencyマップコピーを作成
        temp_map = saliency_map_np.copy()
        
        # 全座標 (Y, X) グリッドを事前に計算
        Y_grid, X_grid = np.ogrid[:h, :w]
        
        # しきい値を下回るピクセルの座標（マスク半径の計算用、これはループの外で1回だけ計算）
        y_low, x_low = np.where(saliency_map_np < MASK_SALIENCY_THRESHOLD_FOR_RADIUS)
        
        # MAX_PEAKS_TO_FIND の回数だけループ
        for i in range(MAX_PEAKS_TO_FIND):
            # 1. 現在のマップ (temp_map) から最大値（ピークN）を見つける
            val_n = np.max(temp_map)
            
            # 2. ピークのSaliency値がしきい値を下回ったら、ループ終了
            if val_n < PEAK_SALIENCY_THRESHOLD and i > MIN_PEAKS_TO_FIND - 1:
                print(f"  次のピークの値 ({val_n:.2f}) がしきい値 ({PEAK_SALIENCY_THRESHOLD}) を下回ったため、検出を終了。")
                break
                
            loc_n_flat = np.argmax(temp_map) 
            y_n, x_n = np.unravel_index(loc_n_flat, temp_map.shape)
            
            print(f"  ピーク{i+1}: (x={x_n}, y={y_n}) (値: {val_n:.2f}) を検出。")
            
            # 3. 見つかったピークをリストに追加
            all_x_peaks.append(x_n)
            all_y_peaks.append(y_n)
            all_vals_peaks.append(val_n)

            # 4. 【反復処理】ピークN (x_n, y_n) の周辺をマスクするための半径を計算
            mask_radius = MIN_MASK_RADIUS # デフォルトは最小半径
            if y_low.size > 0:
                # しきい値を下回る全ピクセルと、今見つけたピークN (x_n, y_n) との距離を計算
                distances_to_low = np.sqrt((x_low - x_n)**2 + (y_low - y_n)**2)
                calculated_radius = np.min(distances_to_low)
                # 最小半径と比較して大きい方を採用
                mask_radius = max(calculated_radius, MIN_MASK_RADIUS)
            else:
                # しきい値を下回るピクセルがない（画像全体が明るい）場合
                print(f"  (半径計算) Saliency < {MASK_SALIENCY_THRESHOLD_FOR_RADIUS} の領域なし。最小半径({MIN_MASK_RADIUS})を使用。")
        
            # 5. 【反復処理】temp_map 上のピークNの周辺をマスク（消去）する
            dist_from_peak_n = np.sqrt((X_grid - x_n)**2 + (Y_grid - y_n)**2)
            temp_map[dist_from_peak_n <= mask_radius] = 0
            # print(f"  ピーク{i+1}の周囲 (半径 {mask_radius:.2f}) をマスクしました。")

        # --- 2-7. 検出された全ピークのSaliency値で加重平均（内分点）を計算 ---

        # リストをNumpy配列に変換
        final_x_peaks = np.array(all_x_peaks)
        final_y_peaks = np.array(all_y_peaks)
        final_vals_peaks = np.array(all_vals_peaks)

        print("\n=== 最終的なテキスト配置の計算に使用するピーク ===")
        
        if final_x_peaks.size == 0:
            # ピークが一つも見つからなかった場合 (しきい値が高すぎるなど)
            print(f"⚠️ Saliency > {PEAK_SALIENCY_THRESHOLD} のピークが0個でした。Saliency最大値をピークとします。")
            val1 = np.max(saliency_map_np)
            loc1_flat = np.argmax(saliency_map_np)
            y1, x1 = np.unravel_index(loc1_flat, saliency_map_np.shape)
            
            final_x_peaks = np.array([x1])
            final_y_peaks = np.array([y1])
            final_vals_peaks = np.array([val1])

        for i in range(final_x_peaks.size):
            print(f"  ピーク{i+1}: (x={final_x_peaks[i]}, y={final_y_peaks[i]}) (値: {final_vals_peaks[i]:.2f})")

        # 重みの合計を計算 (オーバーフロー回避のためfloatで)
        total_weight = np.sum(final_vals_peaks.astype(float))

        if total_weight > 0:
            # x座標とy座標の加重平均を計算
            weighted_x_sum = np.sum(final_x_peaks.astype(float) * final_vals_peaks.astype(float))
            weighted_y_sum = np.sum(final_y_peaks.astype(float) * final_vals_peaks.astype(float))
            
            final_x = int(weighted_x_sum / total_weight)
            final_y = int(weighted_y_sum / total_weight)
            
            print(f"✅ {final_x_peaks.size}個のピークの加重平均を計算。")
        else:
            # フォールバック（通常は発生しないが、0除算を避ける）
            print(f"⚠️ 重みの合計が0です。画像の中心を使います。")
            final_x = img_width // 2
            final_y = img_height // 2
        # ▲▲▲▲▲【ここまでが変更箇所】▲▲▲▲▲
        
        print(f"💡 最適なテキスト配置座標 (x, y): ({final_x}, {final_y})")

        draw = ImageDraw.Draw(generated_pil_image)
        
        # フォントとテキストサイズの準備
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except IOError:
            print(f"警告: 指定されたフォント '{FONT_PATH}' が見つかりません。デフォルトフォントを使用します。")
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_half_width = text_width // 2
        text_half_height = text_height // 2

        left = final_x - POS_OFFSET
        top = final_y - POS_OFFSET
        right = final_x + POS_OFFSET
        bottom = final_y + POS_OFFSET

        factor_flag = float('inf')

        print(f"\n🔍 注目領域 ({left}, {top}) から ({right}, {bottom}) をスキャンし、最適なテキスト配置位置を探します...")

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

                    intensity_std = np.std(patch_np)

                    if (intensity_std < factor_flag):
                        factor_flag = intensity_std
                        final_x, final_y = i, j

        print(f"✅ スキャン完了。")
        print(f"💡 最適なテキスト配置座標 (x, y): ({final_x}, {final_y})")

        # ----- 3. 元画像に解答テキストを描画 -----
        
        # 最終的な座標 (final_x, final_y) を中心にテキストを描画
        text_x = final_x - text_half_width
        text_y = final_y - text_half_height
        
        # テキストが画像からはみ出さないように座標を最終調整
        if text_x < 0:
            text_x = 0
        if text_x + text_width > img_width:
            text_x = img_width - text_width
        if text_y < 0:
            text_y = 0
        if text_y + text_height > img_height:
            text_y = img_height - text_height
        
        draw.text(
            (text_x, text_y),
            answer_text,
            font=font, 
            fill=FILL_COLOR,
            stroke_width=STROKE_WIDTH,
            stroke_fill=STROKE_COLOR
        )

        # ----- 4. 最終画像を保存 -----
        generated_pil_image.save(final_output_filename)
        print(f"🎉 完成！解答テキスト付き画像を '{final_output_filename}' として保存しました。")