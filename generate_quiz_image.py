from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
from io import BytesIO
import torch
import cv2
import numpy as np
from torchvision import transforms, utils, models
from utils.data_process import preprocess_img, postprocess_img
from PIL import Image


def generate_image_from_quiz():
    try:
        client = genai.Client()

        print("クイズの情報を入力してください。")
        question = input("問題文: ")
        answer = input("解答: ")

        prompt = (
            f"""
            あなたは画像を生成するAIです。あなたの唯一のタスクは、以下の情報を基にイラストを生成することです。
            あなたの出力は画像データでなければなりません。説明、文章、その他のテキストは一切出力しないでください。

            # 指示
            - 以下の[問題文]と[解答]の内容を忠実に表現したイラストを生成してください。
            - [問題文]の文脈を正しく読み取り、文章がなくても画像だけで内容を人間が簡単に理解できるような画像の生成を目標としてください。
            - イラスト内には、いかなる文字、単語、数字も含まないでください。
            - イラストは、被写体が大きく描かれ、背景の空白が少なくなるように構成してください。
            - スタイル: 適したスタインを適宜判断

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


        for part in response.candidates[0].content.parts:
            # 回答の１番目の候補の内容の要素のクラスを抽出している
            if part.text is not None:
                print(part.text)
            if part.inline_data is not None:
                # 画像のバイナリデータが存在している場合
                image = Image.open(BytesIO(part.inline_data.data))
                output_filename = "generated_image.png"
                image.save(output_filename)
                print(f"Geminiが生成した画像を'{output_filename}'として保存しました")
                return image, answer

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None, None

if __name__ == "__main__":

    generated_pil_image, answer_text = generate_image_from_quiz()

    if generated_pil_image is None:
        print("画像の生成に失敗したため、処理を終了します。")
    else:
        generated_image_path = "generated_image.png"
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

        model = model.to(device) 
        model.eval()

        img = preprocess_img(generated_image_path) # padding and resizing input image into 384x288
        img = np.array(img)/255.
        img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor).to(device)
        pred_saliency_tensor = model(img)
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency_tensor.squeeze())

        saliency_map_np = postprocess_img(pic, generated_image_path)
        saliency_output_filename = r'example/result_saliency.png'
        cv2.imwrite(saliency_output_filename, saliency_map_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"Saliency Mapを '{saliency_output_filename}' として保存しました。")

        # ----- 2. 最も注視される場所の座標を取得 -----
        # saliency_map_np (NumPy配列) の中で最も値が大きい場所のインデックス(y, x)を見つける
        max_loc_flat = np.argmax(saliency_map_np) #1次元配列として考えた時に最大値がある場所を示す
        max_y, max_x = np.unravel_index(max_loc_flat, saliency_map_np.shape) # 2次元配列の時のインデックスを返す
        print(f"💡 最も注視される座標 (x, y): ({max_x}, {max_y})")

        # ----- 3. 元画像に解答テキストを描画 -----
        # フォント設定（！要変更！）
        # お使いの環境に合わせて日本語フォントのパスを指定してください
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        font_size = 28
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 指定されたフォント '{font_path}' が見つかりません。デフォルトフォントを使用します。")
            font = ImageFont.load_default()

        # 描画オブジェクトを作成
        draw = ImageDraw.Draw(generated_pil_image) #generated_pil_image は Gemini が出力した画像

        # テキストの描画サイズを取得して中央揃えのための位置を計算
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0] # テキストの右下のx座標から左上のx座標を引く
        text_height = text_bbox[3] - text_bbox[1] # テキストの右下のy座標から左上のy座標を引く
        offset_height = 6
        text_x = max_x - (text_width / 2)
        text_y = max_y - (text_height / 2) + offset_height

        # テキストに影をつけて見やすくする
        shadow_color = "black"
        offset = 1
        draw.text((text_x - offset, text_y - offset), answer_text, font=font, fill=shadow_color)
        draw.text((text_x + offset, text_y - offset), answer_text, font=font, fill=shadow_color)
        draw.text((text_x - offset, text_y + offset), answer_text, font=font, fill=shadow_color)
        draw.text((text_x + offset, text_y + offset), answer_text, font=font, fill=shadow_color)

        # テキスト本体を描画
        text_color = "white"
        draw.text((text_x, text_y), answer_text, font=font, fill=text_color)

        # ----- 4. 最終画像を保存 -----
        final_output_filename = r"example/final_result_with_answer.png"
        generated_pil_image.save(final_output_filename)
        print(f"🎉 完成！解答テキスト付き画像を '{final_output_filename}' として保存しました。")