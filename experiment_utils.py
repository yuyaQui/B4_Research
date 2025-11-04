from google import genai
from PIL import Image
from io import BytesIO
import os
import csv

def generate_image_from_quiz(question: str, answer: str) -> Image.Image:
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
            - イラストとして文字を生成しないでください。
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
                    image_found = True
                    return image

            # ループが終了しても画像が見つからなかった場合
            if not image_found:
                print("❌ 応答に画像データが含まれていませんでした。")
                # テキスト応答があれば表示する（デバッグ用）
                if response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
                    print(f"モデルのテキスト応答: {response.candidates[0].content.parts[0].text}")
                return None
        else:
            # response自体が無効だった場合
            print("❌ モデルから有効な応答が得られませんでした。")
            return None

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def load_quizzes_and_generate_images(quiz_csv_path: str) -> tuple[list[Image.Image], list[tuple[str, str]]]:
    images = []
    quizes = []
    with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question = row[0]
            answer = row[1]
            image = generate_image_from_quiz(question, answer)
            if image is not None:
                images.append(image)
                quizes.append((question, answer))
            else:
                print(f"画像の生成に失敗しました: {question}, {answer}")
                continue
    return images, quizes