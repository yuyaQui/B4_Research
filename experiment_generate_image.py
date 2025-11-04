from google import genai
from PIL import Image
from io import BytesIO
import os
import csv

def generate_image_from_quiz(question: str, answer: str) -> Image.Image:
    try:
        client = genai.Client()

        prompt = (
            f"""
            あなたは画像を生成するAIです。
            あなたの唯一のタスクは、ユーザーから提供される情報を基にイラストを生成することです。
            あなたの出力は画像データでなければなりません。説明、文章、その他のテキストは一切出力しないでください。

            # 画像生成の最優先指示
            - 生成するイラスト内には、**いかなる種類の文字、単語、数字、記号を絶対に含めないでください。**
            - イラストとしての文字、単語、数字、記号を絶対に生成しないでください。
            - テキストは描画しないでください。
            - タイポグラフィは含めないでください。
            - レターは含めないでください。

            # 画像生成の追加指示
            - ユーザーから提供される[問題文]と[解答]の内容を言葉を使わずに忠実に表現したイラストを生成してください。
            - [問題文]の文脈を正しく読み取り、画像だけで内容を人間が簡単に理解できるような画像の生成を目標としてください。
            - イラストは、被写体が大きく描かれ、背景の空白が少なくなるように構成してください。
            - スタイルは、[問題文]と[解答]の内容に最も適したものを適宜判断してください。

            # ユーザー入力形式
            ユーザーは以下の形式で[問題文]と[解答]を提供します。
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
