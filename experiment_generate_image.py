from google import genai
from PIL import Image
from io import BytesIO
import os
import csv

import config

def generate_image_from_quiz(question: str, answer: str) -> Image.Image:
    try:
        api_key = getattr(config, 'GOOGLE_API_KEY', None)
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        prompt = (
            f"""
            あなたは画像を生成するAIです。
            あなたの唯一のタスクは、ユーザーから提供される情報を基にイラストを生成することです。
            出力は画像データのみとし、説明やテキストは一切出力しないでください。

            # 画像生成の厳格な制約（文字排除の徹底）
            - **文字・数字・記号の完全排除**: 画像内のいかなる場所（背景、看板、商品のラベル、衣服の柄、装飾、落書きなど）にも、文字、単語、数字、アルファベット、記号を一切描かないでください。
            - **無地のデザイン**: 本の表紙、看板、ポスター、画面などは、すべて無地、または幾何学模様や抽象的な絵のみで表現し、テキスト情報は一切排除してください。
            - **テキストの代替**: 概念や意味を伝える際は、文字ではなく「表情」「ポーズ」「象徴的なアイテム」「色」「状況」などの視覚的要素のみで表現してください。

            # 画像生成の指示
            - [問題文]と[解答]の内容を、言葉を一切介さず、視覚的な状況描写のみで忠実に表現してください。
            - 人間が画像を見ただけで、その状況や文脈を直感的に理解できるような構成を目指してください。
            - **画面構成**: 被写体を大きく描き、背景の空白を最小限に抑えてください。
            - **スタイル**: [問題文]と[解答]の内容に最も適したイラストスタイルを自動的に選択してください。
            - **ランダム性**: 重要なモチーフや特徴的な要素は、中央に固定せず、画面内のランダムな位置に配置してください。

            # ユーザー入力情報
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
                print("もう一度生成を行います")
                # テキスト応答があれば表示する（デバッグ用）
                if response.candidates[0].content.parts and response.candidates[0].content.parts[0].text:
                    print(f"モデルのテキスト応答: {response.candidates[0].content.parts[0].text}")
                return generate_image_from_quiz(question, answer)
        else:
            # response自体が無効だった場合
            print("❌ モデルから有効な応答が得られませんでした。")
            print("もう一度生成を行います")
            return generate_image_from_quiz(question, answer)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("もう一度生成を行います")
        return generate_image_from_quiz(question, answer)
