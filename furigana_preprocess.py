import pickle
import os
from furigana_load_quizzes import load_quizzes
from experiment_generate_image import generate_image_from_quiz

# クイズの最大数
MAX_QUESTION_COUNT = 200
DATASETS_PATH = "experiment_datasets"
TARGET_PATH = "experiment_quiz"

if __name__ == "__main__":
    quizes = load_quizzes(os.path.join(DATASETS_PATH, f"{TARGET_PATH}.csv"))

    quizes_and_images = []
    print(f"画像生成を開始します... 最大 {MAX_QUESTION_COUNT} 問")
    for i, (question_1, question_1_read, target, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes):
        if i >= MAX_QUESTION_COUNT:
            break
        print(f"{i+1}問目を処理中...")  
        image = generate_image_from_quiz(question_1, target)
        if image is not None:
            quizes_and_images.append((question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3))

    # quizes_and_images を保存する処理を追加
    with open(os.path.join(DATASETS_PATH, f"{TARGET_PATH}_quizes_and_images.pkl"), "wb") as f:
        pickle.dump(quizes_and_images, f)

    print(f"全 {len(quizes_and_images)} 問のクイズについて画像を生成しました")
