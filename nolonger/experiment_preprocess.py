import pickle
import os
from nolonger.experiment_load_quizes import load_quizzes
from experiment_generate_image import generate_image_from_quiz

# クイズの最大数
MAX_QUESTION_COUNT = 80
DATASETS_PATH = "experiment_datasets"
TARGET_PATH = "experiment_fixed_2"

if __name__ == "__main__":
    quizes = load_quizzes(os.path.join(DATASETS_PATH, f"{TARGET_PATH}.csv"))

    quizes_and_images = []
    for i, (question_1, target, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes):
        if i >= MAX_QUESTION_COUNT:
            break
        image = generate_image_from_quiz(question_1, target)
        if image is not None:
            quizes_and_images.append((question_1, target, image, question_2, answer, dammy1, dammy2, dammy3))

    # quizes_and_images を保存する処理を追加
    with open(os.path.join(DATASETS_PATH, f"{TARGET_PATH}_quizes_and_images.pkl"), "wb") as f:
        pickle.dump(quizes_and_images, f)

    print(f"全 {len(quizes_and_images)} 問のクイズについて画像を生成しました")
