import experiment_utils
import random

# クイズの最大数
MAX_QUESTION_COUNT = 20

utils = experiment_utils.ExperimentUtils()

quiz_csv_path = "quiz.csv"
quizes = utils.load_quizzes(quiz_csv_path)

quize_and_images = []
for i, (question, answer) in enumerate(quizes):
    if i >= MAX_QUESTION_COUNT:
        break
    image = utils.generate_image_from_quiz(question, answer)
    if image is not None:
        quize_and_images.append((question, answer, image))

unknown_quizes_and_images = utils.ask_unknown_words(quize_and_images)

random.shuffle(unknown_quizes_and_images)

for question, answer, image in unknown_quizes_and_images:
    print(question)
    print(answer)
    image.show()