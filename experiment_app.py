import csv
from generate_image_from_gemini import generate_image_from_quiz

quiz_csv_path = "quiz.csv"
images = []
answers = []

with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        question = row[0]
        answer = row[1]
        image, answer = generate_image_from_quiz(question, answer)
        images.append(image)
        answers.append(answer)