import csv

def load_quizzes(quiz_csv_path: str) -> list[tuple[str, str, str, str, str, str]]:
    quizes = []
    with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question_1 = row[0]
            target = row[1]
            question_2 = row[2]
            answer = row[3]
            dammy1 = row[4]
            dammy2 = row[5]
            dammy3 = row[6]
            quizes.append((question_1, target, question_2, answer, dammy1, dammy2, dammy3))
    return quizes