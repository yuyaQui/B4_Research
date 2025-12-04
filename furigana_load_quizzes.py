import csv

def load_quizzes(quiz_csv_path: str) -> list[tuple[str, str, str, str, str, str, str, str]]:
    quizes = []
    with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question_1 = row[0]
            question_1_read = row[1]
            target = row[2]
            question_2 = row[3]
            answer = row[4]
            dammy1 = row[5]
            dammy2 = row[6]
            dammy3 = row[7]
            quizes.append((question_1, question_1_read, target, question_2, answer, dammy1, dammy2, dammy3))
    return quizes