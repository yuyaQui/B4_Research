import csv

def load_quizzes(quiz_csv_path: str) -> list[tuple[str, str]]:
    quizes = []
    with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question = row[0]
            answer = row[1]
            dammy1 = row[2]
            dammy2 = row[3]
            dammy3 = row[4]
            quizes.append((question, answer, dammy1, dammy2, dammy3))
    return quizes