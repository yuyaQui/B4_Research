import csv

def load_quizzes(quiz_csv_path: str) -> list[tuple[str, str]]:
    quizes = []
    with open(quiz_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            question = row[0]
            answer = row[1]
            quizes.append((question, answer))
    return quizes