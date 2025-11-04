from experiment_utils import load_quizzes_and_generate_images

quiz_csv_path = "quiz.csv"
images, quizes = load_quizzes_and_generate_images(quiz_csv_path)

# answers（解答）を表示し、ユーザーが知っているか確認
unknown_words = []
print("クイズの解答候補です。知っている単語には 'y'、知らない単語には 'n' を入力してください。")
for i, (question, answer) in enumerate(quizes):
    response = input(f"{i+1}. 解答: '{answer}' を知っていますか？ (y/n): ").strip().lower()
    if response == 'n':
        unknown_words.append((question, answer))

print("\nあなたが知らない単語とその問題文リスト：")
for idx, (question, answer) in enumerate(unknown_words):
    print(f"{idx+1}. 問題文: {question}, 解答: {answer}")
