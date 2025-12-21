def ask_unknown_words(quizes_and_images, max_count=20):
    """
    quizes_and_images: list of (question, answer, image) tuples
    max_count: number of items to ask
    return: list of (question, answer, image) tuples where user didn't know the answer
    """
    unknown_quizes_and_images = []

    print("クイズの解答候補です。知っている単語には 'y'、知らない単語には 'n' を入力してください。")
    for i, (question, answer, image) in enumerate(quizes_and_images):
        if i >= max_count:
            break
        response = input(f"{i+1}. 解答: '{answer}' を知っていますか？ (y/n): ").strip().lower()
        while(True):
            if response == 'n':
                unknown_quizes_and_images.append((question, answer, image))
                break
            elif response == 'y':
                break
            else:
                response = input(f"{i+1}. 解答: '{answer}' を知っていますか？ (y/n): ").strip().lower()
    return unknown_quizes_and_images