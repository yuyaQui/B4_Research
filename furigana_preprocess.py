import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from furigana_load_quizzes import load_quizzes
from experiment_generate_image import generate_image_from_quiz

# 日本語フォント設定
rcParams['font.family'] = 'MS Gothic'

# クイズの最大数
MAX_QUESTION_COUNT = 200
DATASETS_PATH = "experiment_datasets"
TARGET_PATH = "experiment_quiz"

def show_image_interactive(image, question_num, question_text, answer_text):
    """
    画像をインタラクティブに表示し、キーボード入力を受け付ける
    
    Returns:
        str: 'y' (承認), 'n' (再生成), 'q' (終了)
    """
    user_choice = {'value': None}
    
    def on_key(event):
        if event.key in ['y', 'n', 'q']:
            user_choice['value'] = event.key
            plt.close()
    
    # 図を作成
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')
    
    # タイトルと説明を追加
    title_text = f"問題 {question_num}\n問題文: {question_text[:50]}...\n解答: {answer_text}"
    fig.suptitle(title_text, fontsize=12, fontweight='bold', y=0.98)
    
    # 操作説明を画像下部に追加
    instruction_text = (
        "【キーボード操作】\n"
        "Y キー: この画像を採用して次へ\n"
        "N キー: 画像を再生成\n"
        "Q キー: 処理を中断"
    )
    fig.text(0.5, 0.02, instruction_text, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # キーイベントを接続
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return user_choice['value']


if __name__ == "__main__":
    quizes = load_quizzes(os.path.join(DATASETS_PATH, f"{TARGET_PATH}.csv"))

    quizes_and_images = []
    start_index = 0  # 開始位置
    
    # 既存のファイルをチェック
    final_path = os.path.join(DATASETS_PATH, f"{TARGET_PATH}_quizes_and_images.pkl")
    partial_path = os.path.join(DATASETS_PATH, f"{TARGET_PATH}_quizes_and_images_partial.pkl")
    
    # 途中経過ファイルまたは完成ファイルが存在する場合
    resume_file = None
    if os.path.exists(partial_path):
        resume_file = partial_path
    elif os.path.exists(final_path):
        resume_file = final_path
    
    if resume_file:
        print(f"\n既存のファイルが見つかりました: {resume_file}")
        print("続きから処理を再開しますか？")
        print("  'y': 続きから再開")
        print("  'n': 最初からやり直す（既存ファイルは上書きされます）")
        
        choice = input().strip().lower()
        if choice == 'y':
            # 既存データを読み込み
            with open(resume_file, "rb") as f:
                quizes_and_images = pickle.load(f)
            
            start_index = len(quizes_and_images)
            print(f"\n✓ {start_index}問まで完了しています。{start_index + 1}問目から再開します。")
        else:
            print("\n最初から処理を開始します。")
            quizes_and_images = []
            start_index = 0
    
    print(f"\n画像生成を開始します... {start_index + 1}問目 ～ 最大 {MAX_QUESTION_COUNT} 問")
    
    for i, (question_1, question_1_read, target, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes):
        # 既に処理済みの問題はスキップ
        if i < start_index:
            continue
        
        if i >= MAX_QUESTION_COUNT:
            break
            
        print(f"\n{'='*60}")
        print(f"{i+1}問目を処理中... (進捗: {i+1}/{min(MAX_QUESTION_COUNT, len(quizes))})")
        print(f"問題：{question_1}")
        print(f"解答：{target}")
        print('='*60)
        
        # ユーザーが承認するまで画像生成を繰り返す
        while True:
            image = generate_image_from_quiz(question_1, target)
            if image is not None:
                # インタラクティブな画像表示（キーボード操作）
                print("\n画像を表示しています... 画像ウィンドウで Y/N/Q キーを押してください")
                selected = show_image_interactive(image, i+1, question_1, target)
                
                if selected == 'y':
                    quizes_and_images.append((question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3))
                    print(f"✓ {i+1}問目を保存しました。")
                    
                    # 定期的に途中経過を保存（5問ごと）
                    if (i + 1) % 5 == 0:
                        with open(partial_path, "wb") as f:
                            pickle.dump(quizes_and_images, f)
                        print(f"💾 途中経過を自動保存しました ({len(quizes_and_images)}問)")
                    
                    break  # whileループを抜けて次の問題へ
                    
                elif selected == 'n':
                    print("🔄 画像を再生成します...")
                    continue  # whileループの最初に戻って再生成
                    
                elif selected == 'q':
                    print("\n⏸️ 処理を中断します。")
                    # 現在までの結果を保存
                    if quizes_and_images:
                        with open(partial_path, "wb") as f:
                            pickle.dump(quizes_and_images, f)
                        print(f"💾 途中経過を保存しました: {len(quizes_and_images)} 問")
                        print(f"📁 保存先: {partial_path}")
                        print("\n次回実行時に続きから再開できます。")
                    exit(0)
                    
                else:
                    # ウィンドウを閉じた場合など
                    print("⚠️ 入力が認識されませんでした。もう一度画像を表示します。")
                    continue
            else:
                # 最大リトライ回数に達して None が返ってきた場合
                print("\n⚠️ 画像生成が最大リトライ回数に達しました。")
                print("  's': この問題をスキップ")
                print("  'r': それでも再試行する")
                print("  'q': 処理を中断")
                
                choice = input(">>> ").strip().lower()
                if choice == 's':
                    print(f"⏭️ 問題 {i+1} をスキップします。")
                    break  # この問題をスキップして次へ
                elif choice == 'r':
                    print("🔄 再試行します...")
                    continue  # もう一度試す
                elif choice == 'q':
                    print("\n処理を中断します。")
                    # 現在までの結果を保存
                    if quizes_and_images:
                        with open(partial_path, "wb") as f:
                            pickle.dump(quizes_and_images, f)
                        print(f"💾 途中経過を保存しました: {len(quizes_and_images)} 問")
                        print(f"📁 保存先: {partial_path}")
                        print("\n次回実行時に続きから再開できます。")
                    exit(0)
                else:
                    print("'s', 'r', または 'q' を入力してください")
                    continue

    # 全て完了したら最終ファイルとして保存
    print(f"\n{'='*60}")
    print(f"✅ 全 {len(quizes_and_images)} 問のクイズについて画像を生成しました")
    print('='*60)
    
    with open(final_path, "wb") as f:
        pickle.dump(quizes_and_images, f)
    print(f"💾 最終結果を保存: {final_path}")
    
    # 途中経過ファイルがあれば削除
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"🗑️ 途中経過ファイルを削除しました")

