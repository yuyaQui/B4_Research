import os
import pickle
import random
import torch
import streamlit as st
from PIL import Image
import pyttsx3
# TranSalNet_Dense, experiment_preprocess, experiment_image_draw は
# 同じディレクトリにあると仮定します
from TranSalNet_Dense import TranSalNet
from experiment_preprocess import DATASETS_PATH
from experiment_image_draw import find_optimal_text_position, find_lower_text_position_and_draw, draw_answer_text_on_image

MODEL_PATH_DENSE = r'pretrained_models\TranSalNet_Dense.pth'
NUM_TO_OPTIMIZE = 30 # 各パターンで処理する最大数
READING_SPEED = 150

# --- セッション状態の初期化 ---
if 'experiment_set' not in st.session_state:
    try:
        with open(os.path.join(DATASETS_PATH, "quizes_and_images.pkl"), "rb") as f:
            st.session_state.experiment_set = pickle.load(f)
            
            # --- ▼ ターミナル出力（読み込みクイズ総数） ▼ ---
            try:
                total_loaded = len(st.session_state.experiment_set)
                print(f"\n--- [初期読み込み] quizes_and_images.pkl から {total_loaded} 問のクイズを読み込みました ---")
            except Exception as e:
                print(f"クイズ総数の出力中にエラー: {e}")
            # --- ▲ ターミナル出力（ここまで） ▲ ---
            
    except FileNotFoundError:
        st.error(f"データファイルが見つかりません: {os.path.join(DATASETS_PATH, 'quizes_and_images.pkl')}")
        st.session_state.experiment_set = [] # エラー時に空リストをセット
    except Exception as e:
        st.error(f"データファイルの読み込み中にエラーが発生しました: {e}")
        st.session_state.experiment_set = []

# 変更: 未知語リストを前半(part1)と後半(part2)に
if 'unknown_quizes_part1' not in st.session_state:
    st.session_state.unknown_quizes_part1 = []
    st.session_state.unknown_quizes_part2 = []
    st.session_state.current_quiz_index = 0
    st.session_state.quiz_selection_done = False

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.device = None

# 変更: 処理済み画像リストも前半(p1)と後半(p2)に
if 'processed_images_p1' not in st.session_state:
    st.session_state.processed_images_p1 = []
if 'processed_images_p2' not in st.session_state:
    st.session_state.processed_images_p2 = []


# --- 関数定義 ---
def read_text(text: str):
    """テキストを読み上げる"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', READING_SPEED)
        engine.say(text)
        engine.runAndWait()
        engine.stop() # 念のため停止処理
    except Exception as e:
        st.warning(f"音声読み上げエラー: {e}")

def load_model():
    """モデルを読み込む（初回のみ）"""
    if st.session_state.model is None:
        with st.spinner("モデルを読み込んでいます..."):
            try:
                st.session_state.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = TranSalNet()
                model.load_state_dict(torch.load(MODEL_PATH_DENSE, map_location=st.session_state.device))
                model = model.to(st.session_state.device)
                model.eval()
                st.session_state.model = model
            except FileNotFoundError:
                st.error(f"モデルファイルが見つかりません: {MODEL_PATH_DENSE}")
            except Exception as e:
                st.error(f"モデル読み込み中にエラーが発生しました: {e}")

def ask_unknown_words_ui(quizes_and_images, max_count=20):
    """
    未知語選択UI（完了ボタン付き）。
    変更: (unknown_part1, unknown_part2, completed) を返す。
    unknown_part1/2 には (question, answer, image, original_index) が含まれる。
    """
    st.header("📝 クイズの解答候補")
    st.write("知っている単語には 'はい'、知らない単語には 'いいえ' を選択してください。")
    
    # ラジオボタンを表示
    for i, (question, answer, image, _, _, _) in enumerate(quizes_and_images):
        if i >= max_count:
            break

        with st.container():
            st.write(f"**{i+1}. '{answer}'**")
            st.radio(
                "知っていますか？",
                ["はい", "いいえ"],
                key=f"quiz_{i}", # セッション状態に直接保存
                horizontal=True,
                index=None # デフォルトは未選択
            )

    # 回答状況をセッション状態から集計
    responses = []
    for i in range(max_count): 
        if f"quiz_{i}" in st.session_state and st.session_state[f"quiz_{i}"] is not None:
            responses.append(st.session_state[f"quiz_{i}"])
            print(st.session_state[f"quiz_{i}"])
    
    all_answered = len(responses) == max_count

    if not all_answered:
        remaining = max_count - len(responses)
        st.info(f"すべての解答を選択してください。（残り {remaining} 問）")
    else:
        st.success("すべての解答が選択されました。「選択を完了」ボタンを押してください。")

    # 完了ボタン
    if st.button("選択を完了", key="complete_selection"):
        if all_answered:
            # 変更: 未知語を前半と後半に振り分ける
            unknown_part1 = []
            unknown_part2 = []
            mid_point = max_count // 2 # 表示したクイズの中間点

            for i, (question, answer, image, dammy1, dammy2, dammy3) in enumerate(quizes_and_images[:max_count]):
                if st.session_state[f"quiz_{i}"] == "いいえ":
                    # 変更: 元のインデックス i もタプルに含める
                    quiz_data = (question, answer, dammy1, dammy2, dammy3, image, i) 
                    if i < mid_point: # 前半グループ
                        unknown_part1.append(quiz_data)
                    else: # 後半グループ
                        unknown_part2.append(quiz_data)
                        
            return unknown_part1, unknown_part2, True 
        else:
            st.error("まだすべての設問に回答していません。")
            return [], [], False
    
    return [], [], False

# --- メインUI ---
# 146行目を変更
# --- メインUI ---
# ▼▼▼ 変更点: タブを6つに増やす ▼▼▼
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "クイズ選択", 
    "画像処理", 
    "パターン1 (Saliency)", 
    "パターン2 (下部固定)",
    "パターン1 クイズ", # 追加
    "パターン2 クイズ"  # 追加
])
# ▲▲▲ 変更点: ここまで ▲▲▲

with tab1:    
    max_quizzes = st.number_input(
        "最大クイズ数（前半と後半に均等に分割されます）", 
        min_value=2, # 最低2問（各1問）
        max_value=1000, 
        value=80, 
        step=1, 
        key="max_quizzes"
    )

    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'max_quizzes_on_start' not in st.session_state:
        st.session_state.max_quizzes_on_start = 20

    if st.button("クイズを開始", key="start_quiz"):
        st.session_state.quiz_started = True
        # 変更: part1 と part2 をリセット
        st.session_state.unknown_quizes_part1 = []
        st.session_state.unknown_quizes_part2 = []
        st.session_state.quiz_selection_done = False
        st.session_state.processed_images_p1 = [] # 処理済みもリセット
        st.session_state.processed_images_p2 = [] # 処理済みもリセット

        st.session_state.p1_quiz_started = False
        st.session_state.p2_quiz_started = False
        st.session_state.p1_quiz_idx = 0
        st.session_state.p2_quiz_idx = 0
        
        max_to_reset = max(50, st.session_state.max_quizzes_on_start) 
        for i in range(max_to_reset): 
            if f"quiz_{i}" in st.session_state:
                del st.session_state[f"quiz_{i}"]
                
        st.session_state.max_quizzes_on_start = int(max_quizzes)
        
        # --- ▼ ターミナル出力（タブ1で出題されなかった問題） ▼ ---
        try:
            total_quizzes_in_set = len(st.session_state.experiment_set)
            num_presented = st.session_state.max_quizzes_on_start
            
            if total_quizzes_in_set > num_presented:
                # max_quizzes_on_start から最後までが「出題されなかった」インデックス
                unpresented_indices = list(range(num_presented, total_quizzes_in_set))
                print("\n--- [タブ1]で出題されなかった問題の番号 (インデックス) ---")
                print(f"（{num_presented+1}番目 から {total_quizzes_in_set}番目 まで）")
                print(unpresented_indices + 1)
                print(f"合計: {len(unpresented_indices)} 問")
                print("------------------------------------------------------\n")
            else:
                print("\n--- [タブ1] すべての問題が出題対象となりました ---")
        except Exception as e:
            print(f"ターミナル出力中にエラーが発生しました: {e}")
        # --- ▲ ターミナル出力（ここまで） ▲ ---

        st.rerun() 

    if st.session_state.quiz_started and not st.session_state.quiz_selection_done:
        # 変更: 戻り値を3つ受け取る
        unknown_p1, unknown_p2, completed = ask_unknown_words_ui(
            st.session_state.experiment_set, 
            max_count=st.session_state.max_quizzes_on_start
        )
        
        if completed:
            # 変更: p1 と p2 をセッションに保存
            st.session_state.unknown_quizes_part1 = unknown_p1
            st.session_state.unknown_quizes_part2 = unknown_p2
            
            # 変更: それぞれをシャッフル
            random.shuffle(st.session_state.unknown_quizes_part1)
            random.shuffle(st.session_state.unknown_quizes_part2)
            
            st.session_state.quiz_selection_done = True
            st.session_state.quiz_started = False
            
            # 変更: メッセージを更新
            st.success(f"前半 {len(st.session_state.unknown_quizes_part1)}個, "
                       f"後半 {len(st.session_state.unknown_quizes_part2)}個 の未知の単語が見つかりました！")
            st.rerun() 
            
    if st.session_state.quiz_selection_done:
        # 変更: メッセージを更新
        st.info(f"✅ 前半 {len(st.session_state.unknown_quizes_part1)}個, "
                f"後半 {len(st.session_state.unknown_quizes_part2)}個 の未知の単語が選択されました。")

with tab2:    
    if not st.session_state.quiz_selection_done:
        st.warning("まず「クイズ選択」タブで未知の単語を選択してください。")
    elif not st.session_state.experiment_set:
         st.warning("データセットが読み込まれていません。")
    # 変更: 未知語が両方ゼロでないかチェック
    elif not st.session_state.unknown_quizes_part1 and not st.session_state.unknown_quizes_part2:
        st.warning("処理対象の未知の単語がありません。")
    else:
        # 変更: num_to_process の number_input は削除
        st.info(f"パターン1 (Saliency) は最大 {NUM_TO_OPTIMIZE} 問、\n"
                f"パターン2 (下部固定) は最大 {NUM_TO_OPTIMIZE} 問を処理します。")
        
        if st.button("画像処理を開始", key="process_images"):
            load_model()
            # モデル読み込み失敗時は処理中断
            if st.session_state.model is None:
                st.error("モデルが読み込まれていないため、処理を中断しました。")
            else:
                # 変更: p1 と p2 のリストをリセット
                st.session_state.processed_images_p1 = []
                st.session_state.processed_images_p2 = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                # --- ▼ ターミナル出力（タブ3, 4で出題されなかった問題） ▼ ---
                quizes_p1 = st.session_state.unknown_quizes_part1
                total_p1 = min(len(quizes_p1), NUM_TO_OPTIMIZE)
                
                quizes_p2 = st.session_state.unknown_quizes_part2
                total_p2 = min(len(quizes_p2), NUM_TO_OPTIMIZE)

                try:
                    # タブ3で出題されなかった問題 (part1 の NUM_TO_OPTIMIZE 以降)
                    # quiz_data[3] は元のインデックス
                    unpresented_p1_indices = [quiz_data[6] for quiz_data in quizes_p1[total_p1:]]
                    if unpresented_p1_indices:
                        print("\n--- [タブ3]で出題されなかった問題の番号 (元のインデックス) ---")
                        print(unpresented_p1_indices + 1)
                        print(f"合計: {len(unpresented_p1_indices)} 問")
                        print("----------------------------------------------------------\n")
                    else:
                        print("\n--- [タブ3] すべての未知語が処理対象となりました ---")

                    # タブ4で出題されなかった問題 (part2 の NUM_TO_OPTIMIZE 以降)
                    # quiz_data[3] は元のインデックス
                    unpresented_p2_indices = [quiz_data[6] for quiz_data in quizes_p2[total_p2:]]
                    if unpresented_p2_indices:
                        print("\n--- [タブ4]で出題されなかった問題の番号 (元のインデックス) ---")
                        print(unpresented_p2_indices + 1)
                        print(f"合計: {len(unpresented_p2_indices)} 問")
                        print("----------------------------------------------------------\n")
                    else:
                        print("\n--- [タブ4] すべての未知語が処理対象となりました ---")
                        
                except Exception as e:
                    print(f"ターミナル出力中にエラーが発生しました: {e}")
                # --- ▲ ターミナル出力（ここまで） ▲ ---

                
                # --- パターン1 (Saliency) の処理 ---
                if total_p1 > 0:
                    status_text.text(f"パターン1 (Saliency) 処理中: 0/{total_p1}")
                    for i in range(total_p1):
                        status_text.text(f"パターン1 (Saliency) 処理中: {i+1}/{total_p1}")
                        progress_bar.progress((i + 1) / total_p1)
                        
                        # 変更: original_index をアンパック
                        question, answer, dammy1, dammy2, dammy3, generated_image, original_index = quizes_p1[i]
                        try:
                            # image_data が PIL.Image オブジェクトかチェック
                            if isinstance(generated_image, Image.Image):
                                generated_image_pil = generated_image
                            # image_data が string (パスの可能性) かチェック
                            elif isinstance(generated_image, str):
                                # 文字列の場合は画像パスとして開く
                                if not os.path.exists(generated_image):
                                    st.error(f"P1: 画像パスが見つかりません: {generated_image} [Index: {original_index}]")
                                    continue # このクイズをスキップ
                                generated_image_pil = Image.open(generated_image)
                            else:
                                # 予期しない型
                                st.error(f"P1: 予期しない画像データ型: {type(generated_image)} [Index: {original_index}]")
                                continue # スキップ
                                
                            # PILオブジェクトのコピーを作成
                            image_copy = generated_image_pil.copy()

                        except Exception as e:
                            st.error(f"P1: 画像 {i+1} ('{answer}') [Index: {original_index}] の読み込み/コピー中にエラー: {e}")
                            continue # スキップ
                        
                        try:
                            x, y = find_optimal_text_position(
                                image_copy, 
                                st.session_state.model, 
                                st.session_state.device
                            )
                            image_with_caption = draw_answer_text_on_image(
                                image_copy, 
                                answer, 
                                x, 
                                y
                            )
                            
                            st.session_state.processed_images_p1.append({
                                'question': question,
                                'answer': answer,
                                'dammy1': dammy1, # 追加
                                'dammy2': dammy2, # 追加
                                'dammy3': dammy3, # 追加
                                'original_image': generated_image_pil, # 修正: PILオブジェクト
                                'processed_image': image_with_caption,
                                'position': (x, y),
                                'original_index': original_index
                            })
                            # ▲▲▲ 変更: ここまで ▲▲▲
                        except Exception as e:
                            st.error(f"パターン1の画像 {i+1} ('{answer}') [Index: {original_index}] の処理中にエラーが発生しました: {e}")

                # --- パターン2 (下部固定) の処理 ---
                if total_p2 > 0:
                    status_text.text(f"パターン2 (下部固定) 処理中: 0/{total_p2}")
                    progress_bar.progress(0) # プログレスバーリセット

                    for i in range(total_p2):
                        status_text.text(f"パターン2 (下部固定) 処理中: {i+1}/{total_p2}")
                        progress_bar.progress((i + 1) / total_p2)
                        
                        # 変更: original_index をアンパック
                        question, answer, dammy1, dammy2, dammy3, generated_image, original_index = quizes_p2[i]
                        try:
                            if isinstance(generated_image, Image.Image):
                                generated_image_pil = generated_image
                            elif isinstance(generated_image, str):
                                if not os.path.exists(generated_image):
                                    st.error(f"P2: 画像パスが見つかりません: {generated_image} [Index: {original_index}]")
                                    continue
                                generated_image_pil = Image.open(generated_image)
                            else:
                                st.error(f"P2: 予期しない画像データ型: {type(generated_image)} [Index: {original_index}]")
                                continue
                                
                            image_copy = generated_image_pil.copy()

                        except Exception as e:
                            st.error(f"P2: 画像 {i+1} ('{answer}') [Index: {original_index}] の読み込み/コピー中にエラー: {e}")
                            continue
                        
                        try:
                            image_with_caption = find_lower_text_position_and_draw(
                                image_copy, answer
                            )
                            img_width, img_height = image_with_caption.size
                            x, y = img_width // 2, img_height // 2 # 参考座標
                            
                            st.session_state.processed_images_p2.append({
                                'question': question,
                                'answer': answer,
                                'dammy1': dammy1, # 追加
                                'dammy2': dammy2, # 追加
                                'dammy3': dammy3, # 追加
                                'original_image': generated_image_pil, # 修正: PILオブジェクト
                                'processed_image': image_with_caption,
                                'position': (x, y),
                                'original_index': original_index
                            })
                        except Exception as e:
                            st.error(f"パターン2の画像 {i+1} ('{answer}') [Index: {original_index}] の処理中にエラーが発生しました: {e}")

                progress_bar.empty()
                status_text.text("処理完了！")
                st.success(f"パターン1 (Saliency): {len(st.session_state.processed_images_p1)}個, "
                           f"パターン2 (下部固定): {len(st.session_state.processed_images_p2)}個 の画像を処理しました。")

with tab3:
    if 'pattern1_started' not in st.session_state:
        st.session_state.pattern1_started = False
    if 'pattern1_idx' not in st.session_state:
        st.session_state.pattern1_idx = 0

    # 変更: processed_images_p1 をチェック
    if not st.session_state.processed_images_p1:
        st.info("「画像処理」タブでパターン1の画像を処理してください。")
    elif not st.session_state.pattern1_started:
        if st.button("学習を開始", key="pattern1_start"):
            idx_start = 0
            st.session_state.pattern1_idx = idx_start
            st.session_state.pattern1_started = True
            st.rerun()
    else:
        curr_idx = st.session_state.pattern1_idx
        
        # 変更: ループ条件を processed_images_p1 の長さに
        if curr_idx < len(st.session_state.processed_images_p1):
            
            if st.button("次の問題", key="pattern1_next"):
                st.session_state.pattern1_idx += 1
                st.rerun()
            # 変更: processed_images_p1 から取得
            item = st.session_state.processed_images_p1[curr_idx]
            st.image(item['processed_image'], use_container_width=True)
            # read_text(item['question'])
            # read_text(item['answer'])

        else:
            st.info("すべての問題を表示し終えました。")
            if st.button("最初からやり直す", key="pattern1_reset"):
                st.session_state.pattern1_idx = 0
                st.session_state.pattern1_started = False
                st.rerun()

with tab4:
    if 'pattern2_started' not in st.session_state:
        st.session_state.pattern2_started = False
    if 'pattern2_idx' not in st.session_state:
        st.session_state.pattern2_idx = 0

    # 変更: processed_images_p2 をチェック
    if not st.session_state.processed_images_p2:
        st.info("「画像処理」タブでパターン2の画像を処理してください。")
    elif not st.session_state.pattern2_started:
        if st.button("学習を開始", key="pattern2_start"):
            # 変更: idx_start は 0 から
            idx_start = 0
            
            # 変更: processed_images_p2 でチェック
            if idx_start >= len(st.session_state.processed_images_p2) and len(st.session_state.processed_images_p2) > 0:
                st.warning(f"開始インデックス({idx_start})が処理済み画像数({len(st.session_state.processed_images_p2)})を超えています。0から開始します。")
                idx_start = 0
            elif len(st.session_state.processed_images_p2) == 0:
                 st.warning("処理済みの画像がありません。")
                 idx_start = 0
            
            st.session_state.pattern2_idx = idx_start
            st.session_state.pattern2_started = True
            st.rerun()
    else:
        curr_idx = st.session_state.pattern2_idx

        # 変更: ループ条件を processed_images_p2 の長さに
        if curr_idx < len(st.session_state.processed_images_p2):
            if st.button("次の問題", key="pattern2_next"):
                st.session_state.pattern2_idx += 1
                st.rerun() 
            # 変更: processed_images_p2 から取得
            item = st.session_state.processed_images_p2[curr_idx]
            st.image(item['processed_image'], use_container_width=True)
            # read_text(item['question'])
            # read_text(item['answer'])
        else:
            st.info("すべての問題を表示し終えました。")
            if st.button("最初からやり直す", key="pattern2_reset"):
                # 変更: idx は 0 に戻る
                st.session_state.pattern2_idx = 0 
                st.session_state.pattern2_started = False
                st.rerun()

# ▼▼▼ 変更点: タブ5を追加 ▼▼▼
with tab5:
    # クイズ用のセッション状態初期化
    if 'p1_quiz_started' not in st.session_state:
        st.session_state.p1_quiz_started = False
    if 'p1_quiz_idx' not in st.session_state:
        st.session_state.p1_quiz_idx = 0
    if 'p1_quiz_score' not in st.session_state:
        st.session_state.p1_quiz_score = 0
    if 'p1_quiz_answered' not in st.session_state:
        # 現在の問題に回答済みかどうかのフラグ
        st.session_state.p1_quiz_answered = False 
    
    quiz_data = st.session_state.processed_images_p1
    total_quizzes = len(quiz_data)

    if not quiz_data:
        st.info("「画像処理」タブでパターン1の画像を処理してください。")
    elif not st.session_state.p1_quiz_started:
        st.info(f"パターン1で学習した {total_quizzes} 問のクイズを開始します。")
        if st.button("クイズ開始", key="p1_quiz_start"):
            # 状態をリセットして開始
            st.session_state.p1_quiz_started = True
            st.session_state.p1_quiz_idx = 0
            st.session_state.p1_quiz_score = 0
            st.session_state.p1_quiz_answered = False
            # 過去の回答をクリア
            for i in range(total_quizzes):
                if f"p1_quiz_radio_{i}" in st.session_state:
                    del st.session_state[f"p1_quiz_radio_{i}"]
                if f"p1_quiz_options_{i}" in st.session_state:
                    del st.session_state[f"p1_quiz_options_{i}"]
            st.rerun()
    else:
        curr_idx = st.session_state.p1_quiz_idx
        
        if curr_idx < total_quizzes:
            item = quiz_data[curr_idx]
            question = item['question']
            correct_answer = item['answer']
            
            # 選択肢をシャッフル (セッションに保存してシャッフルが固定されるようにする)
            options_key = f"p1_quiz_options_{curr_idx}"
            if options_key not in st.session_state:
                options = [correct_answer, item['dammy1'], item['dammy2'], item['dammy3']]
                random.shuffle(options)
                st.session_state[options_key] = options
            else:
                options = st.session_state[options_key]

            st.subheader(f"問題 {curr_idx + 1} / {total_quizzes}")
            st.write(f"**問題:** {question}")
            
            radio_key = f"p1_quiz_radio_{curr_idx}"
            user_answer = st.radio(
                "解答を選択してください:",
                options,
                key=radio_key,
                index=None,
                disabled=st.session_state.p1_quiz_answered # 回答済みなら無効化
            )

            if not st.session_state.p1_quiz_answered:
                # 回答ボタン
                if st.button("回答を確定", key=f"p1_quiz_submit_{curr_idx}"):
                    if user_answer is None:
                        st.warning("解答を選択してください。")
                    else:
                        st.session_state.p1_quiz_answered = True
                        if user_answer == correct_answer:
                            st.session_state.p1_quiz_score += 1
                        st.session_state.p1_quiz_idx += 1
                        st.session_state.p1_quiz_answered = False
                        st.rerun() 
        else:
            # クイズ終了
            st.balloons()
            st.success(f"クイズ終了！ お疲れ様でした。")
            st.metric(
                label="最終スコア",
                value=f"{st.session_state.p1_quiz_score} / {total_quizzes}",
            )
            if st.button("もう一度挑戦する", key="p1_quiz_reset"):
                st.session_state.p1_quiz_started = False
                st.rerun()


with tab6:
    # クイズ用のセッション状態初期化
    if 'p2_quiz_started' not in st.session_state:
        st.session_state.p2_quiz_started = False
    if 'p2_quiz_idx' not in st.session_state:
        st.session_state.p2_quiz_idx = 0
    if 'p2_quiz_score' not in st.session_state:
        st.session_state.p2_quiz_score = 0
    if 'p2_quiz_answered' not in st.session_state:
        # 現在の問題に回答済みかどうかのフラグ
        st.session_state.p2_quiz_answered = False 
    
    quiz_data = st.session_state.processed_images_p2
    total_quizzes = len(quiz_data)

    if not quiz_data:
        st.info("「画像処理」タブでパターン2の画像を処理してください。")
    elif not st.session_state.p2_quiz_started:
        st.info(f"パターン2で学習した {total_quizzes} 問のクイズを開始します。")
        if st.button("クイズ開始", key="p2_quiz_start"):
            # 状態をリセットして開始
            st.session_state.p2_quiz_started = True
            st.session_state.p2_quiz_idx = 0
            st.session_state.p2_quiz_score = 0
            st.session_state.p2_quiz_answered = False
            # 過去の回答をクリア
            for i in range(total_quizzes):
                if f"p2_quiz_radio_{i}" in st.session_state:
                    del st.session_state[f"p2_quiz_radio_{i}"]
                if f"p2_quiz_options_{i}" in st.session_state:
                    del st.session_state[f"p2_quiz_options_{i}"]
            st.rerun()
    else:
        curr_idx = st.session_state.p2_quiz_idx
        
        if curr_idx < total_quizzes:
            item = quiz_data[curr_idx]
            question = item['question']
            correct_answer = item['answer']
            
            # 選択肢をシャッフル (セッションに保存してシャッフルが固定されるようにする)
            options_key = f"p2_quiz_options_{curr_idx}"
            if options_key not in st.session_state:
                options = [correct_answer, item['dammy1'], item['dammy2'], item['dammy3']]
                random.shuffle(options)
                st.session_state[options_key] = options
            else:
                options = st.session_state[options_key]

            st.subheader(f"問題 {curr_idx + 1} / {total_quizzes}")
            st.write(f"**問題:** {question}")
            
            radio_key = f"p2_quiz_radio_{curr_idx}"
            user_answer = st.radio(
                "解答を選択してください:",
                options,
                key=radio_key,
                index=None,
                disabled=st.session_state.p2_quiz_answered # 回答済みなら無効化
            )

            if not st.session_state.p2_quiz_answered:
                # 回答ボタン
                if st.button("回答を確定", key=f"p2_quiz_submit_{curr_idx}"):
                    if user_answer is None:
                        st.warning("解答を選択してください。")
                    else:
                        st.session_state.p2_quiz_answered = True
                        if user_answer == correct_answer:
                            st.session_state.p2_quiz_score += 1
                        st.session_state.p2_quiz_idx += 1
                        st.session_state.p2_quiz_answered = False
                        st.rerun()        
        else:
            # クイズ終了
            st.balloons()
            st.success(f"クイズ終了！ お疲れ様でした。")
            st.metric(
                label="最終スコア",
                value=f"{st.session_state.p2_quiz_score} / {total_quizzes}",
            )
            if st.button("もう一度挑戦する", key="p2_quiz_reset"):
                st.session_state.p2_quiz_started = False
                st.rerun()
