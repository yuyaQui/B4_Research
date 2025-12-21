# ============================================================================
# Imports
# ============================================================================
import os
import pickle
import random
import time
# threading は削除
# cv2, mediapipe は削除（アイトラッキング用だったため）

import torch
import streamlit as st
import numpy as np
from PIL import Image
import pyttsx3

from TranSalNet_Dense import TranSalNet
from furigana_preprocess import DATASETS_PATH
from experiment_image_draw import (
    find_optimal_text_position,
    find_lower_text_position_and_draw,
    draw_answer_text_on_image
)

# ============================================================================
# Constants
# ============================================================================
MODEL_PATH_DENSE = r'pretrained_models\TranSalNet_Dense.pth'
SOURCE_PATH = "test_quiz"
NUM_TO_OPTIMIZE = 25  # 各パターンで処理する最大数
READING_SPEED = 120
# アイトラッキング用の閾値定数は削除

# ============================================================================
# Session State Initialization
# ============================================================================
def initialize_session_state():
    """セッション状態を初期化"""
    # データセット
    if 'experiment_set' not in st.session_state:
        try:
            with open(os.path.join(DATASETS_PATH, f"{SOURCE_PATH}_quizes_and_images.pkl"), "rb") as f:
                st.session_state.experiment_set = pickle.load(f)
                total_loaded = len(st.session_state.experiment_set)
                print(f"\n--- [初期読み込み] {total_loaded} 問のクイズを読み込みました ---")
        except FileNotFoundError:
            st.error(f"データファイルが見つかりません: {os.path.join(DATASETS_PATH, f'{SOURCE_PATH}_quizes_and_images.pkl')}")
            st.session_state.experiment_set = []
        except Exception as e:
            st.error(f"データファイルの読み込み中にエラーが発生しました: {e}")
            st.session_state.experiment_set = []
    
    # 未知語リスト
    if 'unknown_quizes_part1' not in st.session_state:
        st.session_state.unknown_quizes_part1 = []
        st.session_state.unknown_quizes_part2 = []
        st.session_state.current_quiz_index = 0
        st.session_state.quiz_selection_done = False
    
    # モデル
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None
    
    # 処理済み画像リスト
    if 'processed_images_p1' not in st.session_state:
        st.session_state.processed_images_p1 = []
    if 'processed_images_p2' not in st.session_state:
        st.session_state.processed_images_p2 = []

initialize_session_state()

# ============================================================================
# Utility Functions
# ============================================================================
def read_text(text: str):
    """テキストを音声ファイル(WAV)に保存してStreamlitで再生する（プレイヤーはCSSで隠す）"""
    try:
        # 一時ファイル名
        temp_file = "temp_speech.wav"
        
        # エンジン初期化 & 保存
        # 注意: 毎回initすると重くなる場合があるので、モジュールグローバルでinit済みのengineを使う設計もアリですが、
        # Streamlitのrerun特性上、ここでのinit/stopが最も安全です。
        engine = pyttsx3.init()
        engine.setProperty('rate', READING_SPEED)
        engine.save_to_file(text, temp_file)
        engine.runAndWait()
        engine.stop()
        
        # 再生 (CSSで .stAudio { display: none; } となっていれば表示されない)
        if os.path.exists(temp_file):
            with open(temp_file, "rb") as f:
                audio_bytes = f.read()
                
            # autoplay=True で自動再生
            st.audio(audio_bytes, format="audio/wav", autoplay=True)
            
    except Exception as e:
        # エラー時はコンソールに出すか、開発中はst.warningで表示するなど
        print(f"音声読み上げエラー: {e}")


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

# アイトラッキング関連の関数（start/stop/run_gaze_tracker）は削除しました

# ============================================================================
# UI Functions
# ============================================================================
def ask_unknown_words_ui(quizes_and_images, max_count=20):
    """
    未知語選択UI
    Returns: (unknown_part1, unknown_part2, completed)
    """
    st.header("📝 クイズの解答候補")
    st.write("知っている単語には 'はい'、知らない単語には 'いいえ' を選択してください。")
    
    # ラジオボタンを表示
    for i, (question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes_and_images):
        if i >= max_count:
            break
        
        with st.container():
            st.write(f"**{i+1}. '{target}'**")
            st.radio(
                "知っていますか？",
                ["はい", "いいえ"],
                key=f"quiz_{i}",
                horizontal=True,
                index=None
            )
    
    # 回答状況を集計
    responses = []
    for i in range(max_count):
        if f"quiz_{i}" in st.session_state and st.session_state[f"quiz_{i}"] is not None:
            responses.append(st.session_state[f"quiz_{i}"])
    
    all_answered = len(responses) == max_count
    
    if not all_answered:
        remaining = max_count - len(responses)
        st.info(f"すべての解答を選択してください。（残り {remaining} 問）")
    else:
        st.success("すべての解答が選択されました。「選択を完了」ボタンを押してください。")
    
    # 完了ボタン
    if st.button("選択を完了", key="complete_selection"):
        if all_answered:
            unknown_part1 = []
            unknown_part2 = []
            mid_point = max_count // 2
            
            for i, (question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes_and_images[:max_count]):
                if st.session_state[f"quiz_{i}"] == "いいえ":
                    quiz_data = (question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3, i)
                    if i < mid_point:
                        unknown_part1.append(quiz_data)
                    else:
                        unknown_part2.append(quiz_data)
            
            return unknown_part1, unknown_part2, True
        else:
            st.error("まだすべての設問に回答していません。")
            return [], [], False
    
    return [], [], False

# ============================================================================
# Image Processing Functions
# ============================================================================
def process_image_pattern1(quiz_data, index):
    """パターン1（Saliency）の画像処理"""
    question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index = quiz_data
    
    try:
        # 画像の読み込み
        if isinstance(image, Image.Image):
            generated_image_pil = image
        elif isinstance(image, str):
            if not os.path.exists(image):
                st.error(f"P1: 画像パスが見つかりません: {image} [Index: {original_index}]")
                return None
            generated_image_pil = Image.open(image)
        else:
            st.error(f"P1: 予期しない画像データ型: {type(image)} [Index: {original_index}]")
            return None
        
        image_copy = generated_image_pil.copy()
        
        # 最適位置を見つけてテキストを描画
        x, y = find_optimal_text_position(
            image_copy,
            st.session_state.model,
            st.session_state.device
        )
        image_with_caption = draw_answer_text_on_image(image_copy, target, x, y)
        
        return {
            'question_1': question_1,
            'question_1_read': question_1_read,
            'target': target,
            'question_2': question_2,
            'answer': answer,
            'dammy1': dammy1,
            'dammy2': dammy2,
            'dammy3': dammy3,
            'original_image': generated_image_pil,
            'processed_image': image_with_caption,
            'position': (x, y),
            'original_index': original_index
        }
    
    except Exception as e:
        st.error(f"パターン1の画像 {index+1} ('{answer}') [Index: {original_index}] の処理中にエラー: {e}")
        return None


def process_image_pattern2(quiz_data, index):
    """パターン2（下部固定）の画像処理"""
    question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index = quiz_data
    
    try:
        # 画像の読み込み
        if isinstance(image, Image.Image):
            generated_image_pil = image
        elif isinstance(image, str):
            if not os.path.exists(image):
                st.error(f"P2: 画像パスが見つかりません: {image} [Index: {original_index}]")
                return None
            generated_image_pil = Image.open(image)
        else:
            st.error(f"P2: 予期しない画像データ型: {type(image)} [Index: {original_index}]")
            return None
        
        image_copy = generated_image_pil.copy()
        
        # 下部にテキストを描画
        image_with_caption = find_lower_text_position_and_draw(image_copy, target)
        img_width, img_height = image_with_caption.size
        x, y = img_width // 2, img_height // 2
        
        return {
            'question_1': question_1,
            'question_1_read': question_1_read,
            'target': target,
            'question_2': question_2,
            'answer': answer,
            'dammy1': dammy1,
            'dammy2': dammy2,
            'dammy3': dammy3,
            'original_image': generated_image_pil,
            'processed_image': image_with_caption,
            'position': (x, y),
            'original_index': original_index
        }
    
    except Exception as e:
        st.error(f"パターン2の画像 {index+1} ('{answer}') [Index: {original_index}] の処理中にエラー: {e}")
        return None

# ============================================================================
# Tab Functions
# ============================================================================
def render_tab1_quiz_selection():
    """タブ1: クイズ選択"""
    st.header("実験設定")
    
    # データセットのステータス表示
    if not st.session_state.experiment_set:
        st.warning("データセットが読み込まれていません。")
        return

    total_loaded = len(st.session_state.experiment_set)
    st.write(f"読み込み済みクイズ数: {total_loaded} 問")

    max_quizzes = st.number_input(
        "実験に使用する最大クイズ数（前半と後半に均等に分割されます）",
        min_value=2,
        max_value=total_loaded,
        value=min(20, total_loaded),
        step=2,  # 偶数単位で増減
        key="max_quizzes"
    )
    
    st.radio(
        "条件割り当て順序",
        ["1: 前半=P1(Saliency), 後半=P2(固定)", "2: 前半=P2(固定), 後半=P1(Saliency)"],
        key="quiz_order_radio",
        horizontal=False,
        index=0,
    )
    
    # クイズ開始（設定確定）ボタン
    if st.button("実験セットアップを実行", key="start_quiz"):
        # セッション状態のリセット
        st.session_state.quiz_started = True 
        st.session_state.unknown_quizes_part1 = []
        st.session_state.unknown_quizes_part2 = []
        st.session_state.processed_images_p1 = []
        st.session_state.processed_images_p2 = []
        st.session_state.p1_quiz_started = False
        st.session_state.p2_quiz_started = False
        st.session_state.p1_quiz_idx = 0
        st.session_state.p2_quiz_idx = 0
        st.session_state.max_quizzes_on_start = int(max_quizzes)
        
        # 過去の回答記録をクリア
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("quiz_") or k.startswith("pattern")]
        for k in keys_to_remove:
            if k in st.session_state:
                del st.session_state[k]

        # データセットの準備と分割
        full_set = st.session_state.experiment_set
        # 指定数だけ取得（先頭から）
        current_set = full_set[:st.session_state.max_quizzes_on_start]
        
        # データにインデックス情報を付与 (item + (original_index,))
        formatted_set = []
        for i, item in enumerate(current_set):
            formatted_set.append(item + (i,))
            
        # 半分に分割
        mid_point = len(formatted_set) // 2
        part1 = formatted_set[:mid_point]
        part2 = formatted_set[mid_point:]
        
        # 条件順序による入れ替え
        # ラジオボタンの選択肢文字列から判定（"1:..." or "2:..."）
        selected_order = st.session_state.get("quiz_order_radio", "1")[0]
        
        if selected_order == "2":
            st.session_state.unknown_quizes_part1 = part2 # P1用変数にpart2を入れる（変則的だが、ロジック上はP1用のリストに何を入れるか）
            st.session_state.unknown_quizes_part2 = part1 # P2用変数にpart1を入れる
            print("\n--- [タブ1] 条件割り当て: 前半セット->P2(固定), 後半セット->P1(Saliency) ---")
        else:
            st.session_state.unknown_quizes_part1 = part1
            st.session_state.unknown_quizes_part2 = part2
            print("\n--- [タブ1] 条件割り当て: 前半セット->P1(Saliency), 後半セット->P2(固定) ---")
            
        # ランダムシャッフル（実験順序効果の低減のため）
        random.shuffle(st.session_state.unknown_quizes_part1)
        random.shuffle(st.session_state.unknown_quizes_part2)
        
        st.session_state.quiz_selection_done = True
        st.rerun()

    # セットアップ完了後の表示
    if st.session_state.get("quiz_selection_done", False):
        p1_count = len(st.session_state.unknown_quizes_part1)
        p2_count = len(st.session_state.unknown_quizes_part2)
        
        st.success("✅ 実験セットアップが完了しました。")
        st.info(f"**パターン1 (Saliency)**: {p1_count} 問\n\n**パターン2 (下部固定)**: {p2_count} 問")
        st.write("「画像処理」タブへ移動して準備を進めてください。")


def render_tab2_image_processing():
    """タブ2: 画像処理"""
    if not st.session_state.quiz_selection_done:
        st.warning("まず「クイズ選択」タブで未知の単語を選択してください。")
    elif not st.session_state.experiment_set:
        st.warning("データセットが読み込まれていません。")
    elif not st.session_state.unknown_quizes_part1 and not st.session_state.unknown_quizes_part2:
        st.warning("処理対象の未知の単語がありません。")
    else:
        st.info(f"パターン1 (Saliency) は最大 {NUM_TO_OPTIMIZE} 問、\n"
               f"パターン2 (下部固定) は最大 {NUM_TO_OPTIMIZE} 問を処理します。")
        
        if st.button("画像処理を開始", key="process_images"):
            load_model()
            
            if st.session_state.model is None:
                st.error("モデルが読み込まれていないため、処理を中断しました。")
            else:
                st.session_state.processed_images_p1 = []
                st.session_state.processed_images_p2 = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                quizes_p1 = st.session_state.unknown_quizes_part1
                total_p1 = min(len(quizes_p1), NUM_TO_OPTIMIZE)
                
                quizes_p2 = st.session_state.unknown_quizes_part2
                total_p2 = min(len(quizes_p2), NUM_TO_OPTIMIZE)
                
                # パターン1の処理
                if total_p1 > 0:
                    status_text.text(f"パターン1 (Saliency) 処理中: 0/{total_p1}")
                    for i in range(total_p1):
                        status_text.text(f"パターン1 (Saliency) 処理中: {i+1}/{total_p1}")
                        progress_bar.progress((i + 1) / total_p1)
                        
                        result = process_image_pattern1(quizes_p1[i], i)
                        if result:
                            st.session_state.processed_images_p1.append(result)
                
                # パターン2の処理
                if total_p2 > 0:
                    status_text.text(f"パターン2 処理中: 0/{total_p2}")
                    progress_bar.progress(0)
                    
                    for i in range(total_p2):
                        status_text.text(f"パターン2 処理中: {i+1}/{total_p2}")
                        progress_bar.progress((i + 1) / total_p2)
                        
                        result = process_image_pattern2(quizes_p2[i], i)
                        if result:
                            st.session_state.processed_images_p2.append(result)
                
                progress_bar.empty()
                status_text.text("処理完了！")
                st.success(f"パターン1: {len(st.session_state.processed_images_p1)}個, "
                          f"パターン2: {len(st.session_state.processed_images_p2)}個 の画像を処理しました。")


def initialize_learning_session_state(pattern_num):
    """学習セッション状態を初期化"""
    prefix = f"pattern{pattern_num}"
    
    if f'{prefix}_started' not in st.session_state:
        st.session_state[f'{prefix}_started'] = False
    if f'{prefix}_idx' not in st.session_state:
        st.session_state[f'{prefix}_idx'] = 0
    if f'start_time_{pattern_num}' not in st.session_state:
        st.session_state[f'start_time_{pattern_num}'] = 0
    if f'end_time_{pattern_num}' not in st.session_state:
        st.session_state[f'end_time_{pattern_num}'] = 0
    if f'p{pattern_num}_study_time_logged' not in st.session_state:
        st.session_state[f'p{pattern_num}_study_time_logged'] = False


def render_learning_tab(pattern_num, pattern_name, processed_images_key):
    st.markdown(
        """
        <style>
        /* オーディオプレイヤーを非表示にする */
        .stAudio {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    initialize_learning_session_state(pattern_num)
    
    prefix = f"pattern{pattern_num}"
    processed_images = st.session_state[processed_images_key]
    
    if not processed_images:
        st.info(f"「画像処理」タブで{pattern_name}の画像を処理してください。")
    elif not st.session_state[f'{prefix}_started']:
        if st.button("学習を開始", key=f"{prefix}_start"):
            st.session_state[f'{prefix}_idx'] = 0
            st.session_state[f'{prefix}_started'] = True
            st.session_state[f'start_time_{pattern_num}'] = time.time()
            st.session_state[f'p{pattern_num}_study_time_logged'] = False
            st.session_state[f'end_time_{pattern_num}'] = 0
            st.rerun()
    else:
        # 問題表示
        curr_idx = st.session_state[f'{prefix}_idx']
        
        if curr_idx < len(processed_images):
            # 次の問題に進むボタン
            if st.button("次の問題へ", key=f"{prefix}_next"):
                st.session_state[f'{prefix}_idx'] += 1
                st.rerun()

            # JavaScriptを埋め込んでEnterキーでボタンをクリックさせる
            st.components.v1.html(
                f"""
                <script>
                const parentDoc = window.parent.document;
                if (!parentDoc.hasOwnProperty('_enter_listener_attached_{prefix}')) {{
                    parentDoc.addEventListener('keydown', function(e) {{
                        if (e.keyCode === 13) {{
                            const buttons = parentDoc.getElementsByTagName('button');
                            for (let i = 0; i < buttons.length; i++) {{
                                if (buttons[i].innerText.includes("次の問題へ")) {{
                                    buttons[i].click();
                                    e.preventDefault();
                                    e.stopPropagation();
                                    break;
                                }}
                            }}
                        }}
                    }});
                    parentDoc['_enter_listener_attached_{prefix}'] = true;
                }}
                </script>
                """,
                height=0,
                width=0,
            )
            
            item = processed_images[curr_idx]
            st.image(item['processed_image'], use_container_width=True)
            read_text(item['question_1_read'])
        else:
            st.info("すべての問題を表示し終えました。")
            
            if not st.session_state[f'p{pattern_num}_study_time_logged']:
                st.session_state[f'end_time_{pattern_num}'] = time.time()
                study_time = st.session_state[f'end_time_{pattern_num}'] - st.session_state[f'start_time_{pattern_num}']
                
                print(f"\n--- [タブ{pattern_num+2}] {pattern_name} 学習時間: {study_time:.2f} s ---")
                
                st.session_state[f'p{pattern_num}_study_time_logged'] = True
            
            if st.button("最初からやり直す", key=f"{prefix}_reset"):
                st.session_state[f'{prefix}_idx'] = 0
                st.session_state[f'{prefix}_started'] = False
                st.session_state[f'start_time_{pattern_num}'] = 0
                st.session_state[f'end_time_{pattern_num}'] = 0
                st.session_state[f'p{pattern_num}_study_time_logged'] = False
                st.rerun()


def initialize_quiz_session_state(pattern_num):
    """クイズセッション状態を初期化"""
    prefix = f"p{pattern_num}"
    
    if f'{prefix}_quiz_started' not in st.session_state:
        st.session_state[f'{prefix}_quiz_started'] = False
    if f'{prefix}_quiz_idx' not in st.session_state:
        st.session_state[f'{prefix}_quiz_idx'] = 0
    if f'{prefix}_quiz_score' not in st.session_state:
        st.session_state[f'{prefix}_quiz_score'] = 0
    if f'{prefix}_quiz_answered' not in st.session_state:
        st.session_state[f'{prefix}_quiz_answered'] = False
    if f'{prefix}_quiz_order' not in st.session_state:
        st.session_state[f'{prefix}_quiz_order'] = []


def render_quiz_tab(pattern_num, pattern_name, processed_images_key):
    """クイズタブのレンダリング（共通処理）"""
    initialize_quiz_session_state(pattern_num)
    
    prefix = f"p{pattern_num}"
    quiz_data = st.session_state[processed_images_key]
    total_quizzes = len(quiz_data)
    
    if not quiz_data:
        st.info(f"「画像処理」タブで{pattern_name}の画像を処理してください。")
    elif not st.session_state[f'{prefix}_quiz_started']:
        st.info(f"{pattern_name}で学習した {total_quizzes} 問のクイズを開始します。")
        
        if st.button("クイズ開始", key=f"{prefix}_quiz_start"):
            st.session_state[f'{prefix}_quiz_started'] = True
            st.session_state[f'{prefix}_quiz_idx'] = 0
            st.session_state[f'{prefix}_quiz_score'] = 0
            st.session_state[f'{prefix}_quiz_answered'] = False
            
            # インデックスリストを作成しシャッフル
            st.session_state[f'{prefix}_quiz_order'] = list(range(total_quizzes))
            random.shuffle(st.session_state[f'{prefix}_quiz_order'])
            print(f"\n--- [タブ{pattern_num+4}] クイズ順序 (ランダム): {st.session_state[f'{prefix}_quiz_order']} ---")
            
            # 過去の回答をクリア
            for i in range(total_quizzes):
                if f"{prefix}_quiz_radio_{i}" in st.session_state:
                    del st.session_state[f"{prefix}_quiz_radio_{i}"]
                if f"{prefix}_quiz_options_{i}" in st.session_state:
                    del st.session_state[f"{prefix}_quiz_options_{i}"]
            st.rerun()
    else:
        curr_idx = st.session_state[f'{prefix}_quiz_idx']
        
        if curr_idx < total_quizzes:
            # シャッフルされた順序から実際のデータインデックスを取得
            actual_idx = st.session_state[f'{prefix}_quiz_order'][curr_idx]
            item = quiz_data[actual_idx]
            
            question = item['question_2']
            correct_answer = item['answer']
            
            # 選択肢
            options_key = f"{prefix}_quiz_options_{curr_idx}"
            if options_key not in st.session_state:
                options = [correct_answer, item['dammy1'], item['dammy2'], item['dammy3']]
                random.shuffle(options)
                st.session_state[options_key] = options
            else:
                options = st.session_state[options_key]
            
            st.subheader(f"問題 {curr_idx + 1} / {total_quizzes}")
            st.write(f"**問題:** {question}")
            
            radio_key = f"{prefix}_quiz_radio_{curr_idx}"
            user_answer = st.radio(
                "解答を選択してください:",
                options,
                key=radio_key,
                index=None,
                disabled=st.session_state[f'{prefix}_quiz_answered']
            )
            
            if not st.session_state[f'{prefix}_quiz_answered']:
                if st.button("回答を確定", key=f"{prefix}_quiz_submit_{curr_idx}"):
                    if user_answer is None:
                        st.warning("解答を選択してください。")
                    else:
                        st.session_state[f'{prefix}_quiz_answered'] = True
                        if user_answer == correct_answer:
                            st.session_state[f'{prefix}_quiz_score'] += 1
                        st.session_state[f'{prefix}_quiz_idx'] += 1
                        st.session_state[f'{prefix}_quiz_answered'] = False
                        st.rerun()
        else:
            # クイズ終了
            st.metric(
                label="最終スコア",
                value=f"{st.session_state[f'{prefix}_quiz_score']} / {total_quizzes}",
            )
            
            if st.button("もう一度挑戦する", key=f"{prefix}_quiz_reset"):
                st.session_state[f'{prefix}_quiz_started'] = False
                st.session_state[f'{prefix}_quiz_order'] = []
                st.rerun()

# ============================================================================
# Main UI
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "クイズ選択",
    "画像処理",
    "条件A（学習）",
    "条件B（学習）",
    "条件A（テスト）",
    "条件B（テスト）"
])

with tab1:
    render_tab1_quiz_selection()

with tab2:
    render_tab2_image_processing()

with tab3:
    render_learning_tab(1, "パターン1", "processed_images_p1")

with tab4:
    render_learning_tab(2, "パターン2", "processed_images_p2")

with tab5:
    render_quiz_tab(1, "パターン1", "processed_images_p1")

with tab6:
    render_quiz_tab(2, "パターン2", "processed_images_p2")