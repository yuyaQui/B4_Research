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
NUM_TO_OPTIMIZE = 2
READING_SPEED = 150

# --- セッション状態の初期化 ---
if 'experiment_set' not in st.session_state:
    try:
        with open(os.path.join(DATASETS_PATH, "quizes_and_images.pkl"), "rb") as f:
            st.session_state.experiment_set = pickle.load(f)
    except FileNotFoundError:
        st.error(f"データファイルが見つかりません: {os.path.join(DATASETS_PATH, 'quizes_and_images.pkl')}")
        st.session_state.experiment_set = [] # エラー時に空リストをセット
    except Exception as e:
        st.error(f"データファイルの読み込み中にエラーが発生しました: {e}")
        st.session_state.experiment_set = []

if 'unknown_quizes_and_images' not in st.session_state:
    st.session_state.unknown_quizes_and_images = []
    st.session_state.current_quiz_index = 0
    st.session_state.quiz_selection_done = False

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.device = None

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

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
    """未知語選択UI（完了ボタン付き）。(unknown_quizes, completed) を返す。"""
    st.header("📝 クイズの解答候補")
    st.write("知っている単語には 'はい'、知らない単語には 'いいえ' を選択してください。")
    
    # ラジオボタンを表示
    for i, (question, answer, image) in enumerate(quizes_and_images):
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
    
    all_answered = len(responses) == max_count

    if not all_answered:
        remaining = max_count - len(responses)
        st.info(f"すべての解答を選択してください。（残り {remaining} 問）")
    else:
        st.success("すべての解答が選択されました。「選択を完了」ボタンを押してください。")

    # 完了ボタン
    if st.button("選択を完了", key="complete_selection"):
        if all_answered:
            unknown_quizes = []
            for i, (question, answer, image) in enumerate(quizes_and_images[:max_count]):
                if st.session_state[f"quiz_{i}"] == "いいえ":
                    unknown_quizes.append((question, answer, image))
            return unknown_quizes, True 
        else:
            st.error("まだすべての設問に回答していません。")
            return [], False
    
    return [], False

# --- メインUI ---
tab1, tab2, tab3, tab4 = st.tabs(["クイズ選択", "画像処理", "パターン1", "パターン2"])

with tab1:    
    max_quizzes = st.number_input(
        "最大クイズ数", 
        min_value=1, 
        max_value=50, 
        value=20, 
        key="max_quizzes"
    )

    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'max_quizzes_on_start' not in st.session_state:
        st.session_state.max_quizzes_on_start = 20

    if st.button("クイズを開始", key="start_quiz"):
        st.session_state.quiz_started = True
        st.session_state.unknown_quizes_and_images = []
        st.session_state.quiz_selection_done = False
        st.session_state.processed_images = []
        
        max_to_reset = max(50, st.session_state.max_quizzes_on_start) 
        for i in range(max_to_reset): 
            if f"quiz_{i}" in st.session_state:
                del st.session_state[f"quiz_{i}"]
                
        st.session_state.max_quizzes_on_start = int(max_quizzes)
        st.rerun() 

    if st.session_state.quiz_started and not st.session_state.quiz_selection_done:
        unknown_quizes, completed = ask_unknown_words_ui(
            st.session_state.experiment_set, 
            max_count=st.session_state.max_quizzes_on_start
        )
        
        if completed:
            st.session_state.unknown_quizes_and_images = unknown_quizes
            random.shuffle(st.session_state.unknown_quizes_and_images)
            st.session_state.quiz_selection_done = True
            st.session_state.quiz_started = False
            st.success(f"{len(st.session_state.unknown_quizes_and_images)}個の未知の単語が見つかりました！")
            st.rerun() 
            
    if st.session_state.quiz_selection_done:
        st.info(f"✅ {len(st.session_state.unknown_quizes_and_images)}個の未知の単語が選択されました。")

with tab2:    
    if not st.session_state.quiz_selection_done:
        st.warning("まず「クイズ選択」タブで未知の単語を選択してください。")
    elif not st.session_state.experiment_set:
         st.warning("データセットが読み込まれていません。")
    else:
        num_to_process = st.number_input(
            "処理する画像数", 
            min_value=1, 
            max_value=len(st.session_state.unknown_quizes_and_images),
            value=min(5, len(st.session_state.unknown_quizes_and_images)) if st.session_state.unknown_quizes_and_images else 1,
            key="num_to_process"
        )
        
        if st.button("画像処理を開始", key="process_images"):
            load_model()
            # モデル読み込み失敗時は処理中断
            if st.session_state.model is None:
                st.error("モデルが読み込まれていないため、処理を中断しました。")
            else:
                st.session_state.processed_images = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_to_process = int(num_to_process)
                for i in range(total_to_process):
                    status_text.text(f"処理中: {i+1}/{total_to_process}")
                    progress_bar.progress((i + 1) / total_to_process)
                    
                    answer = st.session_state.unknown_quizes_and_images[i][1]
                    generated_image = st.session_state.unknown_quizes_and_images[i][2]
                    
                    image_copy = generated_image.copy()
                    
                    try:
                        if i < NUM_TO_OPTIMIZE:
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
                        else:
                            image_with_caption = find_lower_text_position_and_draw(
                                image_copy, answer
                            )
                            img_width, img_height = image_with_caption.size
                            x, y = img_width // 2, img_height // 2 # 参考座標
                        
                        question = st.session_state.unknown_quizes_and_images[i][0]
                        st.session_state.processed_images.append({
                            'question': question,
                            'answer': answer,
                            'original_image': generated_image,
                            'processed_image': image_with_caption,
                            'position': (x, y)
                        })
                    except Exception as e:
                        st.error(f"画像 {i+1} ('{answer}') の処理中にエラーが発生しました: {e}")
                
                progress_bar.empty()
                status_text.text("処理完了！")
                st.success(f"{len(st.session_state.processed_images)}個の画像を処理しました。")

with tab3:
    if 'pattern1_started' not in st.session_state:
        st.session_state.pattern1_started = False
    if 'pattern1_idx' not in st.session_state:
        st.session_state.pattern1_idx = 0

    if not st.session_state.processed_images:
        st.info("「画像処理」タブで画像を処理してください。")
    elif not st.session_state.pattern1_started:
        if st.button("学習を開始", key="pattern1_start"):
            idx_start = 0
            st.session_state.pattern1_idx = idx_start
            st.session_state.pattern1_started = True
            st.rerun()
    else:
        curr_idx = st.session_state.pattern1_idx
        
        # curr_idx が NUM_TO_OPTIMIZE 未満である間
        if curr_idx < min(NUM_TO_OPTIMIZE, len(st.session_state.processed_images)):
            
            if st.button("次の問題", key="pattern1_next"):
                st.session_state.pattern1_idx += 1
                st.rerun()
            item = st.session_state.processed_images[curr_idx]
            st.image(item['processed_image'], use_container_width=True)
            read_text(item['question'])
            read_text(item['answer'])

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

    if not st.session_state.processed_images:
        st.info("「画像処理」タブで画像を処理してください。")
    elif not st.session_state.pattern2_started:
        if st.button("学習を開始", key="pattern2_start"):
            idx_start = NUM_TO_OPTIMIZE # パターン2はNUM_TO_OPTIMIZEから開始
            
            if idx_start >= len(st.session_state.processed_images) and len(st.session_state.processed_images) > 0:
               st.warning(f"開始インデックス({idx_start})が処理済み画像数({len(st.session_state.processed_images)})を超えています。0から開始します。")
               idx_start = 0
            elif len(st.session_state.processed_images) == 0:
                 st.warning("処理済みの画像がありません。")
                 idx_start = 0
            
            st.session_state.pattern2_idx = idx_start
            st.session_state.pattern2_started = True
            st.rerun()
    else:
        curr_idx = st.session_state.pattern2_idx

        if curr_idx < len(st.session_state.processed_images):
            if st.button("次の問題", key="pattern2_next"):
                st.session_state.pattern2_idx += 1
                st.rerun() 
            item = st.session_state.processed_images[curr_idx]
            st.image(item['processed_image'], use_container_width=True)
            read_text(item['question'])
            read_text(item['answer'])
        else:
            st.info("すべての問題を表示し終えました。")
            if st.button("最初からやり直す", key="pattern2_reset"):
                st.session_state.pattern2_idx = NUM_TO_OPTIMIZE # 開始インデックスに戻る
                st.session_state.pattern2_started = False
                st.rerun()