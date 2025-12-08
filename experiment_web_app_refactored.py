# ============================================================================
# Imports
# ============================================================================
import os
import pickle
import random
import time
import threading

import torch
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import pyttsx3

from TranSalNet_Dense import TranSalNet
from experiment_preprocess import DATASETS_PATH
from experiment_image_draw import (
    find_optimal_text_position,
    find_lower_text_position_and_draw,
    draw_answer_text_on_image
)

# ============================================================================
# Constants
# ============================================================================
MODEL_PATH_DENSE = r'pretrained_models\TranSalNet_Dense.pth'
SOURCE_PATH = "sample_fixed"
NUM_TO_OPTIMIZE = 25  # 各パターンで処理する最大数
READING_SPEED = 120
MOVEMENT_THRESHOLD = 0.8  # アイトラッキングの閾値
FACE_MOVE_THRESHOLD = 1.0  # 顔の動きを検知する閾値

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
            st.error(f"データファイルの読み込み　中にエラーが発生しました: {e}")
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
    """テキストを読み上げる"""
    try:
        time.sleep(0.3)  # 読み上げ前に少し待機
        engine = pyttsx3.init()
        engine.setProperty('rate', READING_SPEED)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
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


def start_gaze_tracker():
    """アイトラッキングを開始"""
    stop_event = threading.Event()
    result_container = {"distance": 0.0, "camera_ready": False}
    
    thread = threading.Thread(target=run_gaze_tracker, args=(stop_event, result_container))
    thread.start()
    
    st.session_state.tracker_thread = thread
    st.session_state.stop_event = stop_event
    st.session_state.result_container = result_container


def stop_gaze_tracker():
    """アイトラッキングを停止して結果を取得"""
    final_distance = 0.0
    
    if st.session_state.tracker_thread is not None:
        st.session_state.stop_event.set()
        st.session_state.tracker_thread.join()
        final_distance = st.session_state.result_container["distance"]
        st.session_state.tracker_thread = None
    
    return final_distance

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
    for i, (question_1, target, image, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes_and_images):
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
            
            for i, (question_1, target, image, question_2, answer, dammy1, dammy2, dammy3) in enumerate(quizes_and_images[:max_count]):
                if st.session_state[f"quiz_{i}"] == "いいえ":
                    quiz_data = (question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, i)
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
# Gaze Tracking
# ============================================================================
def run_gaze_tracker(stop_event, result_container):
    """
    アイトラッキング実行用関数（スレッドで動かす用）
    Args:
        stop_event: スレッドを停止させるためのフラグ
        result_container: 計測結果（距離）を格納する辞書
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # カメラ起動
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera not found")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_distance = 0.0
    prev_left_iris = None
    prev_right_iris = None
    prev_head_pos = None
    camera_initialized = False
    
    # MediaPipe起動
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while not stop_event.is_set() and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # カメラが正常に起動したことを通知
            if not camera_initialized:
                result_container["camera_ready"] = True
                camera_initialized = True
            
            # 画像処理
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # --- 顔の動き検出 ---
                    # 鼻の頭 (Landmark 1) を取得
                    nose_pt = face_landmarks.landmark[1]
                    curr_head_pos = np.array([nose_pt.x * width, nose_pt.y * height])

                    is_head_moving = False
                    if prev_head_pos is not None:
                        head_dist = np.linalg.norm(curr_head_pos - prev_head_pos)
                        if head_dist > FACE_MOVE_THRESHOLD:
                            is_head_moving = True
                    
                    prev_head_pos = curr_head_pos

                    if is_head_moving:
                        # 顔が動いている間はリセット
                        prev_left_iris = None
                        prev_right_iris = None
                        cv2.putText(image, "Head Moving", (30, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue

                    l_pt = face_landmarks.landmark[468]
                    r_pt = face_landmarks.landmark[473]
                    
                    curr_left_iris = np.array([l_pt.x * width, l_pt.y * height])
                    curr_right_iris = np.array([r_pt.x * width, r_pt.y * height])
                    
                    # 描画
                    cv2.circle(image, (int(curr_left_iris[0]), int(curr_left_iris[1])), 3, (0, 255, 0), -1)
                    cv2.circle(image, (int(curr_right_iris[0]), int(curr_right_iris[1])), 3, (0, 255, 0), -1)
                    
                    if prev_left_iris is not None and prev_right_iris is not None:
                        dist_l = np.linalg.norm(curr_left_iris - prev_left_iris)
                        dist_r = np.linalg.norm(curr_right_iris - prev_right_iris)
                        avg_dist = (dist_l + dist_r) / 2.0
                        
                        if avg_dist > MOVEMENT_THRESHOLD:
                            total_distance += avg_dist
                    
                    prev_left_iris = curr_left_iris
                    prev_right_iris = curr_right_iris
            
            # 結果をコンテナに書き込む
            result_container["distance"] = total_distance
            
            # 確認用ウィンドウを表示
            cv2.putText(image, f"Dist: {int(total_distance)}", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Gaze Tracker (Running)', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# Image Processing Functions
# ============================================================================
def process_image_pattern1(quiz_data, index):
    """パターン1（Saliency）の画像処理"""
    question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index = quiz_data
    
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
    question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index = quiz_data
    
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
    max_quizzes = st.number_input(
        "最大クイズ数（前半と後半に均等に分割されます）",
        min_value=2,
        max_value=1000,
        value=80,
        step=1,
        key="max_quizzes"
    )
    
    st.radio(
        "問題順序（パターン割り当て）",
        ["1", "2"],
        key="quiz_order_radio",
        horizontal=True,
        index=0,
    )
    
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'max_quizzes_on_start' not in st.session_state:
        st.session_state.max_quizzes_on_start = 20
    
    if st.button("クイズを開始", key="start_quiz"):
        st.session_state.quiz_started = True
        st.session_state.unknown_quizes_part1 = []
        st.session_state.unknown_quizes_part2 = []
        st.session_state.quiz_selection_done = False
        st.session_state.processed_images_p1 = []
        st.session_state.processed_images_p2 = []
        st.session_state.p1_quiz_started = False
        st.session_state.p2_quiz_started = False
        st.session_state.p1_quiz_idx = 0
        st.session_state.p2_quiz_idx = 0
        
        # クイズ状態をリセット
        max_to_reset = max(50, st.session_state.max_quizzes_on_start)
        for i in range(max_to_reset):
            if f"quiz_{i}" in st.session_state:
                del st.session_state[f"quiz_{i}"]
        
        st.session_state.max_quizzes_on_start = int(max_quizzes)
        
        # ターミナル出力
        try:
            total_quizzes_in_set = len(st.session_state.experiment_set)
            num_presented = st.session_state.max_quizzes_on_start
            
            if total_quizzes_in_set > num_presented:
                unpresented_indices = list(range(num_presented, total_quizzes_in_set))
                print(f"\n--- [タブ1]で出題されなかった問題: {len(unpresented_indices)} 問 ---")
            else:
                print("\n--- [タブ1] すべての問題が出題対象となりました ---")
        except Exception as e:
            print(f"ターミナル出力中にエラー: {e}")
        
        st.rerun()
    
    if st.session_state.quiz_started and not st.session_state.quiz_selection_done:
        # ask_unknown_words_ui の戻り値:
        # - unknown_p1: 前半グループの未知語リスト [(question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index), ...]
        # - unknown_p2: 後半グループの未知語リスト [(question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index), ...]
        # - completed: 選択が完了したかどうか (True/False)
        unknown_p1, unknown_p2, completed = ask_unknown_words_ui(
            st.session_state.experiment_set,
            max_count=st.session_state.max_quizzes_on_start
        )
        
        if completed:
            # 問題順序「2」が選択されていたら入れ替え
            if st.session_state.get("quiz_order_radio") == "2":
                print("\n--- [タブ1] 問題順序「2」が選択されたため、part1とpart2を入れ替えます ---")
                unknown_p1, unknown_p2 = unknown_p2, unknown_p1
            else:
                print("\n--- [タブ1] 問題順序「1」が選択されました (通常) ---")
            
            st.session_state.unknown_quizes_part1 = unknown_p1
            st.session_state.unknown_quizes_part2 = unknown_p2
            
            random.shuffle(st.session_state.unknown_quizes_part1)
            random.shuffle(st.session_state.unknown_quizes_part2)
            
            st.session_state.quiz_selection_done = True
            st.session_state.quiz_started = False
            
            st.success(f"前半 {len(st.session_state.unknown_quizes_part1)}個, "
                      f"後半 {len(st.session_state.unknown_quizes_part2)}個 の未知の単語が見つかりました！")
            
            if st.session_state.get("quiz_order_radio") == "2":
                st.info("問題順序「2」が選択されたため、前半グループが「パターン2 (下部固定)」、後半グループが「パターン1 (Saliency)」に割り当てられます。")
            else:
                st.info("問題順序「1」が選択されたため、前半グループが「パターン1 (Saliency)」、後半グループが「パターン2 (下部固定)」に割り当てられます。")
            
            st.rerun()
    
    if st.session_state.quiz_selection_done:
        st.info(f"✅ 前半 {len(st.session_state.unknown_quizes_part1)}個, "
               f"後半 {len(st.session_state.unknown_quizes_part2)}個 の未知の単語が選択されました。")
        
        if st.session_state.get("quiz_order_radio") == "2":
            st.warning(f"問題順序「2」（入れ替え）が選択されています。\n"
                      f"* 前半グループ ({len(st.session_state.unknown_quizes_part1)}個) は **パターン2 (下部固定)** で学習・テストします。\n"
                      f"* 後半グループ ({len(st.session_state.unknown_quizes_part2)}個) は **パターン1 (Saliency)** で学習・テストします。")
        else:
            st.success(f"問題順序「1」（通常）が選択されています。\n"
                      f"* 前半グループ ({len(st.session_state.unknown_quizes_part1)}個) は **パターン1 (Saliency)** で学習・テストします。\n"
                      f"* 後半グループ ({len(st.session_state.unknown_quizes_part2)}個) は **パターン2 (下部固定)** で学習・テストします。")


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
                
                # quizes_p1: パターン1（Saliency方式）で処理する未知語のリスト
                # 各要素は (question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index) のタプル
                quizes_p1 = st.session_state.unknown_quizes_part1
                total_p1 = min(len(quizes_p1), NUM_TO_OPTIMIZE)
                
                # quizes_p2: パターン2（下部固定方式）で処理する未知語のリスト
                # 各要素は (question_1, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index) のタプル
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
    if f'{prefix}_camera_ready' not in st.session_state:
        st.session_state[f'{prefix}_camera_ready'] = False
    if f'start_time_{pattern_num}' not in st.session_state:
        st.session_state[f'start_time_{pattern_num}'] = 0
    if f'end_time_{pattern_num}' not in st.session_state:
        st.session_state[f'end_time_{pattern_num}'] = 0
    if f'p{pattern_num}_study_time_logged' not in st.session_state:
        st.session_state[f'p{pattern_num}_study_time_logged'] = False
    if "tracker_thread" not in st.session_state:
        st.session_state.tracker_thread = None
        st.session_state.stop_event = None
        st.session_state.result_container = {"distance": 0.0, "camera_ready": False}
        st.session_state.start_time = None


def render_learning_tab(pattern_num, pattern_name, processed_images_key):
    """学習タブのレンダリング（共通処理）"""
    initialize_learning_session_state(pattern_num)
    
    prefix = f"pattern{pattern_num}"
    processed_images = st.session_state[processed_images_key]
    
    if not processed_images:
        st.info(f"「画像処理」タブで{pattern_name}の画像を処理してください。")
    elif not st.session_state[f'{prefix}_started']:
        if st.button("学習を開始", key=f"{prefix}_start"):
            st.session_state[f'{prefix}_idx'] = 0
            st.session_state[f'{prefix}_started'] = True
            st.session_state[f'{prefix}_camera_ready'] = False
            st.session_state[f'start_time_{pattern_num}'] = time.time()
            
            start_gaze_tracker()
            
            st.session_state[f'p{pattern_num}_study_time_logged'] = False
            st.session_state[f'end_time_{pattern_num}'] = 0
            st.rerun()
    else:
        # カメラ準備中の画面
        if not st.session_state[f'{prefix}_camera_ready']:
            # カメラの準備状態を自動検知
            if st.session_state.result_container.get("camera_ready", False):
                st.session_state[f'{prefix}_camera_ready'] = True
                st.rerun()
            else:
                st.info("📷 カメラを起動しています...")
                st.write("しばらくお待ちください...")
                time.sleep(0.5)  # 少し待ってから再チェック
                st.rerun()
        else:
            # 問題表示
            curr_idx = st.session_state[f'{prefix}_idx']
            
            if curr_idx < len(processed_images):
                if st.button("次の問題", key=f"{prefix}_next"):
                    st.session_state[f'{prefix}_idx'] += 1
                    st.rerun()
                
                # itemの中身（辞書）:
                # - question_1: 質問文1
                # - target: ターゲット単語
                # - question_2: 質問文2
                # - answer: 正解
                # - dammy1, dammy2, dammy3: ダミー選択肢
                # - original_image: 元の画像(PIL)
                # - processed_image: 文字入れ後の画像(PIL)
                # - position: 文字の位置 (x, y)
                # - original_index: 元のインデックス
                item = processed_images[curr_idx]
                st.image(item['processed_image'], use_container_width=True)
                read_text(item['question_1'])
            else:
                st.info("すべての問題を表示し終えました。")
                
                if not st.session_state[f'p{pattern_num}_study_time_logged']:
                    st.session_state[f'end_time_{pattern_num}'] = time.time()
                    study_time = st.session_state[f'end_time_{pattern_num}'] - st.session_state[f'start_time_{pattern_num}']
                    
                    final_distance = stop_gaze_tracker()
                    
                    print(f"\n--- [タブ{pattern_num+2}] {pattern_name} 学習時間: {study_time:.2f} s ---")
                    print(f"\n--- [タブ{pattern_num+2}] {pattern_name} 視線移動距離: {final_distance:.2f} ---")
                    
                    st.session_state[f'p{pattern_num}_study_time_logged'] = True
                
                if st.button("最初からやり直す", key=f"{prefix}_reset"):
                    st.session_state[f'{prefix}_idx'] = 0
                    st.session_state[f'{prefix}_started'] = False
                    st.session_state[f'{prefix}_camera_ready'] = False
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
