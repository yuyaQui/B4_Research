# ============================================================================
# Imports
# ============================================================================
import os
import pickle
import random
import time
import torch
import streamlit as st
import numpy as np
from PIL import Image
import pyttsx3
import wave
import contextlib

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
SOURCE_PATH = "experiment_quiz"
DEFAULT_NUM_QUIZZES_PER_CONDITION = 30  # 1条件あたりの問題数
READING_SPEED = 130
TIME_LIMIT_BUFFER_SEC = 1.0 # 音声読了後に追加する猶予時間(秒)
AUDIO_DELAY_SEC = 2.0 # 画像表示から音声再生開始までの遅延時間(秒)

# 実験条件定義
CONDITIONS = {
    'A': {'name': '条件A (動的配置・時間固定あり)', 'type': 'saliency', 'time_limit': True},
    'B': {'name': '条件B (動的配置・時間固定なし)', 'type': 'saliency', 'time_limit': False},
    'C': {'name': '条件C (固定配置・時間固定あり)', 'type': 'fixed', 'time_limit': True},
    'D': {'name': '条件D (固定配置・時間固定なし)', 'type': 'fixed', 'time_limit': False},
}

# ラテン方格の順序パターン
LATIN_SQUARE_ORDERS = [
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'D', 'A'],
    ['C', 'D', 'A', 'B'],
    ['D', 'A', 'B', 'C']
]

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
    
    # 実験ブロック情報 (Block 1~4)
    if 'experiment_blocks' not in st.session_state:
        st.session_state.experiment_blocks = [] # 各要素は {'condition_char': 'A', 'quizzes': [], 'processed_images': [], ...}
        st.session_state.quiz_selection_done = False
    
    # モデル
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None

initialize_session_state()

# ============================================================================
# Utility Functions
# ============================================================================
def generate_audio(text: str):
    """テキストから音声ファイルを生成し、パスと長さを返す"""
    try:
        temp_file = f"audio/temp_speech_{hash(text)}.wav"
        
        # ファイルがなければ生成
        if not os.path.exists(temp_file):
            engine = pyttsx3.init()
            engine.setProperty('rate', READING_SPEED)
            engine.save_to_file(text, temp_file)
            engine.runAndWait()
            engine.stop()
        
        # 長さの計算
        duration = 0
        if os.path.exists(temp_file):
            with contextlib.closing(wave.open(temp_file, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                
        return temp_file, duration
            
    except Exception as e:
        print(f"音声生成エラー: {e}")
        return None, 0

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

# ============================================================================
# Image Processing Functions
# ============================================================================
def process_image(quiz_data, index, process_type):
    """画像処理（Saliency または Fixed）"""
    question_1, question_1_read, target, image, question_2, answer, dammy1, dammy2, dammy3, original_index = quiz_data
    
    try:
        # 画像の読み込み
        if isinstance(image, Image.Image):
            generated_image_pil = image
        elif isinstance(image, str):
            if not os.path.exists(image):
                st.error(f"画像パスが見つかりません: {image}")
                return None
            generated_image_pil = Image.open(image)
        else:
            return None
        
        image_copy = generated_image_pil.copy()
        
        # 処理タイプの分岐
        if process_type == 'saliency':
            x, y = find_optimal_text_position(
                image_copy,
                st.session_state.model,
                st.session_state.device
            )
            image_with_caption = draw_answer_text_on_image(image_copy, target, x, y)
        else: # fixed
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
        st.error(f"画像処理エラー ({process_type}): {e}")
        return None

# ============================================================================
# Tab Functions
# ============================================================================
def render_tab1_config():
    """タブ1: 実験設定"""
    st.header("実験設定")
    
    if not st.session_state.experiment_set:
        st.warning("データセットが読み込まれていません。")
        return

    total_loaded = len(st.session_state.experiment_set)
    st.write(f"読み込み済み総クイズ数: {total_loaded} 問")

    # 1条件あたりの問題数設定
    quizzes_per_cond = st.number_input(
        "1条件あたりの問題数（合計4条件実施します）",
        min_value=1,
        max_value=total_loaded // 4,
        value=min(DEFAULT_NUM_QUIZZES_PER_CONDITION, total_loaded // 4),
        step=1
    )
    
    total_needed = quizzes_per_cond * 4
    st.info(f"実験で使用する総問題数: {total_needed} 問")

    # 実験順序（ラテン方格）選択
    order_options = [f"パターン{i+1}: {' → '.join(order)}" for i, order in enumerate(LATIN_SQUARE_ORDERS)]
    selected_order_idx = st.radio(
        "実験条件の実施順序（ラテン方格）",
        options=range(len(order_options)),
        format_func=lambda x: order_options[x],
        index=0
    )
    
    st.markdown("""
    **条件詳細:**
    - **A**: 動的配置 + 時間固定あり(音声長+1s)
    - **B**: 動的配置 + 時間固定なし
    - **C**: 固定配置 + 時間固定あり(音声長+1s)
    - **D**: 固定配置 + 時間固定なし
    """)

    if st.button("実験条件を確定してセットアップ", key="setup_experiment"):
        # リセット
        st.session_state.experiment_blocks = []
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("block_") or "study_time" in k or "quiz_" in k]
        for k in keys_to_remove:
            if k in st.session_state:
                del st.session_state[k]
        
        full_set = st.session_state.experiment_set
        
        # データセットにID付与
        formatted_set = []
        for i, item in enumerate(full_set):
            formatted_set.append(item + (i,))
        
        # 使用するデータセットを切り出し
        used_set = formatted_set[:total_needed]
        
        # 4つのパートに分割 (Part 1, 2, 3, 4) ... データ自体の並びは固定（順序効果対策はブロック順序で行う）
        parts = []
        for i in range(4):
            start = i * quizzes_per_cond
            end = start + quizzes_per_cond
            parts.append(used_set[start:end])
        
        # 選択された条件順序
        order_chars = LATIN_SQUARE_ORDERS[selected_order_idx]
        
        # ブロック作成
        for i, char in enumerate(order_chars):
            cond = CONDITIONS[char]
            block_data = {
                'id': i + 1,                 # Block Number (1-based)
                'condition_char': char,      # A, B, C, D
                'condition_name': cond['name'],
                'type': cond['type'],        # saliency / fixed
                'time_limit': cond['time_limit'],
                'quizzes': parts[i],         # 割り当てられたデータパート (順序に依存して決定)
                'processed_images': []       # 処理済み画像置き場
            }
            # クイズ内のシャッフルはここで行うか、表示時に行うか
            # 実験統制上、ブロック内はランダムにするのが一般的
            random.shuffle(block_data['quizzes'])
            
            st.session_state.experiment_blocks.append(block_data)
        
        st.session_state.quiz_selection_done = True
        st.success("✅ セットアップ完了")
        st.rerun()

    if st.session_state.quiz_selection_done:
        st.write("---")
        st.subheader("割り当て結果")
        for block in st.session_state.experiment_blocks:
            st.write(f"**Block {block['id']}**: {block['condition_name']} - {len(block['quizzes'])}問")

def render_tab2_processing():
    """タブ2: 画像処理"""
    if not st.session_state.quiz_selection_done:
        st.warning("設定タブでセットアップを実行してください。")
        return

    st.write("各ブロックの画像処理を行います。")
    
    if st.button("全ブロックの画像を処理開始"):
        load_model()
        if st.session_state.model is None:
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_blocks = len(st.session_state.experiment_blocks)
        
        for b_idx, block in enumerate(st.session_state.experiment_blocks):
            quizzes = block['quizzes']
            total_q = len(quizzes)
            block['processed_images'] = []
            process_type = block['type'] # saliency or fixed
            
            status_text.text(f"Block {block['id']} ({block['condition_name']}) 処理中...")
            
            for q_idx, quiz in enumerate(quizzes):
                # 進捗計算: ((現在のブロックまでの完了数) + (現在の問題割合)) / 全ブロック数
                current_global_progress = (b_idx + (q_idx / total_q)) / total_blocks
                progress_bar.progress(current_global_progress)
                
                res = process_image(quiz, q_idx, process_type)
                if res:
                    block['processed_images'].append(res)
        
        progress_bar.progress(1.0)
        status_text.text("全処理完了！")
        st.success("全ての画像の処理が完了しました。")

def initialize_block_state(block_prefix):
    if f'{block_prefix}_started' not in st.session_state:
        st.session_state[f'{block_prefix}_started'] = False
    if f'{block_prefix}_idx' not in st.session_state:
        st.session_state[f'{block_prefix}_idx'] = 0
    if f'{block_prefix}_logged' not in st.session_state:
        st.session_state[f'{block_prefix}_logged'] = False
    if f'{block_prefix}_start_time' not in st.session_state:
        st.session_state[f'{block_prefix}_start_time'] = 0

def render_learning_tab(block_index):
    """学習タブ (Block単位)"""
    if not st.session_state.quiz_selection_done or block_index >= len(st.session_state.experiment_blocks):
        st.warning("セットアップ未完了または無効なブロックです。")
        return
    
    block = st.session_state.experiment_blocks[block_index]
    prefix = f"block_{block['id']}_learn"
    
    initialize_block_state(prefix)
    
    st.subheader(f"{block['condition_name']}")
    
    processed = block['processed_images']
    if not processed:
        st.error("画像処理が完了していません。")
        return

    # スタイル定義 (Audio非表示)
    st.markdown("<style>.stAudio {display: none;}</style>", unsafe_allow_html=True)
    
    # Timeout state initialization
    if f'{prefix}_timeout' not in st.session_state:
        st.session_state[f'{prefix}_timeout'] = False

    if not st.session_state[f'{prefix}_started']:
        if st.button("学習を開始", key=f"{prefix}_btn_start"):
            st.session_state[f'{prefix}_started'] = True
            st.session_state[f'{prefix}_idx'] = 0
            st.session_state[f'{prefix}_start_time'] = time.time()
            st.session_state[f'{prefix}_timeout'] = False
            st.rerun()
    else:
        idx = st.session_state[f'{prefix}_idx']
        
        if idx < len(processed):
            item = processed[idx]
            
            # --- Hidden Button for Timeout Trigger ---
            # Javascriptからクリックするための隠しボタン
            # タブ間で干渉しないよう、テキストにPrefixを含める
            trigger_btn_text = f"TimeoutTrigger_{prefix}"
            if st.button(trigger_btn_text, key=f"{prefix}_timeout_btn"):
                st.session_state[f'{prefix}_timeout'] = True
                st.rerun()
            
            # Note: ボタンの非表示処理は後方の st.components.v1.html 内のJSで行う
            # -----------------------------------------

            # ナビゲーション
            next_btn_text = f"次の問題へ (Block {block['id']})"
            if st.button(next_btn_text, key=f"{prefix}_next"):
                st.session_state[f'{prefix}_idx'] += 1
                st.session_state[f'{prefix}_timeout'] = False
                st.rerun()
            
            # 音声生成とデュレーション取得
            audio_path, audio_duration = generate_audio(item['question_1_read'])

            # Javascript (Time Limit + Enter Key)
            time_limit_sec = 10 # デフォルト
            is_time_limit = block['time_limit']
            
            if is_time_limit:
                # 制限時間 = 遅延(3s) + 音声長 + 猶予
                time_limit_sec = AUDIO_DELAY_SEC + audio_duration + TIME_LIMIT_BUFFER_SEC
            
            timer_div = ""
            timer_js = ""
            comp_height = 0
            
            # タイムアウトしているかどうかで挙動を変える
            is_timeout = st.session_state[f'{prefix}_timeout']

            if is_time_limit:
                if not is_timeout:
                    # --- カウントダウン中 ---
                    timer_div = "" 
                    # タイマーJS: 時間が来たら "TimeoutTrigger" ボタンを押す
                    timer_js = f"""
                    // Index: {idx} (force reload)
                    let timeLeft = {time_limit_sec};
                    
                    // 開始時刻を記録
                    const startTime = Date.now();
                    const durationMs = timeLeft * 1000;
                    
                    const interval = setInterval(() => {{
                        // 経過時間から計算
                        const elapsed = Date.now() - startTime;
                        const remaining = Math.max(0, (durationMs - elapsed) / 1000);
                                                
                        if(remaining <= 0) {{
                            clearInterval(interval);
                            // TimeoutTriggerボタンをクリック
                            const btns = window.parent.document.getElementsByTagName('button');
                            for(let b of btns) {{
                                // このブロック専用のトリガーボタンをクリック
                                if(b.innerText.trim() === '{trigger_btn_text}') {{ 
                                    b.click(); 
                                    break; 
                                }}
                            }}
                        }}
                    }}, 100);
                    """
                else:
                    # --- タイムアウト後 ---
                    # タイマーJSなし
                    timer_js = ""
            
            # Enterキーのリスナー
            # 1. 時間制限なし -> 常に有効
            # 2. 時間制限あり & タイムアウト前 -> 無効
            # 3. 時間制限あり & タイムアウト後 -> 有効
            enable_enter = False
            if not is_time_limit:
                enable_enter = True
            elif is_time_limit and is_timeout:
                enable_enter = True
            
            # フラグの状態をJSに渡す（これは毎回更新する）
            js_enable_flag = "true" if enable_enter else "false"
            js_enable_flag = "true"
            
            enter_key_listener = f"""
            const parentDoc = window.parent.document;
            
            // 現在のEnter有効/無効状態を更新
            parentDoc['allow_enter_{prefix}'] = {js_enable_flag};
           
           
                console.log('Enter key listener added');
                parentDoc.addEventListener('keydown', (e) => {{
                    console.log('Enter key pressed');
                    if (e.key === 'Enter') {{
                        // フラグがTrueのときのみ実行
                        if (parentDoc['allow_enter_{prefix}'] === true) {{
                            const btns = parentDoc.getElementsByTagName('button');
                            for(let b of btns) {{
                                // このブロック専用の次へボタンをクリック
                                if(b.innerText.includes('{next_btn_text}')) {{ 
                                    b.click(); 
                                    e.preventDefault(); 
                                    e.stopPropagation();
                                    break; 
                                }}
                            }}
                        }}
                    }}
                }});
                parentDoc['_listener_{prefix}'] = true;
         
            """
            
            # 音声再生用スクリプト (HTML埋め込み)
            audio_html = ""
            should_show_content = True
            if is_time_limit and is_timeout:
                should_show_content = False
            
            if should_show_content and audio_path and os.path.exists(audio_path):
                import base64
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                b64_audio = base64.b64encode(audio_bytes).decode()
                
                audio_id = f"audio_{prefix}_{idx}"
                # JSで遅延再生
                audio_html = f"""
                <audio id="{audio_id}" style="display:none">
                    <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                </audio>
                <script>
                    setTimeout(function() {{
                        var audio = document.getElementById("{audio_id}");
                        if(audio) {{
                            audio.play().catch(e => console.log("Audio play failed:", e));
                        }}
                    }}, {int(AUDIO_DELAY_SEC * 1000)});
                </script>
                """

            st.components.v1.html(
                f"""
                {timer_div}
                {audio_html}
                <script>
                // Hidden Button Hiding Logic
                if (!window.hideBtnInterval_{prefix}) {{
                    function hideTriggerButton() {{
                        const btns = window.parent.document.getElementsByTagName('button');
                        for (let btn of btns) {{
                            // "TimeoutTrigger_" で始まるボタンをすべて隠す
                            if (btn.innerText.trim().startsWith('TimeoutTrigger_')) {{
                                btn.style.display = 'none';
                            }}
                        }}
                    }}
                    hideTriggerButton();
                    window.hideBtnInterval_{prefix} = setInterval(hideTriggerButton, 200);
                }}

                {enter_key_listener}
                {timer_js}
                </script>
                """,
                height=comp_height
            )

            # 画像表示
            if should_show_content:
                st.image(item['processed_image'], width='stretch')
            else:
                st.info("終了。Enterキーを押して次の問題へ進んでください。")

        else:
            st.success("Block学習終了。テストタブへ進んでください。")
            if not st.session_state[f'{prefix}_logged']:
                duration = time.time() - st.session_state[f'{prefix}_start_time']
                print(f"Block {block['id']} Learning Time: {duration:.2f}s")
                st.session_state[f'{prefix}_logged'] = True
            
            if st.button("リセット (デバッグ用)", key=f"{prefix}_reset"):
                st.session_state[f'{prefix}_started'] = False
                st.rerun()

def render_quiz_tab(block_index):
    """テストタブ (Block単位)"""
    if not st.session_state.quiz_selection_done or block_index >= len(st.session_state.experiment_blocks):
        st.warning("無効なブロックです。")
        return

    block = st.session_state.experiment_blocks[block_index]
    prefix = f"block_{block['id']}_test"
    
    # テスト用State初期化
    if f'{prefix}_started' not in st.session_state:
        st.session_state[f'{prefix}_started'] = False
    if f'{prefix}_idx' not in st.session_state:
        st.session_state[f'{prefix}_idx'] = 0
    if f'{prefix}_score' not in st.session_state:
        st.session_state[f'{prefix}_score'] = 0
    if f'{prefix}_answered' not in st.session_state:
        st.session_state[f'{prefix}_answered'] = False
    if f'{prefix}_ended' not in st.session_state:
        st.session_state[f'{prefix}_ended'] = False
    
    st.header(f"Block {block['id']}: テストフェーズ")
    
    data = block['processed_images'] # 正解データなどを含む
    if not data:
        st.error("データがありません。")
        return
        
    total_q = len(data)

    if not st.session_state[f'{prefix}_started']:
        st.write(f"全 {total_q} 問のテストを開始ます。")
        if st.button("テスト開始", key=f"{prefix}_test_start"):
            st.session_state[f'{prefix}_started'] = True
            st.session_state[f'{prefix}_idx'] = 0
            st.session_state[f'{prefix}_score'] = 0
            # テスト順序のランダマイズが必要ならここでindex listを作るが、今回はそのまま
            st.rerun()
    else:
        idx = st.session_state[f'{prefix}_idx']
        
        if idx < total_q:
            item = data[idx]
            
            st.subheader(f"Q {idx+1} / {total_q}")
            st.write(item['question_2'])
            
            # 選択肢生成 (キャッシュしてリロード耐性をもたせる)
            opt_key = f"{prefix}_opt_{idx}"
            if opt_key not in st.session_state:
                opts = [item['answer'], item['dammy1'], item['dammy2'], item['dammy3']]
                random.shuffle(opts)
                st.session_state[opt_key] = opts
            
            options = st.session_state[opt_key]
            
            user_ans = st.radio("選択肢", options, key=f"{prefix}_radio_{idx}", index=None)
            
            if st.button("回答", key=f"{prefix}_submit"):
                if user_ans:
                    if user_ans == item['answer']:
                        st.session_state[f'{prefix}_score'] += 1
                    st.session_state[f'{prefix}_idx'] += 1
                    st.rerun()
                else:
                    st.warning("選択してください。")
        else:
            score = st.session_state[f'{prefix}_score']
            st.subheader(f"テスト終了！ スコア: {score} / {total_q}")
            
            if st.session_state[f'{prefix}_ended'] == False:
                print(f"Block {block['id']} Test Result: {score}/{total_q}")
                st.session_state[f'{prefix}_ended'] = True
                st.rerun()
            
            if st.button("再テスト (デバッグ用)", key=f"{prefix}_retry"):
                 st.session_state[f'{prefix}_started'] = False
                 st.session_state[f'{prefix}_ended'] = False
                 st.rerun()

# ============================================================================
# Main UI Structure
# ============================================================================
tabs = st.tabs([
    "設定", "画像処理",
    "Block1 学習", "Block1 テスト",
    "Block2 学習", "Block2 テスト",
    "Block3 学習", "Block3 テスト",
    "Block4 学習", "Block4 テスト"
])

with tabs[0]: render_tab1_config()
with tabs[1]: render_tab2_processing()
with tabs[2]: render_learning_tab(0)
with tabs[3]: render_quiz_tab(0)
with tabs[4]: render_learning_tab(1)
with tabs[5]: render_quiz_tab(1)
with tabs[6]: render_learning_tab(2)
with tabs[7]: render_quiz_tab(2)
with tabs[8]: render_learning_tab(3)
with tabs[9]: render_quiz_tab(3)