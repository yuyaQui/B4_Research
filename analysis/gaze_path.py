import cv2
import numpy as np
import json
import gzip
import bisect  # 二分探索用（大量のタイムスタンプから目的の時間を高速に探すため）

# ==========================================
# 1. 設定パラメータ（実験環境に合わせて変更する箇所）
# ==========================================
VIDEO_PATH = './results/fullstream.mp4'       # 解析する動画ファイル
JSON_PATH = './results/livedata.json.gz'      # Tobiiの生データ（GZIP圧縮JSON）
DISPLAY_RES = (3840, 1483)          # 実際のPCモニターの解像度（変換先の基準）

# ★解析時間の指定（秒）
START_TIME = 19  # ここから解析開始（実験開始の合図など）
STUDY_TIME = 310
END_TIME = STUDY_TIME + START_TIME    # ここで終了（Noneの場合は動画の最後まで）
PLAY_SPEED = 25 # 何倍速か
MARGIN_PX_X = -1000 # 横方向の画面外許容ピクセル数（狭くする）
MARGIN_PX_Y = 300 # 縦方向の画面外許容ピクセル数

# ==========================================
# 2. ArUcoマーカーの設定
# ==========================================
# 4x4のグリッドで50種類のIDを持つ辞書を使用（生成時と同じ設定にする必要あり）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# マーカーIDの配置定義（左上=0, 右上=1, 右下=2, 左下=3 と仮定）
MARKER_IDS = [0, 1, 2, 3] 

# ==========================================
# 3. データ読み込み関数
# ==========================================
def load_gaze_data(json_path):
    """
    GZIP圧縮されたJSONデータを読み込み、時刻と視線座標の配列を返す。
    """
    timestamps = []
    gaze_points = []
    
    print("視線データを読み込んでいます...")
    # gzip.open でテキストモード('rt')として開く
    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
        first_ts = None
        for line in f:
            try:
                # 1行ずつJSONとしてパース
                data = json.loads(line)
                
                # 'ts': タイムスタンプ(マイクロ秒), 'gp3': シーンカメラ上の視線座標(3D)
                if 'ts' in data and 'gp' in data:
                    ts = data['ts']
                    gp3 = data['gp'] # [x, y, z] (値は0.0～1.0の正規化座標)
                    
                    # 最初のデータを基準(0秒)とするためのオフセット取得
                    if first_ts is None:
                        first_ts = ts
                    
                    # マイクロ秒を秒に変換し、相対時間を計算
                    rel_time = (ts - first_ts) / 1_000_000.0
                    
                    timestamps.append(rel_time)
                    gaze_points.append(gp3)
            except:
                continue
    
    # 計算高速化のため numpy配列（ベクトル）に変換して返す
    return np.array(timestamps), np.array(gaze_points)

# ==========================================
# 4. ホモグラフィ行列（射影変換行列）の計算
# ==========================================
def get_homography(img):
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten()
    src_pts = []
    found_markers = {}
    for (corner, id_num) in zip(corners, ids):
        c = corner[0]
        cx = (c[0][0] + c[2][0]) / 2
        cy = (c[0][1] + c[2][1]) / 2
        found_markers[id_num] = [cx, cy]

    for target_id in MARKER_IDS:
        if target_id in found_markers:
            src_pts.append(found_markers[target_id])
        else:
            return None

    src_pts = np.array(src_pts, dtype="float32")
    dst_pts = np.array([
        [0, 0],
        [DISPLAY_RES[0], 0],
        [DISPLAY_RES[0], DISPLAY_RES[1]],
        [0, DISPLAY_RES[1]]
    ], dtype="float32")

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    # 視線データをメモリにロード
    timestamps, gaze_data = load_gaze_data(JSON_PATH)
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 動画のFPS（1秒間のコマ数）を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 描画スキップ設定
    # PLAY_SPEEDが速い場合、毎フレーム描画すると処理落ちして速度が出ないため、
    # 指定倍速に合わせて描画を間引く（見た目の更新頻度をFPS程度に保つ）
    render_skip = max(1, int(PLAY_SPEED))
    
    # 待機時間は最小1ms（スキップされるフレームでは待機もしない）
    wait_time = 1
    
    total_distance = 0.0 # 総移動距離の累積用変数
    prev_pos = None      # ★視線座標の平滑化用バッファ（ノイズ除去）
    gaze_buffer = []     # 過去N フレームの視線座標を保持
    SMOOTH_WINDOW = 5    # 移動平均のウィンドウサイズ
    
    # ★ホモグラフィ行列の時間的平滑化（頭の揺れ対策）
    prev_H = None        # 前フレームのホモグラフィ行列
    H_ALPHA = 0.3        # 平滑化係数（0=完全に前フレーム、1=完全に現フレーム）
    
    valid_frames = 0     # 計算に成功したフレーム数
    frame_count = 0      # 処理フレーム数カウンタ
    
    # ★詳細統計用カウンタ
    marker_lost_frames = 0    # マーカー検出失敗フレーム数
    no_gaze_frames = 0        # 視線データなしフレーム数
    off_screen_frames = 0     # 画面外視線フレーム数
    
    # --- シーク処理 ---
    # 指定した開始時間まで動画の再生位置を一気に飛ばす（高速化）
    if START_TIME > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)
        print(f"動画を {START_TIME}秒 地点までスキップしました。")

    print("解析開始...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_count += 1
        
        # 終了判定
        if END_TIME is not None and current_video_time > END_TIME:
            print("終了時間に到達")
            break
        if current_video_time < START_TIME:
            continue

        # 表示更新をするかどうか
        do_render = (frame_count % render_skip == 0)

        # 描画用画像の準備（表示するときのみコピー）
        if do_render:
            img_h, img_w = frame.shape[:2]
            debug_img = frame.copy() 
        else:
            # 処理用には解像度情報だけあればいいが、
            # raw_gaze計算でimg_w/hを使うので取得しておく
            img_h, img_w = frame.shape[:2]

        # =========================================================
        # 1. ArUcoマーカー検出 & ホモグラフィ行列(H)の計算
        # =========================================================
        # 新しいOpenCVの書き方で検出器を用意
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(frame)
        
        H = None
        found_count = 0
        
        if ids is not None:
            found_count = len(ids)
            
            # 描画: 表示するときのみ
            if do_render:
                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids) 
            
            # マーカーが4つ以上ある場合のみ行列計算を試みる
            if found_count >= 4:
                # --- get_homography関数のロジックをインライン展開 ---
                ids_flat = ids.flatten()
                
                # 最適化: 辞書作成コスト削減
                # found_markers = {} 
                # for ... 
                # → numpyで一括処理もできるが、可読性維持
                found_markers = {}
                for (corner, id_num) in zip(corners, ids_flat):
                    c = corner[0]
                    
                    # ID 0 (左上), ID 1 (右上) は、x座標は中央、y座標は上部(上辺の中心)を採用
                    if id_num in [0, 1]:
                        cx = (c[0][0] + c[2][0]) / 2
                        cy = (c[0][1] + c[1][1]) / 2
                    # ID 2 (右下), ID 3 (左下) は、マーカーの中央を採用
                    else:
                        cx = (c[0][0] + c[2][0]) / 2
                        cy = (c[0][1] + c[2][1]) / 2
                        
                    found_markers[id_num] = [cx, cy]

                # 必要なID(0,1,2,3)が揃っているか確認
                if all(mid in found_markers for mid in MARKER_IDS):
                    src_pts = np.array([found_markers[mid] for mid in MARKER_IDS], dtype="float32")
                    
                    # ★サブピクセル精度でコーナーを再計算（精度向上）
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    src_pts_refined = cv2.cornerSubPix(
                        gray, 
                        src_pts.reshape(-1, 1, 2), 
                        (5, 5),  # 探索ウィンドウサイズ
                        (-1, -1),  # デッドゾーン（なし）
                        criteria
                    ).reshape(-1, 2)
                    
                    dst_pts = np.array([
                        [0, 0], [DISPLAY_RES[0], 0],
                        [DISPLAY_RES[0], DISPLAY_RES[1]], [0, DISPLAY_RES[1]]
                    ], dtype="float32")
                    
                    # ★RANSACを使用してより頑健なホモグラフィを計算
                    H_current, mask = cv2.findHomography(
                        src_pts_refined, 
                        dst_pts, 
                        cv2.RANSAC,  # 外れ値に強い
                        5.0  # 再投影誤差の閾値（ピクセル）
                    )
                    
                    # ★ホモグラフィの時間的平滑化（頭の揺れによるジッター除去）
                    if H_current is not None:
                        if prev_H is not None:
                            # 指数移動平均（EMA）で平滑化
                            H = H_ALPHA * H_current + (1 - H_ALPHA) * prev_H
                        else:
                            H = H_current
                        prev_H = H.copy()
                    else:
                        H = prev_H  # 検出失敗時は前フレームの値を使用


                if do_render and H is not None:
                     # 逆変換行列でディスプレイ上の境界（マージン含む）をカメラ画像上に投影
                    try:
                        H_inv = np.linalg.inv(H)
                        
                        # 拡大された領域の4隅 (左上, 右上, 右下, 左下)
                        margin_corners = np.array([
                            [-MARGIN_PX_X, -MARGIN_PX_Y],
                            [DISPLAY_RES[0] + MARGIN_PX_X, -MARGIN_PX_Y],
                            [DISPLAY_RES[0] + MARGIN_PX_X, DISPLAY_RES[1] + MARGIN_PX_Y],
                            [-MARGIN_PX_X, DISPLAY_RES[1] + MARGIN_PX_Y]
                        ], dtype='float32').reshape(-1, 1, 2)
                        
                        projected_corners = cv2.perspectiveTransform(margin_corners, H_inv)
                        
                        # 描画 (シアン色の枠線)
                        cv2.polylines(debug_img, [np.int32(projected_corners)], True, (255, 255, 0), 2)
                    except Exception as e:
                        pass

        # =========================================================
        # 2. 視線データの取得 & 描画 & 距離計算
        # =========================================================
        idx = bisect.bisect_left(timestamps, current_video_time)
        if idx >= len(timestamps): idx = len(timestamps) - 1
        
        has_gaze = False
        on_screen = False # 画面内を見ているかフラグ
        
        # 同期ズレ許容範囲内なら処理
        if abs(timestamps[idx] - current_video_time) <= 0.1:
            raw_gaze = gaze_data[idx] 
            
            # 視線有効チェック
            if raw_gaze[0] > 0 and raw_gaze[1] > 0:
                has_gaze = True
                
                # A. カメラ映像上の位置（黄色丸）
                gx_px = int(raw_gaze[0] * img_w)
                gy_px = int(raw_gaze[1] * img_h)
                
                if do_render:
                    cv2.circle(debug_img, (gx_px, gy_px), 15, (0, 255, 255), -1)
                
                # B. ディスプレイ座標への変換 & 距離計算
                if H is not None:
                    point = np.array([[[gx_px, gy_px]]], dtype="float32")
                    transformed = cv2.perspectiveTransform(point, H)
                    screen_x = transformed[0][0][0]
                    screen_y = transformed[0][0][1]
                    
                    # 画面内判定 (マージンを設定して大きく外れたら無視)
                    if -MARGIN_PX_X <= screen_x <= DISPLAY_RES[0] + MARGIN_PX_X and -MARGIN_PX_Y <= screen_y <= DISPLAY_RES[1] + MARGIN_PX_Y:
                        on_screen = True
                        valid_frames += 1  # 有効フレーム数をカウント
                        
                        # 距離加算
                        if prev_pos is not None:
                            dist = np.sqrt((screen_x - prev_pos[0])**2 + (screen_y - prev_pos[1])**2)
                            total_distance += dist
                        
                        prev_pos = (screen_x, screen_y)
                    else:
                        on_screen = False
                        prev_pos = None
                        off_screen_frames += 1  # ★画面外カウント
                else:
                    prev_pos = None # マーカーロスト時も連続性を切る
                    if has_gaze:
                        marker_lost_frames += 1  # ★マーカーロストカウント
        
        # 視線データがない場合
        if not has_gaze:
            no_gaze_frames += 1  # ★視線データなしカウント

        # =========================================================
        # 3. 情報表示 (UI) - 表示タイミングのみ
        # =========================================================
        if do_render:
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # マーカー認識数
            color_m = (0, 255, 0) if found_count == 4 else (0, 0, 255)
            cv2.putText(debug_img, f"Markers: {found_count}/4", (30, 50), font, 1.0, color_m, 2)
            
            # 視線状態
            state_text = "No Gaze"
            color_g = (0, 0, 255)
            if has_gaze:
                if H is None: state_text = "Gaze OK (No Markers)"
                elif on_screen: state_text = "On Screen (Tracking)"
                else: state_text = "Off Screen"
                color_g = (0, 255, 0) if on_screen else (255, 255, 0)
                
            cv2.putText(debug_img, f"State: {state_text}", (30, 90), font, 1.0, color_g, 2)

            # ★総移動距離の表示（大きく表示）
            cv2.putText(debug_img, f"Total Dist: {total_distance:.1f} px", (30, 150), font, 1.5, (255, 0, 255), 4)

            # 表示
            cv2.imshow("Debug View", cv2.resize(debug_img, (960, 540)))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    
    # 結果表示
    print("-" * 30)
    print(f"解析範囲: {START_TIME}秒 ～ {END_TIME if END_TIME else '最後'} まで")
    print(f"総フレーム数: {frame_count}")
    print(f"有効フレーム数: {valid_frames}")
    print(f"有効率: {valid_frames / frame_count * 100:.1f}%" if frame_count > 0 else "有効率: N/A")
    print()
    print("【無効フレームの内訳】")
    invalid_frames = frame_count - valid_frames
    print(f"  無効フレーム総数: {invalid_frames} ({invalid_frames / frame_count * 100:.1f}%)")
    print(f"  - マーカー検出失敗: {marker_lost_frames} ({marker_lost_frames / frame_count * 100:.1f}%)")
    print(f"  - 視線データなし: {no_gaze_frames} ({no_gaze_frames / frame_count * 100:.1f}%)")
    print(f"  - 画面外視線: {off_screen_frames} ({off_screen_frames / frame_count * 100:.1f}%)")
    print()
    print(f"ディスプレイ上の総移動距離: {total_distance:.2f} px")
    if valid_frames > 0:
        print(f"平均移動距離/フレーム: {total_distance / valid_frames:.2f} px")

if __name__ == "__main__":
    main()