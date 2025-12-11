import cv2
import numpy as np
import json
import gzip
import bisect  # 二分探索用（大量のタイムスタンプから目的の時間を高速に探すため）

# ==========================================
# 1. 設定パラメータ（実験環境に合わせて変更する箇所）
# ==========================================
VIDEO_PATH = 'fullstream.mp4'       # 解析する動画ファイル
JSON_PATH = 'livedata.json.gz'      # Tobiiの生データ（GZIP圧縮JSON）
DISPLAY_RES = (1920, 1080)          # 実際のPCモニターの解像度（変換先の基準）

# ★解析時間の指定（秒）
START_TIME = 10.5  # ここから解析開始（実験開始の合図など）
END_TIME = 60.0    # ここで終了（Noneの場合は動画の最後まで）

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
                if 'ts' in data and 'gp3' in data:
                    ts = data['ts']
                    gp3 = data['gp3'] # [x, y, z] (値は0.0～1.0の正規化座標)
                    
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
    """
    画像内のArUcoマーカーを検出し、カメラ画像平面から
    ディスプレイ平面への変換行列(Homography Matrix)を計算する。
    """
    # 画像からマーカーを検出
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
    # 4つ未満しか見つからなかった場合は変換不能なのでNoneを返す
    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten() # 1次元配列にならす
    src_pts = []        # 変換元（カメラ画像上）の座標リスト
    
    # 検出された各マーカーの中心座標を計算
    found_markers = {}
    for (corner, id_num) in zip(corners, ids):
        c = corner[0] # マーカーの4隅の座標群
        cx = (c[0][0] + c[2][0]) / 2 # 対角線の平均＝中心X
        cy = (c[0][1] + c[2][1]) / 2 # 対角線の平均＝中心Y
        found_markers[id_num] = [cx, cy]

    # 定義したID順（左上→右上→右下→左下）に座標を並べ直す
    for target_id in MARKER_IDS:
        if target_id in found_markers:
            src_pts.append(found_markers[target_id])
        else:
            return None # 必要なIDが欠けていたら中断

    # OpenCVの形式に合わせてfloat32のnumpy配列に変換
    src_pts = np.array(src_pts, dtype="float32")
    
    # 変換先（ディスプレイ座標）の定義：解像度の4隅に対応させる
    dst_pts = np.array([
        [0, 0],                           # 左上
        [DISPLAY_RES[0], 0],              # 右上
        [DISPLAY_RES[0], DISPLAY_RES[1]], # 右下
        [0, DISPLAY_RES[1]]               # 左下
    ], dtype="float32")

    # 変換元の4点と、変換先の4点の対応関係から行列Hを計算
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
    
    total_distance = 0.0 # 総移動距離の累積用変数
    prev_pos = None      # 1フレーム前の視線位置（距離計算用）
    valid_frames = 0     # 計算に成功したフレーム数
    
    # --- シーク処理 ---
    # 指定した開始時間まで動画の再生位置を一気に飛ばす（高速化）
    if START_TIME > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, START_TIME * 1000)
        print(f"動画を {START_TIME}秒 地点までスキップしました。")

    print("解析開始...")

    while True:
        # 動画から1フレーム読み込み
        ret, frame = cap.read()
        if not ret:
            break # 動画が終わったらループを抜ける
            
        # 現在のフレームの時刻を取得（秒単位）
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # --- 時間フィルタリング ---
        # 終了時間を超えたら解析終了
        if END_TIME is not None and current_video_time > END_TIME:
            print(f"終了時間 ({END_TIME}秒) に到達しました。")
            break
            
        # 開始時間より前ならスキップ（cap.setで飛ばしたが念のため）
        if current_video_time < START_TIME:
            continue
        
        # --- 同期処理（Synchronization） ---
        # 動画の時刻に最も近い視線データのインデックスを二分探索で探す
        # bisectを使うことで、数万行あるデータから一瞬で見つけられる
        idx = bisect.bisect_left(timestamps, current_video_time)
        if idx >= len(timestamps):
            idx = len(timestamps) - 1
            
        # データの時刻と動画の時刻が大きくズレている場合（0.1秒以上）は
        # 対応データなしとみなしてスキップ
        if abs(timestamps[idx] - current_video_time) > 0.1:
            continue

        raw_gaze = gaze_data[idx] # 生の視線データ [x, y] (0.0~1.0)
        
        # 視線データが無効値（瞬き等でロスト）の場合はスキップ
        if raw_gaze[0] <= 0 or raw_gaze[1] <= 0:
            continue

        # --- 座標変換処理 ---
        # そのフレームの画像からホモグラフィ行列Hを計算
        H = get_homography(frame)
        
        if H is not None:
            # 正規化座標(0.0~1.0)を、カメラ画像のピクセル座標に戻す
            img_h, img_w = frame.shape[:2]
            gx_px = raw_gaze[0] * img_w
            gy_px = raw_gaze[1] * img_h
            
            # perspectiveTransform用の形式 (1, 1, 2) に変換
            # これは [[[x, y]]] という3次元配列の形
            point = np.array([[[gx_px, gy_px]]], dtype="float32")
            
            # 行列Hを使って射影変換（カメラ画像座標 → ディスプレイ座標）
            transformed = cv2.perspectiveTransform(point, H)
            
            screen_x = transformed[0][0][0]
            screen_y = transformed[0][0][1]
            
            # --- 距離計算 ---
            # 変換後の座標がディスプレイの範囲内に収まっているか判定
            if 0 <= screen_x <= DISPLAY_RES[0] and 0 <= screen_y <= DISPLAY_RES[1]:
                
                # 前回の有効な座標があれば、距離を計算して加算
                if prev_pos is not None:
                    # ユークリッド距離 (√((x2-x1)^2 + (y2-y1)^2))
                    dist = np.sqrt((screen_x - prev_pos[0])**2 + (screen_y - prev_pos[1])**2)
                    total_distance += dist
                
                # 今回の座標を「前回」として保存
                prev_pos = (screen_x, screen_y)
                valid_frames += 1

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    
    # 結果表示
    print("-" * 30)
    print(f"解析範囲: {START_TIME}秒 ～ {END_TIME if END_TIME else '最後'} まで")
    print(f"有効フレーム数: {valid_frames}")
    print(f"ディスプレイ上の総移動距離: {total_distance:.2f} px")

if __name__ == "__main__":
    main()