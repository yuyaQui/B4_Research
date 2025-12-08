import cv2
import mediapipe as mp
import math
import numpy as np

def calculate_gaze_distance():
    # --- 設定 ---
    # ノイズ対策：これ以下の微細な動きは無視する閾値（ピクセル）
    MOVEMENT_THRESHOLD = 0.8
    # 顔の動きを検知する閾値
    FACE_MOVE_THRESHOLD = 1.0    
    # MediaPipeの設定
    mp_face_mesh = mp.solutions.face_mesh
    
    # --- カメラ起動 ---
    # Windowsでカメラ起動が遅い場合は、第2引数に cv2.CAP_DSHOW を指定すると改善することがあります
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("カメラが見つかりません。接続を確認するか、番号を 0 から 1 に変更してください。")
        return 0

    # 画面サイズ取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 変数初期化
    total_distance = 0.0
    prev_left_iris = None
    prev_right_iris = None
    prev_head_pos = None
    
    print("計測を開始します。終了するには画面上で 'q' キーを押してください。")

    # refine_landmarks=True にすることで、虹彩（瞳）の座標が取得可能になります
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("フレームの取得に失敗しました。")
                continue

            # パフォーマンス向上のため書き込み不可にしてMediaPipeに渡す
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            # 描画準備
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 瞬き検出 (Blink Detection)
                    left_eye_top = face_landmarks.landmark[159]
                    left_eye_bottom = face_landmarks.landmark[145]
                    right_eye_top = face_landmarks.landmark[386]
                    right_eye_bottom = face_landmarks.landmark[374]
                    
                    # 縦方向の距離（ピクセル換算）
                    l_dist = abs(left_eye_top.y - left_eye_bottom.y) * height
                    r_dist = abs(right_eye_top.y - right_eye_bottom.y) * height
                    
                    # 閾値以下なら瞬きとみなす
                    BLINK_THRESHOLD = 5.5
                    
                    if l_dist < BLINK_THRESHOLD or r_dist < BLINK_THRESHOLD:
                        # 瞬き中は前の位置情報をリセット
                        prev_left_iris = None
                        prev_right_iris = None
                        cv2.putText(image, "Blink", (30, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue


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

                    # --- 虹彩（瞳）の中心座標を取得 ---
                    # 468: 左目の虹彩中心
                    # 473: 右目の虹彩中心
                    l_pt = face_landmarks.landmark[468]
                    r_pt = face_landmarks.landmark[473]

                    # 0.0~1.0 で正規化されているため、画面サイズ(width, height)を掛けてピクセル座標にする
                    curr_left_iris = np.array([l_pt.x * width, l_pt.y * height])
                    curr_right_iris = np.array([r_pt.x * width, r_pt.y * height])

                    # 画面に緑色の点を描画
                    cv2.circle(image, (int(curr_left_iris[0]), int(curr_left_iris[1])), 3, (0, 255, 0), -1)
                    cv2.circle(image, (int(curr_right_iris[0]), int(curr_right_iris[1])), 3, (0, 255, 0), -1)

                    # --- 距離計算 ---
                    # 前回のフレームがある場合のみ計算
                    if prev_left_iris is not None and prev_right_iris is not None:
                        # 左目と右目の移動距離をそれぞれ計算
                        dist_l = np.linalg.norm(curr_left_iris - prev_left_iris)
                        dist_r = np.linalg.norm(curr_right_iris - prev_right_iris)
                        
                        # 両目の平均移動距離を採用
                        avg_dist = (dist_l + dist_r) / 2.0

                        # ノイズ除去（微細な震えは無視して加算しない）
                        if avg_dist > MOVEMENT_THRESHOLD:
                            total_distance += avg_dist

                    # 現在の座標を「前回の座標」として保存
                    prev_left_iris = curr_left_iris
                    prev_right_iris = curr_right_iris

            # 画面左上に累計距離を表示
            cv2.putText(image, f"Total Distance: {int(total_distance)} px", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 映像を表示
            cv2.imshow('Gaze Distance Tracker', image)

            # 'q'キーで終了
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return total_distance

if __name__ == "__main__":
    final_dist = calculate_gaze_distance()
    print(f"計測終了。最終的な視線移動距離: {final_dist:.2f} px")