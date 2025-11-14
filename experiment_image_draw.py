import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torchvision import transforms
from experiment_data_process import preprocess_img, postprocess_img

MIN_PEAKS_TO_FIND = 2
MAX_PEAKS_TO_FIND = 5
PEAK_SALIENCY_THRESHOLD = 180
MASK_SALIENCY_THRESHOLD_FOR_RADIUS = 100
MIN_MASK_RADIUS = 50

FONT_PATH = "C:/Windows/Fonts/meiryob.ttc"
FONT_SIZE = 40
FILL_COLOR = "#00ff00"
STROKE_COLOR = "black"
STROKE_WIDTH = 3

POS_OFFSET = 40

def find_optimal_text_position(generated_image, model, device):
    """Saliencyマップに基づきテキストの最適配置座標 (x, y) を返す。"""
    org_img = np.array(generated_image.convert("RGB"))
    img = preprocess_img(org_img)
    img = np.array(img) / 255
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    img = torch.from_numpy(img).type(torch.cuda.FloatTensor).to(device)
    pred_saliency_tensor = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency_tensor.squeeze())
    saliency_map_np = postprocess_img(pic, org_img)

    img_width, img_height = generated_image.size
    h, w = saliency_map_np.shape

    print(f"\n🔍 反復的なピーク検出を開始 (最小{MIN_PEAKS_TO_FIND}個、最大{MAX_PEAKS_TO_FIND}個、しきい値 > {PEAK_SALIENCY_THRESHOLD})")

    all_x_peaks = []
    all_y_peaks = []
    all_vals_peaks = []

    temp_map = saliency_map_np.copy()

    Y_grid, X_grid = np.ogrid[:h, :w]

    y_low, x_low = np.where(saliency_map_np < MASK_SALIENCY_THRESHOLD_FOR_RADIUS)

    for i in range(MAX_PEAKS_TO_FIND):
        val_n = np.max(temp_map)

        if val_n < PEAK_SALIENCY_THRESHOLD and i >= MIN_PEAKS_TO_FIND:
            break

        loc_n_flat = np.argmax(temp_map)
        y_n, x_n = np.unravel_index(loc_n_flat, temp_map.shape)

        all_x_peaks.append(x_n)
        all_y_peaks.append(y_n)
        all_vals_peaks.append(val_n)

        mask_radius = MIN_MASK_RADIUS
        if y_low.size > 0:
            distance_to_low = np.sqrt((x_low - x_n) ** 2 + (y_low - y_n) ** 2)
            calculated_radius = np.min(distance_to_low)
            mask_radius = max(calculated_radius, MIN_MASK_RADIUS)
        else:
            print(f"(半径計算) Saliency < {MASK_SALIENCY_THRESHOLD_FOR_RADIUS} の領域なし。最小半径({MIN_MASK_RADIUS})を使用")

        dist_from_peak_n = np.sqrt((X_grid - x_n) ** 2 + (Y_grid - y_n) ** 2)
        temp_map[dist_from_peak_n <= mask_radius] = 0

    final_x_peaks = np.array(all_x_peaks)
    final_y_peaks = np.array(all_y_peaks)
    final_vals_peaks = np.array(all_vals_peaks)

    # 0ピークの場合はSaliency最大値を採用
    if final_x_peaks.size == 0:
        print(f"⚠️ Saliency > {PEAK_SALIENCY_THRESHOLD} のピークが0個でした。Saliency最大値をピークとします。")
        val1 = np.max(saliency_map_np)
        loc1_flat = np.argmax(saliency_map_np)
        y1, x1 = np.unravel_index(loc1_flat, saliency_map_np.shape)

        final_x_peaks = np.array([x1])
        final_y_peaks = np.array([y1])
        final_vals_peaks = np.array([val1])

    for i in range(final_x_peaks.size):
        print(f"  ピーク{i+1}: (x={final_x_peaks[i]}, y={final_y_peaks[i]}) (値: {final_vals_peaks[i]:.2f})")

    total_weight = np.sum(final_vals_peaks.astype(float))

    if total_weight > 0:
        weight_x_sum = np.sum(final_x_peaks.astype(float) * final_vals_peaks.astype(float))
        weight_y_sum = np.sum(final_y_peaks.astype(float) * final_vals_peaks.astype(float))

        final_x = int(weight_x_sum / total_weight)
        final_y = int(weight_y_sum / total_weight)
    else:
        print("重みの合計が0です。画像の中心を使います。")
        final_x = img_width // 2
        final_y = img_height // 2

    print(f"最適なテキスト配置座標 (x, y) : ({final_x}, {final_y})")
    return final_x, final_y

def find_lower_text_position_and_draw(
        generated_image, answer_text,
        font_path=FONT_PATH, font_size=FONT_SIZE,
        fill_color=FILL_COLOR, stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH, pos_offset=POS_OFFSET
    ):
        """画像中心にテキストを描画し、描画済み画像を返す。"""
        draw = ImageDraw.Draw(generated_image)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 指定されたフォント '{font_path}' が見つかりませんでした。デフォルトフォントを使用します")
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_half_width = text_width // 2
        text_half_height = text_height // 2
        image_width, image_height = generated_image.size
        final_x = image_width // 2
        final_y = image_height  - text_height
        text_x = final_x - text_half_width
        text_y = final_y - text_half_height
        draw.text(
            (text_x, text_y),
            answer_text,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )
        return generated_image

def find_central_text_position_and_draw(
        generated_image, answer_text,
        font_path=FONT_PATH, font_size=FONT_SIZE,
        fill_color=FILL_COLOR, stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH, pos_offset=POS_OFFSET
    ):
        """画像中心にテキストを描画し、描画済み画像を返す。"""
        draw = ImageDraw.Draw(generated_image)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 指定されたフォント '{font_path}' が見つかりませんでした。デフォルトフォントを使用します")
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_half_width = text_width // 2
        text_half_height = text_height // 2
        image_width, image_height = generated_image.size
        final_x = image_width // 2
        final_y = image_height  - text_height
        text_x = final_x - text_half_width
        text_y = final_y - text_half_height
        draw.text(
            (text_x, text_y),
            answer_text,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )
        return generated_image


def draw_answer_text_on_image(
        generated_image, answer_text, x, y,
        font_path=FONT_PATH, font_size=FONT_SIZE,
        fill_color=FILL_COLOR, stroke_color=STROKE_COLOR,
        stroke_width=STROKE_WIDTH, pos_offset=POS_OFFSET
    ):
        """周辺パッチの画素分散が小さい位置を探索し、テキストを描画して返す。"""
        draw = ImageDraw.Draw(generated_image)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"警告: 指定されたフォント '{font_path}' が見つかりませんでした。デフォルトフォントを使用します")
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), answer_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_half_width = text_width // 2
        text_half_height = text_height // 2
        img_width, img_height = generated_image.size
        left = x - pos_offset
        top = y - pos_offset
        right = x + pos_offset
        bottom = y + pos_offset
        factor_flag = float('inf')
        final_x, final_y = x, y
        for i in range(left, right):
            for j in range(top, bottom):
                crop_left = max(0, i - text_half_width)
                crop_top = max(0, j - text_half_height)
                crop_right = min(img_width, i + text_half_width)
                crop_bottom = min(img_height, j + text_half_height)
                if crop_left < crop_right and crop_top < crop_bottom:
                    patch_pil = generated_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                    patch_gray = patch_pil.convert("L")
                    patch_np = np.array(patch_gray)
                    if patch_np.size == 0:
                        continue
                    intensity_std = np.std(patch_np)
                    if intensity_std < factor_flag:
                        factor_flag = intensity_std
                        final_x, final_y = i, j
        print(f"✅ スキャン完了。")
        print(f"💡 最適なテキスト配置座標 (x, y): ({final_x}, {final_y})")
        text_x = final_x - text_half_width
        text_y = final_y - text_half_height
        if text_x < 0:
            text_x = 0
        if text_x + text_width > img_width:
            text_x = img_width - text_width
        if text_y < 0:
            text_y = 0
        if text_y + text_height > img_height:
            text_y = img_height - text_height
        draw.text(
            (text_x, text_y),
            answer_text,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color
        )
        return generated_image
