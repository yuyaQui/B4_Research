import os
import pickle
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from TranSalNet_Dense import TranSalNet
from experiment_ask_unknown import ask_unknown_words
from experiment_preprocess import DATASETS_PATH
from experiment_image_draw import find_optimal_text_position, draw_answer_text_on_image

MODEL_PATH_DENSE = r'pretrained_models\TranSalNet_Dense.pth'

if __name__ == "__main__":
    with open(os.path.join(DATASETS_PATH, "quizes_and_images.pkl"), "rb") as f:
        experiment_set = pickle.load(f)

    unknown_quizes_and_images = ask_unknown_words(experiment_set)

    random.shuffle(unknown_quizes_and_images)

    num_of_quizes = len(unknown_quizes_and_images)

    for i in range(num_of_quizes // 2):
        if i == 0: 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = TranSalNet()
            model.load_state_dict(torch.load(MODEL_PATH_DENSE))
            model = model.to(device)
            model.eval()
        
        answer = unknown_quizes_and_images[1]
        generated_image = unknown_quizes_and_images[2]
        x, y = find_optimal_text_position(generated_image, model, device)
        image_with_caption = draw_answer_text_on_image(generated_image, answer, x, y)
        print(image_with_caption)