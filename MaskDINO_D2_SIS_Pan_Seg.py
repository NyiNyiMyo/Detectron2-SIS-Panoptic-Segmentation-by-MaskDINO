import random
import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import math

def run_pan_inference_on_random_images(num_images):
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("sis_val")
    dataset_dicts = DatasetCatalog.get("sis_val")
    sampled_dicts = random.sample(dataset_dicts, num_images)

    # Prepare grid layout
    cols = 4
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 4, 3 * rows))

    for idx, d in enumerate(sampled_dicts):
        img_path = d["file_name"]
        img = cv2.imread(img_path)

        # 1. Run prediction
        outputs = predictor(img)

        panoptic_seg, segments_info = outputs["panoptic_seg"]

        # FIX segments_info
        for seg in segments_info:
            if not seg["isthing"]:
                seg["category_id"] -= len(metadata.thing_classes)

        v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)

        height, width, _ = img.shape

        v._default_line_width = 7 if width > 800 else 5
        v._default_font_size = 40 if width > 800 else 20

        panoptic_result = v.draw_panoptic_seg(
            panoptic_seg.to("cpu"),
            segments_info
        ).get_image()

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(panoptic_result)
        plt.axis("off")
        plt.title(f"Image {idx + 1}")

    plt.tight_layout()
    plt.show()

run_pan_inference_on_random_images(12)