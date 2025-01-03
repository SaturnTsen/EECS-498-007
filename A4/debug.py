from a4_helper import hello_helper
from one_stage_detector import hello_one_stage_detector
from common import hello_common
from a4_helper import VOC2007DetectionTiny
import os
import time
import torch
import multiprocessing

if __name__ == '__main__':
    GOOGLE_DRIVE_PATH = '.'

    hello_common()
    hello_one_stage_detector()
    hello_helper()

    one_stage_detector_path = os.path.join(
        GOOGLE_DRIVE_PATH, "one_stage_detector.py")
    one_stage_detector_edit_time = time.ctime(
        os.path.getmtime(one_stage_detector_path)
    )
    print("one_stage_detector.py last edited on %s" %
          one_stage_detector_edit_time)

    if torch.cuda.is_available():
        print("Good to go!")
        DEVICE = torch.device("cuda")
    else:
        print("Please set GPU via Edit -> Notebook Settings.")
        DEVICE = torch.device("cpu")

    GOOGLE_DRIVE_PATH = '.'
    NUM_CLASSES = 20
    BATCH_SIZE = 16
    IMAGE_SHAPE = (224, 224)
    NUM_WORKERS = 10

    # NOTE: Set `download=True` for the first time when you set up Google Drive folder.
    # Turn it back to `False` later for faster execution in the future.
    # If this hangs, download and place data in your drive manually as shown above.
    train_dataset = VOC2007DetectionTiny(
        GOOGLE_DRIVE_PATH, "train", image_size=IMAGE_SHAPE[0],
        download=False  # True (for the first time)
    )
    val_dataset = VOC2007DetectionTiny(
        GOOGLE_DRIVE_PATH, "val", image_size=IMAGE_SHAPE[0])

    print(f"Dataset sizes: train ({
        len(train_dataset)}), val ({len(val_dataset)})")

    # `pin_memory` speeds up CPU-GPU batch transfer, `num_workers=NUM_WORKERS` loads data
    # on the main CPU process, suitable for Colab.

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    # Use batch_size = 1 during inference - during inference we do not center crop
    # the image to detect all objects, hence they may be of different size. It is
    # easier and less redundant to use batch_size=1 rather than zero-padding images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
    )

    train_loader_iter = iter(train_loader)
    start_time = time.time()
    image_paths, images, gt_boxes = next(train_loader_iter)
    # print(f"image paths           : {image_paths}")
    print(f"image batch has shape : {images.shape}")
    print(f"gt_boxes has shape    : {gt_boxes.shape}")

    print(f"Five boxes per image  :")
    # print(gt_boxes[:, :5, :])

    end_time = time.time()
    print(f"Access batches took {end_time - start_time:.3f} s")


#%% 
import multiprocessing
multiprocessing.get_start_method()