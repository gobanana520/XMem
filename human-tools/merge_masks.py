import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
from colors import COLORS


def draw_overlay_image(
    background: np.ndarray, overlay: np.ndarray, alpha: float = 0.8
) -> np.ndarray:
    """Draw an overlay image on top of a background image.
    @param
        background [H, W, BGR]: Background image.
        overlay [H, W, BGR]: Overlay image.
        alpha: Alpha value for the overlay image.
    @return:
        Image with overlay drawn on top [H, W, BGR].
    """
    assert (
        background.shape[:2] == overlay.shape[:2]
    ), "overlay image should be same size as background"
    img_back = background.copy()
    img_over = overlay.copy()
    mask = img_over.astype(bool)
    img_back[mask] = cv2.addWeighted(img_back, alpha, img_over, 1 - alpha, 0)[mask]
    return img_back


def draw_overlay_mask(
    background: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.8,
) -> np.ndarray:
    """Draw an overlay image on top of a background image.
    @param
        background [H, W, BGR]: Background image.
        overlay [H, W, BGR]: Overlay image.
        alpha: Alpha value for the overlay image.
    @return:
        Image with overlay drawn on top [H, W, BGR].
    """
    assert (
        background.shape[:2] == mask.shape[:2]
    ), "overlay image should be same size as background"
    img_back = background.copy()
    mask = mask.copy()
    img_over = np.zeros_like(img_back)
    for idx, class_id in enumerate(np.unique(mask)):
        if class_id == 0:
            continue
        img_over[mask == class_id] = COLORS[idx]
        img_back[mask == class_id] = cv2.addWeighted(
            img_back, alpha, img_over, 1 - alpha, 0
        )[mask == class_id]
    return img_back


def save_image_files_to_video(
    image_files: list,
    save_path: str,
    fps: int = 30,
    format: str = "mp4v",
    is_color: bool = True,
):
    """Convert a list of images to a video.
    @param
        images: List of images [H, W, BGR] or imagefile paths.
        output_file: Path to the output video.
        fps: Frame rate of the video.
        size: Size of the video.
        is_color: If True, the video will be in color.
        format: Format of the video.
    """
    assert os.path.splitext(save_path)[1] == ".mp4", "Save path should be .mp4 file"
    size = cv2.imread(image_files[0]).shape[:2][::-1]
    video = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*format), fps, size, is_color
    )
    for image_file in image_files:
        video.write(cv2.imread(image_file, cv2.IMREAD_COLOR))
    video.release()


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    assert args.generic_path is not None, "Please provide a generic path"
    # assert args.output is not None, "Please provide a output path"
    assert os.path.exists(args.generic_path), "Generic path does not exist"
    # assert os.path.exists(args.output), "Output path does not exist"
    data_folder = os.path.join(args.generic_path, "./")
    mask_folder = os.path.join(args.generic_path, "./data_processing/xmem/output")
    out_folder = os.path.join(args.generic_path, "./data_processing/xmem/visualization")
    for dir in os.listdir(mask_folder):
        bg_dir = os.path.join(data_folder, dir)
        mask_dir = os.path.join(mask_folder, dir)
        assert os.path.exists(bg_dir), "Background path does not exist"
        assert os.path.exists(mask_dir), "Mask path does not exist"
        bg_image_files = sorted(glob(os.path.join(bg_dir, "*.jpg")))
        mask_image_files = sorted(glob(os.path.join(mask_dir, "*.png")))
        assert len(bg_image_files) == len(
            mask_image_files
        ), "Number of images do not match"
        save_folder = os.path.join(out_folder, dir)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print("Processing:", dir)
        for bg_image_file, mask_image_file in tqdm(
            zip(bg_image_files, mask_image_files)
        ):
            bg_image = cv2.imread(bg_image_file)
            mask_image = cv2.imread(mask_image_file, cv2.IMREAD_GRAYSCALE)
            overlay_image = draw_overlay_mask(bg_image, mask_image, args.alpha)
            cv2.imwrite(
                os.path.join(save_folder, os.path.basename(bg_image_file)),
                overlay_image,
            )

        save_image_files_to_video(
            image_files=sorted(glob(os.path.join(save_folder, "*.jpg"))),
            save_path=os.path.join(out_folder, f"{dir}.mp4"),
            fps=30,
        )
