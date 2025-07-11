import cv2
import numpy as np
from scipy.signal import convolve2d
import argparse


def create_stylization_parser():
    """
    Creates an argument parser for image stylization.

    Returns:
        argparse.ArgumentParser: The argument parser configured for stylization options.
    """
    parser = argparse.ArgumentParser(
        description="Apply various image stylization effects.",
        formatter_class=argparse.RawTextHelpFormatter  # Keep the formatting
    )
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_image", type=str, help="Path to save the stylized image.")
    parser.add_argument(
        "--stylization_type",
        "-s",
        type=str,
        required=True,
        choices=[
            "sketch",
            "contouring",
            "emboss",
            "sobel",
        ],
        help="Type of stylization to apply. Options: sketch, contouring, emboss, sobel (required).",
    )
    return parser


def apply_pencil_sketch(args):
    # Applies a pencil sketch effect to an image and saves the result.
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        cv2.imwrite(output_path, dst_gray)
        print(f"Pencil sketch effect applied and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_contouring(args):
    # Applies contouring effect to an image and saves the result.
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        dst = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
        cv2.imwrite(output_path, dst)
        print(f"Contouring effect applied and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def rgb_convolve2d(image, kernel):
    # Applies a 2D convolution to each color channel (R, G, B) of an image.
    red = convolve2d(image[:, :, 0], kernel, 'valid')
    green = convolve2d(image[:, :, 1], kernel, 'valid')
    blue = convolve2d(image[:, :, 2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

def apply_emboss(args):
    # Applies a custom convolution filter ("emboss") to an image and saves the result.
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        kernel8 = np.array([[-2, -1, 0],
                            [-1, 1, 1],
                            [0, 1, 2]])
        emboss_img = rgb_convolve2d(img, kernel8[::-1, ::-1]).clip(0, 255).astype(np.uint8)
        cv2.imwrite(output_path, emboss_img)
        print(f"Emboss effect applied and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_sobel(args):
    # Applies a Sobel filter to an image (either color or grayscale) and saves the result.
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        kernel6 = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        test = img.shape
        if len(test) == 3 and test[-1] == 3:
            sobel = rgb_convolve2d(img, kernel6[::-1, ::-1]).clip(0, 255).astype(np.uint8)
        elif len(test) == 2:
            sobel = convolve2d(img, kernel6[::-1, ::-1]).clip(0, 255).astype(np.uint8)
        else:
            print("Unsupported format!")
            return

        cv2.imwrite(output_path, sobel)
        print(f"A Sobel filter applied and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def apply_stylization(args):
    """
    Applies a selected stylization effect to an image.

    Args:
        args (argparse.Namespace): Object containing parsed arguments.
    """
    stylization_type = args.stylization_type
    stylization_functions = {
        "sketch": apply_pencil_sketch,
        "contouring": apply_contouring,
        "emboss": apply_emboss,
        "sobel": apply_sobel,
    }
    stylization_function = stylization_functions.get(stylization_type)

    if stylization_function:
        stylization_function(args)
    else:
        print(f"Unknown stylization type: {stylization_type}")
