import cv2
import argparse


def create_blur_parser():
    """
    Creates an argument parser for blur options.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Blurs an image using various blur filters.")

    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_image", type=str, help="Path to save the blurred image.")
    parser.add_argument(
        "--blur_type",
        "-b",
        type=str,
        default="gaussian",
        choices=["gaussian", "median", "box", "bilateral"],
        help="Blur type. Valid options: gaussian, median, box, bilateral (default: gaussian).",
    )
    parser.add_argument(
        "--radius",
        "-r",
        type=int,
        default=5,
        help="Blur radius (default: 5). Use only odd numbers.",
    )
    parser.add_argument(
        "--sigma_x",
        type=float,
        default=0,
        help="Standard deviation in the X direction (Used only with gaussian). Defaults to 0.",
    )
    parser.add_argument(
        "--sigma_y",
        type=float,
        default=0,
        help="Standard deviation in the Y direction (Used only with gaussian). Defaults to 0.",
    )
    parser.add_argument(
        "--sigma_color",
        type=float,
        default=75,
        help="Used only with bilateral. Defaults to 75",
    )
    parser.add_argument(
        "--sigma_space",
        type=float,
        default=75,
        help="Used only with bilateral. Defaults to 75",
    )
    return parser


def blur_image(args):
    """
    Blurs an image with the specified blur type and parameters.

    Args:
        args (argparse.Namespace): An object containing parsed arguments.
    """
    input_path = args.input_image
    output_path = args.output_image
    blur_type = args.blur_type
    radius = args.radius
    sigma_x = args.sigma_x
    sigma_y = args.sigma_y
    sigma_color = args.sigma_color
    sigma_space = args.sigma_space

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        if blur_type == "gaussian":
            blurred_img = cv2.GaussianBlur(img, (radius, radius), sigmaX=sigma_x, sigmaY=sigma_y)
        elif blur_type == "median":
            blurred_img = cv2.medianBlur(img, radius)
        elif blur_type == "box":
            blurred_img = cv2.blur(img, (radius, radius))
        elif blur_type == "bilateral":
            blurred_img = cv2.bilateralFilter(img, radius, sigma_color, sigma_space)
        else:
            raise ValueError(f"Unknown blur type: {blur_type}. Valid types: gaussian, median, box, bilateral.")

        cv2.imwrite(output_path, blurred_img)
        print(f"Image blurred ({blur_type}) successfully and saved to {output_path}")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
