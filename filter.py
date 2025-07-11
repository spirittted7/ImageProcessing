import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import argparse


def create_filter_parser():
    """
    Creates an argument parser for image filtering.

    Returns:
        argparse.ArgumentParser: The argument parser configured for filter options.
    """
    parser = argparse.ArgumentParser(description="Apply various image filters.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_image", type=str, help="Path to save the filtered image.")
    parser.add_argument(
        "--filter_type",
        "-f",
        type=str,
        required=True,
        choices=["sepia", "negative", "color", "temperature", "grayscale"],
        help="Type of filter to apply (required).",
    )
    parser.add_argument(
        "--temperature_type",
        "-t",
        type=str,
        choices=["warm", "cold"],
        help="Temperature type for temperature filter. Required when filter_type is temperature.",
    )
    parser.add_argument(
        "--color_name",
        "-c",
        type=str,
        choices=["red", "green", "blue", "yellow", "cyan", "magenta"],
        help="Color name for color filter. Required when filter_type is color.",
    )
    return parser


def apply_grayscale(args):
    # Applies a grayscale filter to an image
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_img)
        print(f"Grayscale filter applied and saved to {output_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_sepia(args):
    # Applies a sepia filter to an image
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(img, sepia_kernel)
        cv2.imwrite(output_path, sepia_img)
        print(f"Sepia filter applied and saved to {output_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_negative(args):
    # Applies a negative filter to an image
    input_path = args.input_image
    output_path = args.output_image
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")
        negative_img = 255 - img
        cv2.imwrite(output_path, negative_img)
        print(f"Negative filter applied and saved to {output_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_color(args):
    # Applies a color filter to an image.
    input_path = args.input_image
    output_path = args.output_image
    color_name = args.color_name
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")
        color_map = {
            "red": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            "green": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "blue": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "yellow": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            "cyan": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            "magenta": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        }

        if not color_name:
            print("Error: --color_name must be specified with the 'color' filter. "
                  "Available colors: red, green, blue, yellow, cyan, magenta.")
            return  # Exit if color_name is not provided

        color_kernel = color_map.get(color_name.lower())
        if color_kernel is None:
            print(f"Error: Unknown color: {color_name}. Available colors: {', '.join(color_map.keys())}")
            return  # Exit if color_name is invalid

        color_image = cv2.transform(img.astype(np.float32), color_kernel)
        cv2.imwrite(output_path, color_image)
        print(f"Color filter '{color_name}' applied and saved to {output_path}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_temperature(args):
    # Applies a temperature filter (warm or cold) to an image using LUTs.
    input_path = args.input_image
    output_path = args.output_image
    temperature_type = args.temperature_type
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image from {input_path}. Check the path and file format.")

        increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256)).astype(np.uint8)
        decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256)).astype(np.uint8)

        blue_channel, green_channel, red_channel = cv2.split(img)

        if not temperature_type:
            print("Error: --temperature_type must be specified with the 'temperature' filter. "
                  "Available temperature_type: cold, warm.")
            return  # Exit if temperature_type is not provided

        if temperature_type.lower() == "warm":
            red_channel = cv2.LUT(red_channel, increase_table)
            blue_channel = cv2.LUT(blue_channel, decrease_table)
        elif temperature_type.lower() == "cold":
            red_channel = cv2.LUT(red_channel, decrease_table)
            blue_channel = cv2.LUT(blue_channel, increase_table)
        else:
            print(f"Error: Unknown temperature type: {temperature_type}.  Valid options: warm, cold.")
            return

        output_image = cv2.merge((blue_channel, green_channel, red_channel))
        cv2.imwrite(output_path, output_image)
        print(f"Temperature filter '{temperature_type}' applied and saved to {output_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def apply_filter(args):
    """
    Applies a selected filter to an image.

    Args:
        args (argparse.Namespace): Object containing parsed arguments.
    """
    filter_type = args.filter_type
    filter_functions = {
        "sepia": apply_sepia,
        "negative": apply_negative,
        "color": apply_color,
        "temperature": apply_temperature,
        "grayscale": apply_grayscale,  # Add grayscale
    }
    filter_function = filter_functions.get(filter_type)

    if filter_function:
        filter_function(args)
    else:
        print(f"Unknown filter type: {filter_type}")
