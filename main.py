import argparse
import blur
import filter
import stylization

def main():
    parser = argparse.ArgumentParser(description="Image processing tool.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Subparser for blur
    blur_parser = blur.create_blur_parser()
    subparsers.add_parser("blur", help="gaussian, median, box, bilateral.",
                          parents=[blur_parser], add_help=False)

    # Subparser for filter
    filter_parser = filter.create_filter_parser()
    subparsers.add_parser("filter", help="sepia, negative, color, temperature, grayscale.",
                          parents=[filter_parser], add_help=False)

    # Subparser for stylization
    stylization_parser = stylization.create_stylization_parser()
    subparsers.add_parser("stylization", help="sketch, contouring, emboss, sobel.",
                          parents=[stylization_parser], add_help=False)

    args = parser.parse_args()

    if args.command == "blur":
        blur.blur_image(args)
    elif args.command == "filter":
        filter.apply_filter(args)
    elif args.command == "stylization":
        stylization.apply_stylization(args)
    else:
        print("Invalid command.")


if __name__ == "__main__":
    main()
