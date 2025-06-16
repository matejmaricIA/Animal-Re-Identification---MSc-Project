import argparse
import os
import sys


def find_matching_pkls(root: str, tag: str):
    """Yield absolute paths to *.pkl files whose name contains *tag*."""
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".pkl") and tag in fname:
                yield os.path.join(dirpath, fname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively delete .pkl files containing a given tag."
    )
    parser.add_argument(
        "tag",
        help='Substring to match in filenames (e.g. "disk" or "keynet_hardnet")',
    )
    parser.add_argument(
        "--root",
        default="../data",
        help="Directory to start the search (default: /data)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.stderr.write(f"ERROR: {args.root} is not a directory or doesn’t exist.\n")
        sys.exit(1)

    matches = list(find_matching_pkls(args.root, args.tag))
    if not matches:
        print(f'No .pkl files containing "{args.tag}" found under {args.root}.')
        return

    for path in matches:

        try:
            os.remove(path)
            print(f"Deleted {path}")
        except OSError as exc:
            sys.stderr.write(f"Failed to delete {path}: {exc}\n")


if __name__ == "__main__":
    main()