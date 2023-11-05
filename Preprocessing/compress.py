import gzip
import os
import argparse
from tqdm import tqdm


def compress_nifti_path(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(os.listdir(input_dir)):
        out_name = file + ".gz"
        if not file.endswith(".nii.gz"):
            with open(os.path.join(input_dir, file), "rb") as f_in:
                uncompressed_data = f_in.read()
        compressed_data = gzip.compress(uncompressed_data)
        with open(
            os.path.join(output_dir, out_name), "wb"
        ) as f_out:
            f_out.write(compressed_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress Nifti Files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing nifti files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to directory to save compressed nifti files",
    )
    args = parser.parse_args()
    compress_nifti_path(args.input_dir, args.output_dir)
