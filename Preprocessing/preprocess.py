import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import argparse
import os


def resample_image(image, new_spacing, reference_image):
    """Resamples an image to a new spacing, using the reference image to determine the size of the output image.

    Args:
      image: The image to resample.
      new_spacing: The new spacing of the image.
      reference_image: The reference image, which is used to determine the size of the output image.

    Returns:
      The resampled image.
    """
    output_origin = reference_image.GetOrigin()
    output_spacing = new_spacing
    output_direction = reference_image.GetDirection()

    resampled_image = sitk.Resample(
        image,
        reference_image.GetSize(),
        sitk.Transform(),
        sitk.sitkBSpline,
        output_origin,
        output_spacing,
        output_direction,
    )

    resample_image = sitk.GetArrayFromImage(resampled_image)

    return np.transpose(resample_image, (2, 1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Nifti Images"
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        help="input directory containing all the images (ct and pet, mixed)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="directory to save the images",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "PET"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "CT"), exist_ok=True)

    all_files = os.listdir(args.in_dir)
    # all_files = sorted(
    #     list(filter(lambda s: ".nii" in s, all_files))
    # )

    pet_files = sorted(
        list(filter(lambda s: "PET" in s, all_files))
    )
    ct_files = sorted(list(filter(lambda s: "CT" in s, all_files)))

    assert len(ct_files) == len(pet_files)

    for ct_filename, pet_filename in tqdm(zip(ct_files, pet_files)):
        assert ct_filename.replace(
            "CT", ""
        ) == pet_filename.replace("PET", "")

        ct_path = os.path.join(args.in_dir, ct_filename)
        pet_path = os.path.join(args.in_dir, pet_filename)

        pet_nib = nib.load(pet_path)

        ct_image = sitk.ReadImage(ct_path)
        pet_image = sitk.ReadImage(pet_path)

        pet_spacing = pet_image.GetSpacing()

        ct_downsampled_image = resample_image(
            ct_image, pet_spacing, pet_image
        )

        ct_hu_min = -200
        ct_hu_max = 1000
        clipped_ct_data = np.clip(
            ct_downsampled_image, ct_hu_min, ct_hu_max
        )

        pet_hu_min = 0
        pet_hu_max = 100
        clipped_pet_data = np.clip(
            pet_nib.get_fdata(), pet_hu_min, pet_hu_max
        )
        ct_data_norm = (
            clipped_ct_data - np.min(clipped_ct_data)
        ) / (np.max(clipped_ct_data) - np.min(clipped_ct_data))
        pet_data_norm = (
            clipped_pet_data - np.min(clipped_pet_data)
        ) / (np.max(clipped_pet_data) - np.min(clipped_pet_data))
        final_ct_image = nib.Nifti1Image(
            ct_data_norm,
            pet_nib.affine,
            pet_nib.header,
        )
        final_pet_image = nib.Nifti1Image(
            pet_data_norm, pet_nib.affine, pet_nib.header
        )

        if pet_data_norm.shape == ct_data_norm.shape:
            nib.save(
                final_ct_image,
                os.path.join(
                    args.out_dir,
                    "CT",
                    os.path.splitext(os.path.basename(ct_filename))[
                        0
                    ]
                    + ".nii.gz",
                ),
            )
            nib.save(
                final_pet_image,
                os.path.join(
                    args.out_dir,
                    "PET",
                    os.path.splitext(
                        os.path.basename(pet_filename)
                    )[0]
                    + ".nii.gz",
                ),
            )
