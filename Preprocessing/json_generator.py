import argparse
import os
import random
import json
# import nibabel as nib
import numpy as np

random.seed(42)

def ssl_generator(args):
    ct_files = sorted(os.listdir(os.path.join(args.path, "CT")))
    pet_files = sorted(os.listdir(os.path.join(args.path, "PET")))

    assert len(ct_files) == len(pet_files)

    cut_index = int(args.ratio * len(ct_files))

    zipped = list(zip(ct_files, pet_files))
    random.shuffle(zipped)
    ct_files, pet_files = zip(*zipped)

    ct_files_train = ct_files[cut_index:]
    pet_files_train = pet_files[cut_index:]
    ct_files_val = ct_files[:cut_index]
    pet_files_val = pet_files[:cut_index]

    print(
        f"# Training samples: {len(ct_files_train)}\t# Validation samples: {len(ct_files_val)}"
    )

    training = []
    validation = []

    for ct, pet in zip(ct_files_train, pet_files_train):
        assert ct.replace("CT", "") == pet.replace("PET", "")
        training.append({"image": [f"./CT/{ct}", f"./PET/{pet}"]})
    for ct, pet in zip(ct_files_val, pet_files_val):
        assert ct.replace("CT", "") == pet.replace("PET", "")
        validation.append({"image": [f"./CT/{ct}", f"./PET/{pet}"]})

    to_json = {"training": training, "validation": validation}

    with open(args.json, "w") as f:
        json.dump(to_json, f)


def supervised_generator(args):
    ct_files = sorted(os.listdir(os.path.join(args.path, "CT")))
    pet_files = sorted(os.listdir(os.path.join(args.path, "PET")))
    mask_files = sorted(os.listdir(os.path.join(args.path, "Mask")))

    

    assert len(ct_files) == len(pet_files) == len(mask_files)

    cut_index = int(args.ratio * len(ct_files))

    zipped = list(zip(ct_files, pet_files, mask_files))
    random.shuffle(zipped)
    ct_files, pet_files, mask_files = zip(*zipped)

    folds_ct = np.array_split(ct_files, args.folds, axis=0)
    folds_pet = np.array_split(pet_files, args.folds, axis=0)
    folds_mask = np.array_split(mask_files, args.folds, axis=0)

    for i in range(args.folds):

        
        ct_files_val, pet_files_val, mask_files_val = folds_ct[i], folds_pet[i], folds_mask[i]

        
        ct_files_train = np.concatenate(folds_ct[:i] + folds_ct[i+1:], axis=0)
        pet_files_train = np.concatenate(folds_pet[:i] + folds_pet[i+1:], axis=0)
        mask_files_train = np.concatenate(folds_mask[:i] + folds_mask[i+1:], axis=0)

    # ct_files_train = ct_files[cut_index:]
    # pet_files_train = pet_files[cut_index:]
    # mask_files_train = mask_files[cut_index:]
    # ct_files_val = ct_files[:cut_index]
    # pet_files_val = pet_files[:cut_index]
    # mask_files_val = mask_files[:cut_index]

        print(
            f"# Training samples: {len(ct_files_train)}\t# Validation samples: {len(ct_files_val)}"
        )

        training = []
        validation = []

        for ct, pet, mask in zip(
            ct_files_train, pet_files_train, mask_files_train
        ):
            if not (
                ct.replace("CT", "")
                == pet.replace("PET", "")
                == mask.replace("SegP", "")
            ):
                print(f"not aligned:\tct:{ct}\tpet:{pet}\tmask:{mask}")
        
            training.append(
                {
                    "image": [f"./CT/{ct}", f"./PET/{pet}"],
                    "label": f"./Mask/{mask}",
                }
            )
        for ct, pet, mask in zip(
            ct_files_val, pet_files_val, mask_files_val
        ):
            assert (
                ct.replace("CT", "")
                == pet.replace("PET", "")
                == mask.replace("SegP", "")
            )

            validation.append(
                {
                    "image": [f"./CT/{ct}", f"./PET/{pet}"],
                    "label": f"./Mask/{mask}",
                }
            )

        to_json = {"training": training, "validation": validation}

        with open(f"fold{i}.json", "w") as f:
            json.dump(to_json, f)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(
        description="Generating JSON for an unlabeled dataset"
    )
    parser.add_argument(
        "--path",
        default="dataset/dataset0",
        type=str,
        help="path to the images",
    )
    parser.add_argument(
        "--json",
        default="jsons/dataset0.json",
        type=str,
        help="path to the json output",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="whether handling a ssl-pretraining or sl-finetune data",
    )
    parser.add_argument(
        "--ratio",
        default=0.1,
        type=float,
        help="ratio of validation data",
    )
    parser.add_argument(
        "--folds",
        default=1,
        type=int,
        help="number of folds",
    )
    args = parser.parse_args()

    if args.mode == "ssl":
        ssl_generator(args)
    if args.mode == "sl":
        supervised_generator(args)
