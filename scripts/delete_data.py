from __future__ import annotations

import argparse

from calmqa.dataset import Dataset
from models.model import ModelName


def delete_data(
    dataset_load_path: str,
    dataset_save_path: str,
    *,
    model_names: list[ModelName],
    delete_all_mc: bool = False,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    for entry in dataset.entries:
        filtered_answers = []
        for answer in entry.answers:
            if answer.prompting_state.model_name in model_names:
                continue

            if (
                delete_all_mc
                and answer.prompting_state.other_state.get("is_multiple_choice") is True
            ):
                continue

            filtered_answers.append(answer)

        entry.answers = filtered_answers

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        default=[],
        nargs="+",
        type=lambda name: ModelName[name],
        help="If set, delete all answer data of these model",
    )
    parser.add_argument(
        "--all_mc",
        action="store_true",
        help="Delete all mc data",
    )

    parser.add_argument(
        "dataset_load_path",
        type=str,
        default="data/dataset.json",
        help="Path of json file containing the dataset",
    )
    parser.add_argument(
        "--sp",
        "--dataset_save_path",
        type=str,
        dest="dataset_save_path",
        default=None,
        help="Path of file to save the dataset to. Defaults to the load path",
    )

    args = parser.parse_args()

    delete_data(
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        model_names=args.model_names,
        delete_all_mc=args.all_mc,
    )


if __name__ == "__main__":
    main()
