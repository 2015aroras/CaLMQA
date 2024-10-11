from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

from calmqa.dataset import Answer, Dataset, Question, QuestionType
from models.model import Model, ModelName
from tqdm import tqdm


def _extract_rating(response: str, criteria: str, allowed_ratings: list[int]) -> int:
    rating = None

    match = re.search(r"<rating>([0-9]+?)</rating>", response)
    if match is None:
        raise ValueError(f"Rating not found in response: {response}")

    rating = int(match.group(1))

    # # Just a number is returned
    # match = re.match(r"([0-9]+?)", response)
    # if match is not None:
    #     rating = int(match.string)
    # else:
    #     match = re.search(rf"{criteria}: ([0-9]+?)", response)
    #     if match is not None:
    #         rating = int(match.group(1))
    #     else:
    #         raise ValueError(f"Score not found in response: {response}")

    if rating not in allowed_ratings:
        raise ValueError(f"Invalid rating {rating}")

    return rating


def _get_criteria_key(criteria: str, rater_model_name: ModelName, answer: Answer) -> str:
    return f"{criteria}|{rater_model_name.name}"


def rate_answer(
    answer: Answer,
    rater_model: Model,
    question: Question,
    prompt_template: str,
    criteria_name: str,
    criteria_desc: str,
    dataset: Dataset,
    *,
    overwrite: bool = False,
    save_progress: bool = True,
) -> bool:
    ratings = answer.ratings or {}
    criteria_key = _get_criteria_key(criteria_name, rater_model.name, answer)
    if not overwrite and ratings.get(criteria_key, -1) != -1:
        return False

    cleaned_question = question.untranslated.get_text()
    cleaned_question = re.sub("\n+", "\n", cleaned_question)

    prompt = prompt_template.replace("[source]", question.source)
    prompt = prompt_template.replace("[question]", cleaned_question)
    prompt = prompt.replace("[answer]", answer.untranslated.text)

    if "[answer_human]" in prompt_template:
        human_answers = dataset.get_answers(question, model_name=ModelName.HUMAN)
        assert len(human_answers) == 1

        prompt = prompt.replace("[answer_human]", human_answers[0].untranslated.text)

    prompt = prompt.replace("[source_language]", question.language.value)
    prompt = prompt.replace("[criteria_name]", criteria_name)
    prompt = prompt.replace("[criteria_desc]", criteria_desc)

    response, _ = rater_model.prompt(prompt)

    ratings[criteria_key] = _extract_rating(response, criteria_name, list(range(1, 6)))

    answer_dict = dataclasses.asdict(answer)
    answer_dict["ratings"] = ratings

    updated_answer = Answer(**answer_dict)

    dataset.add_or_update_answer(question, updated_answer)

    if save_progress:
        dataset.to_file()

    return True


def rate_answers(  # noqa: PLR0913
    rater_model_name: ModelName,
    ratee_model_name: ModelName,
    prompt_file_path: str,
    criteria_name: str,
    criteria_desc_file_path: str,
    dataset_load_path: str,
    dataset_save_path: str,
    max_output_tokens: int,
    *,
    only_human_evaluated: bool = False,
    max_questions: int | None = None,
    overwrite: bool = False,
    save_progress: bool = True,
    **kwargs,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    rater_model = Model.make(rater_model_name, max_output_tokens, **kwargs)
    criteria_desc = Path(criteria_desc_file_path).read_text()
    prompt_template = Path(prompt_file_path).read_text()

    questions = dataset.get_questions(QuestionType.NONE)
    if only_human_evaluated:
        questions = [question for question in questions if question.human_evaluated]

    if max_questions is not None:
        questions = questions[:max_questions]

    for question in tqdm(
        questions,
        desc=f"Rating {len(questions)} questions using {rater_model.name.value}",
    ):
        answers = dataset.get_answers(question, model_name=ratee_model_name)
        assert len(answers) <= 1
        if answers == 0:
            continue

        rate_answer(
            answers[0],
            rater_model,
            question,
            prompt_template,
            criteria_name,
            criteria_desc,
            dataset,
            overwrite=overwrite,
            save_progress=save_progress,
        )

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rater_model_name",
        type=lambda name: ModelName[name.upper()],
        choices=list(ModelName),
        help="Name of model to use to rate answers",
    )
    parser.add_argument(
        "ratee_model_name",
        type=lambda name: ModelName[name.upper()],
        choices=list(ModelName),
        help="Name of model whose answers are being rated",
    )
    parser.add_argument(
        "criteria_name",
        help="Name of criteria being rated",
    )
    parser.add_argument(
        "--human_evaluated",
        action="store_true",
        help="If set, only rate answers that were human evaluated.",
    )

    parser.add_argument(
        "-d",
        "--criteria_desc_file",
        type=str,
        required=True,
        help="Path of file containing the description of the rating criteria",
    )
    parser.add_argument(
        "-p",
        "--prompt_file",
        type=str,
        default="data/prompts/rating-prompt.txt",
        help="Path of file containing the prompt for rating",
    )

    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="If set, limits the number of questions to the number provided. Used for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrites existing translations with newly-generated translation.",
    )
    parser.add_argument(
        "--lp",
        "--dataset_load_path",
        type=str,
        dest="dataset_load_path",
        help="Path of json file containing the dataset",
    )
    parser.add_argument(
        "--sp",
        "--dataset_save_path",
        type=str,
        dest="dataset_save_path",
        help="Path to save dataset. Defaults to load path",
    )

    # Local models args
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[],
        help="Ids of gpus available for use.",
    )
    parser.add_argument(
        "--max_gpu_mem",
        type=float,
        default=None,
        help="Max memory to use on each GPU in bytes.",
    )

    # Model parameter args
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens in output.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Max tokens in output.",
    )

    args = parser.parse_args()

    rate_answers(
        rater_model_name=args.rater_model_name,
        ratee_model_name=args.ratee_model_name,
        prompt_file_path=args.prompt_file,
        criteria_name=args.criteria_name,
        criteria_desc_file_path=args.criteria_desc_file,
        only_human_evaluated=args.human_evaluated,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path or args.dataset_load_path,
        max_output_tokens=args.max_tokens,
        max_questions=args.max_questions,
        overwrite=args.overwrite,
        gpus=args.gpus,
        max_gpu_mem=int(args.max_gpu_mem) if args.max_gpu_mem is not None else None,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
