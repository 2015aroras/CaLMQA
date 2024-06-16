from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from calmqa.dataset import Answer, Dataset, Question, QuestionType
from calmqa.language import Language
from models.model import ModelName


def _get_lang_code(language: Language) -> str:
    lang_to_code = {
        Language.English: "en",
        Language.Arabic: "ar",
        Language.Chinese: "zh",
        Language.German: "de",
        Language.Hebrew: "he",
        Language.Hindi: "hi",
        Language.Hungarian: "hu",
        Language.Japanese: "ja",
        Language.Korean: "ko",
        Language.Polish: "pl",
        Language.Russian: "ru",
        Language.Spanish: "es",
        Language.Afar: "aa",
        Language.Balochi: "bal",
        Language.Faroese: "fo",
        Language.Fijian: "fj",
        Language.Hiligaynon: "hil",
        Language.Kirundi: "rn",
        Language.Papiamento: "pap",
        Language.Pashto: "ps",
        Language.Samoan: "sm",
        Language.Tongan: "to",
        Language.Tswana: "tn",
        Language.Wolof: "wo",
    }
    return lang_to_code[language]


def _get_model_code(model_name: ModelName) -> str:
    model_name_to_code = {
        ModelName.GPT_4_TURBO: "gpt4turbo",
        ModelName.MIXTRAL_8X22B_API: "mixtral22b",
        ModelName.MIXTRAL_8X22B_TOGETHER: "mixtral22b",
        ModelName.CLAUDE_OPUS: "claude",
    }
    return model_name_to_code[model_name]


def create_human_eval_ui_entry(  # noqa: PLR0913
    question: Question,
    gold_answer_str: str,
    gold_answer_description: str,
    model_answers: list[Answer],
    language: Language,
    q_type: QuestionType,
    rng: random.Random,
) -> dict:
    # Randomize answer order
    rng.shuffle(model_answers)

    entry = {
        "question": question.translations[language].get_text(),
        "type": "cult" if q_type & QuestionType.CULTURAL == QuestionType.CULTURAL else "non-cult",
        "gold_ans": gold_answer_str,
        "gold_ans_desc": gold_answer_description,
        "lang": _get_lang_code(language),
        "items": [],
    }

    for i, answer in enumerate(model_answers):
        base_key = f"ans{i+1}"
        entry[base_key] = answer.translations[language].text
        entry[f"{base_key}-decoded"] = _get_model_code(answer.prompting_state.model_name)
        entry["items"].append(
            {
                "id": base_key,
                "title": f"Answer {i+1}",
                "body": entry[base_key],
            },
        )

    return entry


def create_human_eval_ui_cultural_data(
    model_names: list[ModelName],
    languages: list[Language],
    dataset: Dataset,
    rng: random.Random,
) -> list[dict]:
    human_eval_ui_data = []
    questions = dataset.get_questions(QuestionType.CULTURAL, languages)

    for question in questions:
        if not question.human_evaluated:
            continue

        model_answers = []
        for model_name in model_names:
            answers = dataset.get_answers(question, model_name=model_name)
            answers = list(
                filter(
                    lambda ans: ans.prompting_state.other_state.get("is_multiple_choice")
                    is not True,
                    answers,
                ),
            )
            if len(answers) != 1:
                msg = f"Expected 1 answer for question {question.name}, found {len(answers)}"
                raise ValueError(msg)
            model_answers.append(answers[0])

        human_answers = dataset.get_answers(question, model_name=ModelName.HUMAN)
        if len(human_answers) > 1:
            msg = (
                f"Expected 1 human answer for question {question.name}, found {len(human_answers)}"
            )
            raise ValueError(msg)
        if len(human_answers) == 0:
            human_answers = dataset.get_answers(question, model_name=ModelName.HUMAN_DOT_POINTS)
            if len(human_answers) > 1:
                msg = (
                    f"Expected 1 human dot points answer for question {question.name}, found {len(human_answers)}"
                )
                raise ValueError(msg)
        gold_answer_str = human_answers[0].untranslated.text if len(human_answers) > 0 else ""

        human_eval_entry = create_human_eval_ui_entry(
            question,
            gold_answer_str,
            f"For this question make sure that the answer is relevant to the region where {question.language.value} is spoken",
            model_answers,
            question.language,
            QuestionType.CULTURAL,
            rng,
        )
        human_eval_ui_data.append(human_eval_entry)

    return human_eval_ui_data


def create_human_eval_ui_non_cultural_data(
    model_names: list[ModelName],
    languages: list[Language],
    dataset: Dataset,
    rng: random.Random,
) -> list[dict]:
    human_eval_ui_data = []
    questions = dataset.get_questions(QuestionType.NON_CULTURAL)

    for question in questions:
        if not question.human_evaluated:
            continue

        for language in languages:
            model_answers = []
            for model_name in model_names:
                answers = dataset.get_answers(question, language=language, model_name=model_name)
                answers = list(
                    filter(
                        lambda ans: ans.prompting_state.other_state.get("is_multiple_choice")
                        is not True,
                        answers,
                    ),
                )
                if len(answers) != 1:
                    msg = f"Expected 1 {model_name} {language} answer for question {question.name}, found {len(answers)}"
                    raise ValueError(msg)
                model_answers.append(answers[0])

            human_answers = dataset.get_answers(question, model_name=ModelName.HUMAN_DOT_POINTS)
            if len(human_answers) != 1:
                msg = f"Expected 1 human answer for question {question.name}, found {len(human_answers)}"  # noqa: E501
                raise ValueError(msg)
            gold_answer_str = human_answers[0].untranslated.text

            human_eval_entry = create_human_eval_ui_entry(
                question,
                gold_answer_str,
                "",
                model_answers,
                language,
                QuestionType.NON_CULTURAL,
                rng,
            )
            human_eval_ui_data.append(human_eval_entry)

    return human_eval_ui_data


def create_human_eval_ui_data(
    model_names: list[ModelName],
    languages: list[Language],
    cultural_datasets: list[Dataset],
    non_cultural_datasets: list[Dataset],
    seed: int,
) -> list[dict]:
    rng = random.Random()
    rng.seed(seed)

    human_eval_ui_data = []
    for non_cultural_dataset in non_cultural_datasets:
        human_eval_ui_data += create_human_eval_ui_non_cultural_data(
            model_names,
            languages,
            non_cultural_dataset,
            rng,
        )
    for cultural_dataset in cultural_datasets:
        human_eval_ui_data += create_human_eval_ui_cultural_data(
            model_names,
            languages,
            cultural_dataset,
            rng,
        )

    rng.shuffle(human_eval_ui_data)
    return human_eval_ui_data


def create_and_save_human_eval_ui_data(  # noqa: PLR0913
    model_names: list[ModelName],
    languages: list[Language],
    cultural_dataset_paths: list[str],
    non_cultural_dataset_paths: list[str],
    human_eval_save_path: str,
    seed: int,
) -> None:
    cultural_datasets = [
        Dataset.from_file(cultural_dataset_path) for cultural_dataset_path in cultural_dataset_paths
    ]
    non_cultural_datasets = [
        Dataset.from_file(non_cultural_dataset_path)
        for non_cultural_dataset_path in non_cultural_dataset_paths
    ]

    human_eval_ui_data = create_human_eval_ui_data(
        model_names,
        languages,
        cultural_datasets,
        non_cultural_datasets,
        seed,
    )

    with Path(human_eval_save_path).open("w", encoding="utf-8") as save_file:
        json.dump(human_eval_ui_data, save_file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_names",
        nargs="+",
        choices=[model_name.name for model_name in ModelName],
        help="Models to use",
    )
    parser.add_argument(
        "--languages",
        type=Language,
        nargs="+",
        default=None,
        help="Languages to use in human eval",
    )
    parser.add_argument(
        "--cdps",
        "--cultural_dataset_paths",
        type=str,
        nargs="*",
        default=[],
        dest="cultural_dataset_paths",
        help="Paths of jsons file containing cultural datasets",
    )
    parser.add_argument(
        "--ncdps",
        "--non_cultural_dataset_paths",
        type=str,
        nargs="*",
        default=[],
        dest="non_cultural_dataset_paths",
        help="Paths of jsons file containing non-cultural datasets",
    )
    parser.add_argument(
        "-o",
        "--human_eval_save_path",
        type=str,
        dest="human_eval_save_path",
        default="data/human_eval_format.json",
        help="Path of file to save human eval json to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Seed for rng.",
    )

    args = parser.parse_args()

    create_and_save_human_eval_ui_data(
        model_names=[ModelName[name] for name in args.model_names],
        languages=args.languages,
        cultural_dataset_paths=args.cultural_dataset_paths,
        non_cultural_dataset_paths=args.non_cultural_dataset_paths,
        human_eval_save_path=args.human_eval_save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
