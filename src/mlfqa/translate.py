from __future__ import annotations

import argparse
import re
from pathlib import Path

from models.model import Model, ModelName, PromptingState
from tqdm import tqdm

from mlfqa.dataset import Dataset, Question, QuestionTranslation, QuestionType
from mlfqa.language import Language


def _translate_text(
    text: str,
    source_lang: Language,
    target_lang: Language,
    prompt_template: str,
    model: Model,
    *,
    context: str | None = None,
) -> tuple[str, PromptingState]:
    prompt = prompt_template

    text = re.sub("\n+", "\n", text)
    prompt = prompt.replace("[text]", text)

    if "[context]" in prompt_template:
        assert "[context]" in prompt
        if context is None:
            msg = "Context expected in prompt but not provided"
            raise ValueError(msg)

        context = re.sub("\n+", "\n", context)
        prompt = prompt.replace("[context]", context)

    prompt = prompt.replace("[source_language]", source_lang.value)
    prompt = prompt.replace("[target_language]", target_lang.value)

    return model.prompt(prompt)


def _translate_and_store(  # noqa: PLR0913
    model: Model,
    questions: list[Question],
    target_langs: list[Language],
    prompt_template: str,
    dataset: Dataset,
    *,
    translate_questions: bool = False,
    translate_answers: bool = False,
    overwrite_existing: bool = False,
    save_progress: bool = True,
) -> None:
    if translate_answers:
        msg = "Answer translation not yet implemented"
        raise NotImplementedError(msg)

    for question in tqdm(questions, desc=f"Translating using {model.name}"):
        q_language = question.language

        for target_lang in target_langs:
            if target_lang == q_language:
                continue

            if translate_questions and (
                target_lang not in question.translations or overwrite_existing
            ):
                original_translation = question.translations[q_language]

                human_answers = dataset.get_answers(
                    question,
                    q_language,
                    model_name=ModelName.HUMAN,
                )
                if len(human_answers) > 1:
                    msg = f"{len(human_answers)} {q_language} human answers found for question"
                    raise RuntimeError(msg)
                context = human_answers[0].untranslated.text if len(human_answers) > 0 else None

                translated_question, prompting_state = _translate_text(
                    original_translation.get_text(),
                    q_language,
                    target_lang,
                    prompt_template,
                    model,
                    context=context,
                )

                question_translation = QuestionTranslation(
                    target_lang,
                    translated_question,
                    prompting_state,
                )
                dataset.add_or_update_question_translation(question, question_translation)

        if save_progress:
            dataset.to_file()


def translate(  # noqa: PLR0913
    model_name: ModelName,
    prompt_file_path: str,
    source_langs: list[Language] | None,
    target_langs: list[Language],
    question_type: QuestionType,
    dataset_load_path: str,
    dataset_save_path: str,
    *,
    max_questions: int | None = None,
    translate_questions: bool = False,
    translate_answers: bool = False,
    overwrite_existing: bool = False,
    save_progress: bool = True,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)
    dataset.default_save_path = dataset_save_path

    model = Model.make(model_name)
    prompt_template = Path(prompt_file_path).read_text()

    questions = dataset.get_questions(question_type, source_langs)
    if max_questions is not None:
        questions = questions[:max_questions]

    _translate_and_store(
        model,
        questions,
        target_langs,
        prompt_template,
        dataset,
        translate_questions=translate_questions,
        translate_answers=translate_answers,
        overwrite_existing=overwrite_existing,
        save_progress=save_progress,
    )

    dataset.to_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type",
        choices=["questions", "answers"],
        help="What to translate",
    )
    parser.add_argument(
        "model_name",
        choices=[model_name.name for model_name in ModelName],
        help="Name of model to use",
    )
    parser.add_argument(
        "-p",
        "--prompt_file",
        type=str,
        default="data/prompts/translation-prompt.txt",
        help="Path of file containing the translation prompt",
    )
    parser.add_argument(
        "--target_langs",
        type=Language,
        nargs="+",
        default=[Language.English],
        help="Languages to translate questions/answers to",
    )
    parser.add_argument(
        "--source_langs",
        type=Language,
        nargs="*",
        default=None,
        help="If set, only do translations with questions originally of the given languages",
    )
    parser.add_argument(
        "--question_types_list",
        type=lambda inp: QuestionType[inp],
        nargs="+",
        default=[QuestionType.ALL],
        help="Filters the type of questions for which translation occurs",
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
        default=None,
        help="If set, overwrites existing translations with newly-generated translation.",
    )
    parser.add_argument(
        "--lp",
        "--dataset_load_path",
        type=str,
        dest="dataset_load_path",
        default="data/dataset.json",
        help="Path of json file containing the dataset",
    )
    parser.add_argument(
        "--sp",
        "--dataset_save_path",
        type=str,
        dest="dataset_save_path",
        default="data/dataset.json",
        help="Path of file to save the dataset to",
    )

    args = parser.parse_args()

    question_type = QuestionType.NONE
    for q_type in args.question_types_list:
        question_type |= q_type

    translate(
        model_name=ModelName[args.model_name],
        prompt_file_path=args.prompt_file,
        source_langs=args.source_langs,
        target_langs=args.target_langs,
        question_type=question_type,
        dataset_load_path=args.dataset_load_path,
        dataset_save_path=args.dataset_save_path,
        max_questions=args.max_questions,
        translate_questions=args.type == "questions",
        translate_answers=args.type == "answers",
        overwrite_existing=args.overwrite,
    )


if __name__ == "__main__":
    main()
