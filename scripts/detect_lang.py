from __future__ import annotations

import argparse
from collections import Counter

import langid
from calmqa.dataset import Dataset, QuestionType
from calmqa.language import Language
from models.model import ModelName
from polyglot.detect import Detector
from tqdm import tqdm


def should_use_polyglot_detector(language: Language) -> bool:
    return language in (
        Language.Afar,
        Language.Hiligaynon,
        Language.Tswana,
        Language.Fijian,
        Language.Samoan,
        Language.Tongan,
        Language.Kirundi,
        Language.Wolof,
        Language.Hindi,
        Language.Faroese,
        Language.Pashto,
        Language.Russian,
        Language.Samoan,
        Language.Tswana,
        Language.Wolof,
        Language.Arabic,
        Language.Balochi,
    )


def detect_texts_language(
    texts: list[str],
    language: Language,
) -> None:
    predicted_lang_counts = Counter()
    for text in tqdm(texts):
        if should_use_polyglot_detector(language):
            try:
                detector = Detector(text)
                predicted_lang = detector.language.code
            except:  # apply langid if polyglot complains
                predicted_lang, _ = langid.classify(text)
        else:
            predicted_lang, _ = langid.classify(text)

        predicted_lang_counts[predicted_lang] += 1

    for lang, count in sorted(predicted_lang_counts.items(), key=lambda item: -item[1]):
        print(lang, count)


def detect_questions_language(
    dataset: Dataset,
    *,
    q_translation_language: Language | None = None,
    max_questions: int | None = None,
) -> None:
    questions = dataset.get_questions(QuestionType.NONE)
    if max_questions is not None:
        questions = questions[:max_questions]

    language = q_translation_language or questions[0].language

    question_texts = [question.translations[language].text for question in questions]
    detect_texts_language(question_texts, language)


def detect_answers_language(
    dataset: Dataset,
    model_name: ModelName,
    *,
    max_answers: int | None = None,
) -> None:
    questions = dataset.get_questions(QuestionType.NONE)

    answers = [
        answer
        for question in questions
        for answer in dataset.get_answers(question, model_name=model_name)
    ]

    if max_answers is not None:
        answers = answers[:max_answers]

    language = answers[0].language

    answer_texts = [answer.translations[language].text for answer in answers]
    detect_texts_language(answer_texts, language)



def detect_language(
    dataset_load_path: str,
    *,
    model_name: str | None = None,
    check_questions: bool = False,
    max_entries: int | None = None,
    q_translation_language: Language | None = None,
) -> None:
    dataset = Dataset.from_file(dataset_load_path)

    if check_questions:
        detect_questions_language(
            dataset, q_translation_language=q_translation_language, max_questions=max_entries
        )
    else:
        assert model_name is not None
        detect_answers_language(dataset, ModelName[model_name], max_answers=max_entries)


def main() -> None:
    parser = argparse.ArgumentParser()

    question_or_answer_parser = parser.add_mutually_exclusive_group(required=True)

    question_or_answer_parser.add_argument(
        "--model_name",
        choices=[model_name.name for model_name in ModelName],
        help="Name of model the answers of which to detect language",
    )
    question_or_answer_parser.add_argument(
        "--check_questions",
        action="store_true",
        help="Detect language of questions",
    )

    parser.add_argument(
        "--q_translation_lang",
        type=lambda lang: Language(lang.title()),
        default=None,
        help="Only prompts using translations of questions in the given languages. If not set, untranslated questions are used.",  # noqa: E501
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="If set, limits the number of entries to the number provided. Used for testing.",
    )
    parser.add_argument(
        "--lp",
        "--dataset_load_path",
        type=str,
        dest="dataset_load_path",
        default="data/dataset.json",
        help="Path of json file containing the dataset",
    )

    args = parser.parse_args()

    detect_language(
        dataset_load_path=args.dataset_load_path,
        model_name=args.model_name,
        check_questions=args.check_questions,
        max_entries=args.max_entries,
        q_translation_language=args.q_translation_lang,
    )


if __name__ == "__main__":
    main()
