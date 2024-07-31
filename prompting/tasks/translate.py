import tqdm
import bittensor as bt
import argostranslate.package
import argostranslate.translate
import random
from typing import List, Tuple
from prompting.tasks import Task
from dataclasses import dataclass
from argostranslate.package import AvailablePackage
from prompting.shared import Context

SUPPORTED_LANGUAGES = ["en", "es", "fr", "pt", "uk"]


class TranslationPipeline:
    def __init__(self):
        self.supported_language_pairs = self.load_translation_packages()

    def load_translation_packages(self) -> List[AvailablePackage]:
        # Update package index and get available and installed packages
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        installed_packages = argostranslate.package.get_installed_packages()

        # Helper function to check if a package is installed
        def is_package_installed(from_code, to_code, packages):
            return any(pkg for pkg in packages if pkg.from_code == from_code and pkg.to_code == to_code)

        # Filter available packages for supported language pairs
        supported_language_pairs = [
            pkg
            for pkg in available_packages
            if pkg.from_code in SUPPORTED_LANGUAGES and pkg.to_code in SUPPORTED_LANGUAGES
        ]

        bt.logging.info(f"Supported language pairs: {supported_language_pairs}")

        # Check for installed packages
        pbar = tqdm.tqdm(supported_language_pairs, desc="Checking installed packages")
        for package in pbar:
            if not is_package_installed(package.from_code, package.to_code, installed_packages):
                bt.logging.info(f"Installing package from {package.from_code} to {package.to_code}")
                package_path = str(package.download())
                argostranslate.package.install_from_path(package_path)
                bt.logging.success(f"Package successfully installed at {package_path}")
            else:
                bt.logging.info(
                    f"Package from {package.from_code} to {package.to_code} is already installed, skipping..."
                )

        return supported_language_pairs

    def random_translation(self, content: str) -> str:
        # TODO: NOT RANDOM
        from_code = self.SUPPORTED_LANGUAGES[0]
        to_code = self.SUPPORTED_LANGUAGES[1]
        return argostranslate.translate.translate(content, from_code, to_code)

    def translate(self, content: str, from_code: str, to_code: str):
        self.reference = argostranslate.translate.translate(content, from_code, to_code)

    def translate_to_random_language(self, content: str, from_code: str = "en") -> Tuple[AvailablePackage, str]:
        english_supported_languages = list(filter(lambda x: x.from_code == from_code, self.supported_language_pairs))
        available_translations = list(map(lambda x: x, english_supported_languages))

        random_translation_obj = random.choice(available_translations)
        translation_code = random_translation_obj.to_code

        translated_content = argostranslate.translate.translate(content, from_code, to_code=translation_code)

        return random_translation_obj, translated_content


@dataclass
class TranslationTask(Task):
    challenge_type = "query"
    static_reference = True
    static_query = True
    name = "translation"
    desc = "get translation help"
    goal = "to get the translation for the given piece of text"

    templates = [
        "Could you assist me with translating the following text into {another_language}? \n{text}",
        "I need some help translating this text into {another_language}. Can you do it? \n{text}",
        "Is it possible for you to translate this text for me into {another_language}? Here it is: \n{text}",
        "Would you mind helping me convert this text into {another_language}? \n{text}",
        "Could you please convert this into {another_language} for me? \n{text}",
        "I was wondering if you could help translate this into {another_language}? \n{text}",
        "Can you provide a translation for this text into {another_language}? \n{text}",
        "Hey, can you turn this text into {another_language} for me? \n{text}",
        "Could I get some assistance in translating this into {another_language}? \n{text}",
        "Are you able to help me render this text in {another_language}? \n{text}",
        "I'd appreciate your help translating this text into {another_language}. Here's the text: \n{text}",
        "Please could you translate the following text into {another_language}? \n{text}",
        "Might you help me by translating this text to {another_language}? \n{text}",
        "I'm looking for help to translate this text into {another_language}. Any chance you can assist? \n{text}",
        "How about translating this text into {another_language} for me? \n{text}",
        "Would it be possible for you to help translate this text into {another_language}? \n{text}",
        "I need your expertise to translate this text into {another_language}, can you help? \n{text}",
        "Can you work your magic and translate this text into {another_language}? \n{text}",
        "I require assistance translating the following into {another_language}. Can you help? \n{text}",
        "Hey, could you take a moment to translate this text into {another_language} for me? \n{text}",
    ]

    # TODO: TEST BLEU SCORE
    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=1),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=1),
    ]

    cleaning_pipeline = []

    def __init__(self, translation_pipeline: TranslationPipeline, context: Context):
        # Set task internal variables
        self.context = context
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

        # Translates english text to a random language
        content_translation_obj, translated_content = translation_pipeline.translate_to_random_language(context.content)

        # Translates the translation to another random language
        reference_translation_obj, reference_translation_content = translation_pipeline.translate_to_random_language(
            content=translated_content, from_code=content_translation_obj.to_code
        )
        self.reference = reference_translation_content

        # Composes the query
        # TODO: Implement template translation
        template = random.choice(self.templates)
        self.query = template.format(another_language=reference_translation_obj.to_name, text=translated_content)
