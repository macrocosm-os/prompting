import tqdm
import bittensor as bt
import argostranslate.package
import argostranslate.translate
from typing import List
from prompting.tasks import Task
from dataclasses import dataclass
from argostranslate.package import AvailablePackage


def load_translation_packages() -> List[AvailablePackage]:
    # Update package index and get available and installed packages
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    installed_packages = argostranslate.package.get_installed_packages()

    # Helper function to check if a package is installed
    def is_package_installed(from_code, to_code, packages):
        return any(pkg for pkg in packages if pkg.from_code == from_code and pkg.to_code == to_code)

    # Filter available packages for supported language pairs
    supported_language_pairs = [
        pkg for pkg in available_packages
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
            bt.logging.success(f'Package successfully installed at {package_path}')
        else:
            bt.logging.info(f"Package from {package.from_code} to {package.to_code} is already installed, skipping...")

    return supported_language_pairs


SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "pt", "uk"
]


@dataclass
class TranslationTask(Task):
    name = "translation"
    desc = "get translation help"
    goal = "to get the translation for the given piece of text"

    

    # TODO: TEST BLEU SCORE
    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),        
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]
    
    
    def __init__(self, context: str):
        # Load necessary packages if not already installed
        self.available_packages = load_translation_packages()
        
        from_code = SUPPORTED_LANGUAGES[0].from_code
        to_code = SUPPORTED_LANGUAGES[1].to_code
        
        self.reference = argostranslate.translate.translate("Test", from_code, to_code)