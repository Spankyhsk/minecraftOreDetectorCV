# -*- coding: utf-8 -*-
"""
Template-Ladezugriff fuer Erzvarianten.
"""

import os
from typing import Dict

import numpy as np

from detection import load_template
from utils import log


class TemplateRepository:
    """
    Laedt und cached die Template-Bank pro Erzfamilie.
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def get_for_ore(self, ore_key: str) -> Dict[str, np.ndarray]:
        if ore_key not in self._cache:
            self._cache[ore_key] = self._load_for_ore(ore_key)
        return self._cache[ore_key]

    def _load_for_ore(self, ore_key: str) -> Dict[str, np.ndarray]:
        bank: Dict[str, np.ndarray] = {}

        if not os.path.isdir(self.templates_dir):
            log(f"Template-Ordner nicht gefunden: {self.templates_dir}")
            return bank

        for name in os.listdir(self.templates_dir):
            if not name.lower().endswith(".png"):
                continue
            if not name.startswith(f"{ore_key}_"):
                continue

            path = os.path.join(self.templates_dir, name)
            variant = name[:-4]

            try:
                bank[variant] = load_template(path)
            except Exception as exc:
                log(f"Template konnte nicht geladen werden: {path} ({exc})")

        return bank

