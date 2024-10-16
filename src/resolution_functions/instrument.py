from __future__ import annotations

from collections import ChainMap
import dataclasses
import os
import yaml
from typing import Any, Optional, Union

from .models import MODELS


INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')

INSTRUMENT_MAP: dict[str, tuple[str, Union[None, str]]] = {
    'Lagrange': ('lagrange', None),
    'MAPS': ('maps', None),
    'PANTHER': ('panther', None),
    'TFXA': ('tosca', 'TFXA'),
    'TOSCA': ('tosca', None),
    'VISION': ('vision', None),
}


class InvalidInstrumentError(Exception):
    pass

class InvalidSettingError(Exception):
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    name: str
    version: str
    models: dict[str, dict[str, Union[str, Union[dict[str, Union[float, int, str, list[float], dict]],
                                                 dict[str, dict[str, Union[float, int, str, list[float]]]]]]]]
    default_model: str

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        version_data = data['version'][version]

        return cls(
            data['name'],
            version,
            version_data['models'],
            version_data['default_model'],
        )

    @classmethod
    def from_default(cls, name: str, version: Optional[str] = None):
        try:
            file_name, implied_version = INSTRUMENT_MAP[name]
        except KeyError:
            raise InvalidInstrumentError(f'"{name}" is not a valid instrument name. Only the following instruments are '
                                         f'supported: {list(INSTRUMENT_MAP.keys())}')

        if version is None:
            version = implied_version

        return cls.from_file(os.path.join(INSTRUMENT_DATA_PATH, file_name + '.yaml'), version)

    def get_model_parameter(self, model_name: str, parameter_name: str, setting: str) -> Union[Any, None]:
        return self.models[model_name].get_value(parameter_name, setting)

    def get_resolution_function(self,
                                model_name: Optional[str] = None,
                                **kwargs):
        if model_name is None:
            model_name = self.default_model

        model = self.models[model_name]
        available_settings = model['settings']

        settings = []
        for setting_name, options in available_settings.items():
            kwarg = kwargs.pop(setting_name, None)
            if kwarg is None:
                kwarg = options['default_setting']

            settings.append(options[kwarg])

        model_class = MODELS[model['function']]
        return model_class(model_class.data_class(function=model['function'],
                                                  citation=model['citation'],
                                                  **ChainMap(*settings, model['parameters'])),
                           **kwargs)

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())

    @property
    def available_models_and_settings(self) -> dict[str, list[str]]:
        return {model_name: list(model['settings'].keys()) for model_name, model in self.models.items()}

    @property
    def all_available_models_options(self) -> dict[str, dict[str, list[str]]]:
        return {model_name: {setting: self._get_options(value) for setting, value in list(model['settings'].items())}
                for model_name, model in self.models.items()}

    def possible_settings_for_model(self, model: str) -> list[str]:
        return list(self.models[model]['settings'].keys())

    def possible_options_for_model(self, model: str) -> dict[str, list[str]]:
        return {setting: self._get_options(value) for setting, value in self.models[model]['settings'].items()}

    def possible_options_for_model_and_setting(self, model: str, setting: str) -> list[str]:
        return self._get_options(self.models[model]['settings'][setting])

    @staticmethod
    def _get_options(setting: dict[str, Union[str, dict]]) -> list[str]:
        return [value for value in setting.keys() if value != 'default_setting']

    def default_option_for_setting(self, model: str, setting: str) -> str:
        return self.models[model]['settings'][setting]['default_setting']
