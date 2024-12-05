"""
The main entry-point to the library - contains the `Instrument` class used for managing all data and
computing :term:`resolution functions<resolution function>`.
"""
from __future__ import annotations

from collections import ChainMap
import dataclasses
import os
import yaml
from typing import Optional, Union, TYPE_CHECKING

from .models import MODELS

if TYPE_CHECKING:
    from .models.model_base import ModelData, InstrumentModel
    from inspect import Signature

INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')

INSTRUMENT_MAP: dict[str, tuple[str, Union[None, str]]] = {
    'ARCS': ('arcs', None),
    'CNCS': ('cncs', None),
    'HYSPEC': ('hyspec', None),
    'Lagrange': ('lagrange', None),
    'LET': ('let', None),
    'MAPS': ('maps', None),
    'MARI': ('mari', None),
    'MERLIN': ('merlin', None),
    'PANTHER': ('panther', None),
    'TFXA': ('tosca', 'TFXA'),
    'TOSCA': ('tosca', None),
    'VISION': ('vision', None),
    'SEQUOIA': ('sequoia', None),
}


class InvalidInstrumentError(Exception):
    """An Exception representing an invalid user input for the :term:`instrument` name."""
    pass


class InvalidModelError(Exception):
    """
    An Exception representing an invalid user input for the :term:`model` of an :term:`instrument`.

    This class does not support an arbitrary message; instead the message is constructed in here
    from the provided information.

    Parameters
    ----------
    provided_name
        The invalid name for the model that the user provided.
    instrument
        The instance of the `Instrument` object in which the `provided_name` was used.
    """
    def __init__(self, provided_name: str, instrument: Instrument):
        message = f'"{provided_name}" is not a valid model for the {instrument.name} instrument ' \
                  f'version {instrument.version}. This instrument only supports the following ' \
                  f'models: {instrument.available_models}.'

        super().__init__(message)


class InvalidSettingError(Exception):
    """
    An Exception representing an invalid user input for the :term:`setting` of a :term:`model` of 
    an :term:`instrument`.

    This class does not support an arbitrary message; instead the message is constructed in here
    from the provided information.

    Parameters
    ----------
    provided_name
        The invalid name for the setting that the user provided.
    model_name
        The name of the model for which the `provided_name` was provided.
    instrument
        The instance of the `Instrument` object in which the `provided_name` was used.
    """
    def __init__(self, provided_name: str, model_name: str, instrument: Instrument):
        message = f'"{provided_name}" is not a valid setting for the {model_name} model of the ' \
                  f'{instrument.version} version of the {instrument.name} instrument. This ' \
                  f'instrument only supports the following models: {instrument.available_models}.'

        super().__init__(message)


class InvalidVersionError(Exception):
    """An Exception representing an invalid user input for the :term:`version`                                                                         of an instrument."""
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    """
    Instrument is a representation of a physical :term:`INS` :term:`instrument`, containing all its
    associated data.

    To be precise, it holds all information about one :term:`version` of an :term:`instrument` (for
    more about :term:`instrument` versions, see ...), which makes it the centrepiece of this
    library; the data is necessary for computing the
    :term:`resolution functions<resolution function>`.

    However, this information is static and curated by the library, which is why `Instrument` is a
    frozen data class. It should never be instantiated directly; instead the `from_default`
    constructor should be used. Similarly, it should not be inspected directly; a variety of methods
    and properties are provided for querying relevant information.

    Regardless, the most important function of Instrument is to construct a resolution function,
    which can be done by using the `get_resolution_function` method.

    Parameters
    ----------
    name
        The name of the INS instrument.
    version
        The name of a particular version of that INS instrument.
    _models
        A dictionary detailing all the models and their data available for this version of this
        instrument.
    default_model
        The default model for this version of this instrument.

    Attributes
    ----------
    name
        The name of the INS instrument represented by this instance.
    version
        The version of the INS instrument represented by this instance.
    default_model
        The name of the model for this version of this INS instrument that is used by default.
    instrument_versions
    available_models
    available_models_and_settings
    all_available_models_options
    """
    name: str
    version: str
    _models: dict[str, dict[str, Union[str, Union[dict[str, Union[float, int, str, list[float], dict]],
    dict[str, dict[str, Union[float, int, str, list[float]]]]]]]]
    default_model: str

    def __str__(self):
        return f'Instrument(name={self.name}, version={self.version})'

    @classmethod
    def available_instruments(cls) -> list[str]:
        """
        Lists all :term:`INS` :term:`instruments<instrument>` currently available.

        Returns
        -------
        instrument_list
            A list of names of INS instruments supported by this library.
        """
        return list(INSTRUMENT_MAP.keys())

    @classmethod
    def _available_versions(cls, path: str) -> tuple[list[str], str]:
        """
        Lists the names of all versions of the INS instrument contained in the file found at `path`.

        Parameters
        ----------
        path
            The path to the file that will be inspected for versions.

        Returns
        -------
        available_versions
            A list of the version names found in the provided file.
        default_version
            The default version of this instrument, as specified in the file.

        Warnings
        --------
        This method performs an I/O (read) operation.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return list(data['version'].keys()), data['default_version']

    @classmethod
    def available_versions(cls, instrument_name: str) -> tuple[list[str], str]:
        """
        Lists the names of all :term:`versions<version>` available for an :term:`INS`
        :term:`instrument`, as well as the default :term:`version`.

        Parameters
        ----------
        instrument_name
            The name of the INS instrument whose versions to retrieve.

        Returns
        -------
        available_versions
            A list of the version names available for the instrument.
        default_version
            The version of the instrument that is used by default.

        Warnings
        --------
        This method performs an I/O (read) operation.

        See Also
        --------
        available_instruments : Lists the available instruments
        """
        path, implied_version = cls._get_file(instrument_name)

        versions, default_version = list(cls._available_versions(path))

        if implied_version is None:
            return versions, default_version
        else:
            return versions, implied_version

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None) -> Instrument:
        """
        Instantiates an `Instrument` from the data loaded from the file found at `path`.

        Please note that while this method is a part of the public API, it is marked as being for
        advanced use only. For most use cases, use the `from_default` method.

        This method assumes that the data in the file follows the standard (please see ... for more
        details). No validation is performed, so if there are any issues, either unhandled
        exceptions will be raised, or the errors will be silently propagated. For adding new
        instruments, versions, or models, please open an issue on our GitHub. For other purposes,
        please see ... .

        While the data file may contain multiple versions, only the one specified by `version` will
        be saved in memory. If the `version` parameter is not provided, the default version is read
        from the file.

        Parameters
        ----------
        path
            The path to the file containing the instrument data.
        version
            The version of the instrument to load. If not provided, ``the default_version``
            specified in the file will be used.

        Returns
        -------
        instrument
            An instance of the `Instrument` class containing the data found at `path`

        Raises
        ------
        InvalidVersionError
            If the file does not contain the provided `version`.
        KeyError
            If certain parts of the file are egregiously incorrect.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        version_data = data['version']
        try:
            version_data = version_data[version]
        except KeyError:
            versions = list(version_data.keys())
            raise InvalidVersionError(f'"{version}" is not a valid version name. Only the following'
                                      f' versions are supported for this instrument: {versions}')

        return cls(
            data['name'],
            version,
            version_data['models'],
            version_data['default_model'],
        )

    @classmethod
    def from_default(cls, name: str, version: Optional[str] = None) -> Instrument:
        """
        Instantiates an `Instrument` class with the data of the `name` :term:`INS`
        :term:`instrument` and its `version`.

        This is the primary, recommended way of instantiating the `Instrument` class. It loads the
        instrument data as curated in this library for a particular :term:`version` of an
        :term:`INS` :term:`instrument`.

        Parameters
        ----------
        name
            The name of the INS instrument to instantiate.
        version
            The version of the `name` INS instrument to instantiate. If not provided, the default
            version of that instrument is instantiated.

        Returns
        -------
        instrument
            An instance of the `Instrument` class containing the data corresponding to `name` and
            `version`.

        Raises
        ------
        InvalidInstrumentError
            If the specified instrument `name` does not exist.
        InvalidVersionError
            If the specified `version` is not available for the specified instrument `name`.

        See Also
        --------
        available_instruments : Lists the available instruments
        available_versions : Lists the available versions of an instrument and its default version.
        """
        path, implied_version = cls._get_file(name)

        if version is None:
            version = implied_version

        return cls.from_file(path, version)

    @staticmethod
    def _get_file(instrument_name: str) -> tuple[str, Union[str, None]]:
        """
        Private method for obtaining the path to the default file corresponding to `instrument_name`.

        This method is mostly a wrapper around the ``INSTRUMENT_MAP`` dictionary. Therefore, it also
        provides the functionality that `instrument_name` does not have to be a unique "instrument",
        but can instead be an alias for a combination of an ``instrument`` and ``version``. For
        example, TFXA is in this library considered a version of the TOSCA instrument, but it can
        be passed in to this method as ``instrument_name='TFXA'`, which will be correctly
        interpreted as ``name='TOSCA'`` and ``version='TFXA'``. This is the purpose of the
        ``implied_version`` output parameter.

        Parameters
        ----------
        instrument_name
            The name of the instrument whose file to retrieve.

        Returns
        -------
        path
            The path to the file corresponding to `instrument_name`
        implied_version
            The version implied by `instrument_name`

        Raises
        ------
        InvalidInstrumentError
            If the specified instrument `name` does not exist.
        """
        try:
            file_name, implied_version = INSTRUMENT_MAP[instrument_name]
        except KeyError:
            raise InvalidInstrumentError(
                f'"{instrument_name}" is not a valid instrument name. Only the following instruments are '
                f'supported: {list(INSTRUMENT_MAP.keys())}')

        return os.path.join(INSTRUMENT_DATA_PATH, file_name + '.yaml'), implied_version

    def get_model_data(self, model_name: Optional[str] = None, **kwargs) -> ModelData:
        """
        Retrieves the physical parameters associated with the specified `model_name`.

        This method can be used for inspecting the parameters of a particular :term:`model`, but
        cannot be used to modify them. It returns a subclass of the `ModelData` class corresponding to the
        particular model. Another use for this method is to inspect the default values for the
        model's parameters, as well as any restrictions that they might have, via the
        `ModelData.restrictions` and `ModelData.defaults` attributes.

        Parameters
        ----------
        model_name
            The name of the model whose parameters to retrieve. If not provided, the parameters of
            the default_model will be retrieved.
        **kwargs
            Keyword arguments can be passed in to choose an option for each of settings specific to
            the `model_name`. If not provided, default values are used.

        Returns
        -------
        model_data
            The data associated with `model_name`.

        See Also
        --------
        default_model : The default model for this instrument.
        available_models : List of models available for this instrument
        possible_settings_for_model : List of settings that can be chosen for this model.
        """
        out, _ = self._get_model_data(model_name, **kwargs)
        return out

    def _get_model_data(self, model_name: Optional[str] = None, **kwargs) -> tuple[ModelData, str]:
        """
        The specific implementation for `get_model_data`.

        Parameters
        ----------
        model_name
            The name of the model whose parameters to retrieve. If not provided, the parameters of
            the default_model will be retrieved.
        kwargs
            Keyword arguments can be passed in to choose an option for each of settings specific to
            the `model_name`. If not provided, default values are used.

        Returns
        -------
        model_data
            The data associated with `model_name`.
        model_name
            The name of the returned model. This will be the same as `model_name` if it was
            provided, otherwise it is the default model name.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.
        """
        if model_name is None:
            model_name = self.default_model

        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        available_settings = model['settings']

        settings = []
        for setting_name, options in available_settings.items():
            kwarg = kwargs.pop(setting_name, None)
            if kwarg is None:
                kwarg = options['default_setting']

            settings.append(options[kwarg])

        model_class = MODELS[model['function']]
        return model_class.data_class(function=model['function'],
                                      citation=model['citation'],
                                      **ChainMap(*settings, model['parameters'])), model_name

    def get_resolution_function(self, model_name: Optional[str] = None, **kwargs) -> InstrumentModel:
        """
        Generates a :term:`resolution function`, as modelled by the `model_name` :term:`model`, for
        a set of parameters.

        This method is the main use case of the `Instrument` class. It generates a callable object
        that, when called, returns the :term:`resolution` of the :term:`instrument` at an energy
        value(s).

        However, while a simple, common interface is provided, different :term:`models<model>` (and
        sometimes the same :term:`model` for different :term:`instruments!<instrument>`) require
        different :term:`settings<setting>` to be chosen and different parameters to be provided.
        All of these have to be passed in as keyword arguments (though sensible defaults are
        provided). These keyword arguments correspond to physical user choices made when running an
        INS experiment on the corresponding :term:`instrument`. For example, direct instruments
        have a tunable incident energy, so their models usually require an ``e_init`` parameter.
        For more information about a model, please see its corresponding documentation,
        or for programmatic querying, please see "How to programmatically query model".

        Parameters
        ----------
        model_name
            The name of the model to instantiate. If not provided, the `default_model` is used.
        **kwargs
            Keyword arguments specifying the various settings and parameters of the `model_name` model

        Returns
        -------
        model
            An instance of the requested `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.
        InvalidInputError
            If the model has restrictions on its inputs and these have been violated. The
            restrictions can be checked by using the `get_model_data` method and then viewing the
            ``restrictions`` attribute of the returned object.
        Exception
            Other model-specific exceptions may be raised.

        Warnings
        --------
        If there are any mistakes in the model-specific parameters passed in as
        keyword arguments, they will be silently ignored and the default values for the missing
        parameters will be used.

        See Also
        --------
        available_models : List of models available for this version of this instrument
        get_model_signature : Constructs a call signature for calling this method for a particular model.

        Examples
        --------
        >>> from resolution_functions import Instrument
        >>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
        >>> print(tosca.get_resolution_function())
        PolynomialModel1D(citation="")

        If a model is not provided, the default model will be used - this differs between versions
        and instruments.

        >>> print(tosca.get_resolution_function('vision'))
        VisionPaperModel(citation="https://doi.org/10.1016/j.nima.2009.03.204")
        """
        model_data, model_name = self._get_model_data(model_name, **kwargs)
        model_class = MODELS[model_data.function]

        return model_class(model_data, **kwargs)

    def get_model_signature(self, model_name: Optional[str] = None) -> Signature:
        """
        Constructs a call signature for the `get_resolution_function` method with a specific
        :term:`model`.

        This method provides a programmatic way of inspecting the call signature of the
        `get_resolution_function` method required when calling it for the `model_name`
        :term:`model`. This is useful because its default signature uses the ``**kwargs`` construct
        to provide a unified interface, but in fact different :term:`models<model>` require
        different sets of parameters that have to be passed in through the keyword arguments.

        There are other methods and properties that can be used to inspect some of the options, but
        this method retrieves all the information and returns it as an `inspect.Signature` object
        that can be used to examine the signature in detail. The only other comprehensive source
        of this information is the documentation for the relevant model.

        Parameters
        ----------
        model_name
            The name of the model whose signature to construct. If not provided, the signature of
            the `default_model` is constructed.

        Returns
        -------
        signature
            The call signature of the `get_resolution_method` for the `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.

        See Also
        --------
        available_models : List of models available for this version of this instrument.
        get_model_data : Allows for checking the default values of and restrictions on model parameters.
        possible_options_for_model : Lists the settings and their options for a model.

        Examples
        --------
        >>> maps = Instrument.from_default('MAPS')
        >>> sig = maps.get_model_signature()
        >>> sig
        <Signature (model_name: Optional[str] = 'PyChop_fit', *, chopper_package: Literal['A', 'B', 'S'] = 'A', e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500, chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400, fitting_order: 'Optional[int]' = 4, _)>
        >>> sig.parameters['e_init']
        <Parameter "e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500">
        >>> sig.parameters['e_init'].kind
        <_ParameterKind.KEYWORD_ONLY: 3>

        The `inspect.Signature` object provides easy inspection of a call signature.
        """
        from inspect import signature, Signature, Parameter
        from typing import Annotated, Literal

        model_data, model_name = self._get_model_data(model_name)
        model_class = MODELS[model_data.function]

        signature = signature(model_class)

        params = {
            'model_name': Parameter('model_name',
                                    Parameter.POSITIONAL_OR_KEYWORD,
                                    default=model_name,
                                    annotation=Optional[str])
        }

        for setting_name, options in self._models[model_name]['settings'].items():
            option_names = self._get_options(options)
            params[setting_name] = Parameter(setting_name,
                                             Parameter.KEYWORD_ONLY,
                                             default=options['default_setting'],
                                             annotation=Literal[tuple(option_names)])

        for key, value in signature.parameters.items():
            if key == 'model_data':
                continue

            args = {}
            try:
                args['default'] = model_data.defaults[key]
            except KeyError:
                pass

            try:
                args['annotation'] = Annotated[value.annotation, f'restriction={model_data.restrictions[key]}']
            except KeyError:
                pass

            params[key] = value.replace(**args, kind=Parameter.KEYWORD_ONLY)

        return Signature(parameters=list(params.values()))

    @property
    def available_models(self) -> list[str]:
        """
        A list of all :term:`models<model>` available for this :term:`version` of this
        :term:`instrument`.

        Returns
        -------
        available_models
            A list of all available models.
        """
        return list(self._models.keys())

    @property
    def available_models_and_settings(self) -> dict[str, list[str]]:
        """
        A dictionary mapping each available :term:`model` to the user :term:`settings<setting>`
        available for that :term:`model`.

        All :term:`models<model>` available for this :term:`version` of this :term:`instrument`,
        and all their :term:`settings<setting>` are listed.

        Returns
        -------
        models_and_settings
            All models and all their settings.
        """
        return {model_name: list(model['settings'].keys()) for model_name, model in self._models.items()}

    @property
    def all_available_models_options(self) -> dict[str, dict[str, list[str]]]:
        """
        A dictionary mapping each available :term:`model`, to the user :term:`settings<setting>`
        and its :term:`options<option>`.

        All :term:`models<model>` available for this :term:`version` of this :term:`instrument`, all
        the :term:`settings<setting>` of each of the :term:`models<model>`, and all the
        :term:`options<option>` for each of the :term:`settings<setting>`, are listed.

        Returns
        -------
        everything
            All models, all their settings, and all their options.
        """
        return {model_name: {setting: self._get_options(value) for setting, value in list(model['settings'].items())}
                for model_name, model in self._models.items()}

    def possible_settings_for_model(self, model_name: str) -> list[str]:
        """
        Returns all the :term:`settings<setting>` that the `model_name` :term:`model` supports.

        Parameters
        ----------
        model_name
            The name of the model whose settings to retrieve.

        Returns
        -------
        settings
            A list of settings available for the `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        return list(model['settings'].keys())

    def possible_options_for_model(self, model_name: str) -> dict[str, list[str]]:
        """
        Returns a dictionary mapping all the :term:`settings<setting>` of the `model_name`
        :term:`model` to their :term:`options<option>`.

        Parameters
        ----------
        model_name
            The name of the model whose settings to retrieve.

        Returns
        -------
        settings_and_options
            All the settings available for the `model_name` model and all their possible options.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        return {setting: self._get_options(value) for setting, value in model['settings'].items()}

    def possible_options_for_model_and_setting(self, model_name: str, setting: str) -> list[str]:
        """
        Lists each :term:`option` that can be chosen for a given :term:`setting` of the `model_name`
        :term:`model`.

        Parameters
        ----------
        model_name
            The name of the model to which the `setting` belongs.
        setting
            The name of the setting whose to retrieve.

        Returns
        -------
        options
            A list of options available for the `setting` and `model_name`.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        InvalidSettingError
            If the provided `setting` is not supported for the `model_name` model of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        settings = model['settings']

        try:
            settings = settings[setting]
        except KeyError:
            raise InvalidSettingError(setting, model_name, self)

        return self._get_options(settings)

    @staticmethod
    def _get_options(setting: dict[str, Union[str, dict]]) -> list[str]:
        """
        Retrieves all the possible options from ``self._models[model_name]['settings'][setting]``.

        Private method that takes the subset of the raw data in ``_models`` that corresponds to one
        setting of one model, and lists the options, ignoring the ``default_setting`` parameter.

        Parameters
        ----------
        setting
            A dictionary corresponding to one setting of one model, containing all the options.

        Returns
        -------
        options
            A list of options as found in the provided `setting` dictionary.
        """
        return [value for value in setting.keys() if value != 'default_setting']

    def default_option_for_setting(self, model_name: str, setting: str) -> str:
        """
        Returns the default :term:`option` for the `setting` :term:`setting` of the `model_name`
        :term:`model` of this :term:`instrument`.

        Parameters
        ----------
        model_name
            The name of the model whose `setting` to look up.
        setting
            The name of the setting whose default option to retrieve.

        Returns
        -------
        default_option
            The default option for the `setting` setting.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        InvalidSettingError
            If the provided `setting` is not supported for the `model_name` model of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        settings = model['settings']

        try:
            settings = settings[setting]
        except KeyError:
            raise InvalidSettingError(setting, model_name, self)

        return settings['default_setting']
