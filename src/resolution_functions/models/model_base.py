"""
Abstract base classes for all instrument models.

The classes defined within provide a common interface that all models must follow. A model consists
of two objects:

1. Data required by that model (subclassed from `ModelData`)
2. The model itself (subclassed from `InstrumentModel`)

Any new models must implement a subclass of each of these base classes (see their individual
documentation for details about how to use them). Additionally, the model must be added to the
`models.MODELS` mapping (found in ``models/__init__.py``) or they won't be found from the
`Instrument` class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


class InvalidInputError(Exception):
    """
    A custom Exception, common to all models, signalling invalid user input.

    This exception should be raised whenever a user-provided parameter to a model is outside
    the valid bounds for that model of a particular instrument. For example, it is raised when the
    incident energy (``e_init``) provided to a model of a direct instrument is outside the the range
    available to that instrument.
    """


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelData(ABC):
    """
    Abstract base class for defining the data required by a model.

    Subclasses of this base class define all the static data (e.g. if an instrument is modelled by a
    polynomial, this class will contain the coefficients) that the associated `InstrumentModel` will
    have access to - the only things it will have at its disposal are the
    attributes defined here and user-choice parameters. Notably, though, this
    dataclass is used as a transient data storage - the model itself does not
    keep a reference to it and stores as little of its information as possible.

    Therefore, it is used largely as a data verification for the data contained in the yaml files.
    Subclasses of this class must mirror the data defined in the yaml files that use the model
    - this class is the programming encoding of that data. In other words, this class defines which
    data a model uses while the yaml files provide the concrete values for a specific version of a
    specific instrument. If there is any discrepancy between the two (either the yaml file missing
    or having extra data), an error will be raised.

    This base class is a `dataclasses.dataclass`, so all subclasses have to be written accordingly.
    Furthermore, it is created with ``slots=True`` and ``frozen=True`` to prevent tampering with the
    static data. These points should be kept in mind when writing a subclass of `InstrumentModel`,
    since they mean that the attributes of this class cannot be edited at runtime.

    Additionally, this class is only constructed via the `instrument.Instrument._get_model_data`
    method, which means that no complex types are supported inside this class, only the basic Python
    types supported by the yaml format. Please do not use ``dataclass`` magic; if the information
    required for the model to work requires more complex structuring, the use of `typing.TypedDict`
    is recommended.

    Lastly, this class provides an implementation for the `restrictions` and `defaults` properties,
    but one that is only valid for models that do not have restrictions or defaults. Any subclasses
    should overwrite these as appropriate for the model.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for the corresponding `InstrumentModel`.
    citation
        The citation for a particular model. Please use this to look up more details and cite the
        model.
    restrictions
    defaults
    """
    function: str
    citation: str

    @property
    def restrictions(self) -> dict[str, list[int | float]]:
        """
        The restrictions that the corresponding model places on each user-input argument.

        This is provided via a dictionary where the keys are the name of the user-input argument,
        and the values are the restriction for that argument. The restrictions can be the upper or
        lower bounds, or even a list of allowed values. If an invalid input is passed in, a
        `InvalidInputError` will be raised.

        Returns
        -------
        restrictions
            A mapping of user argument name to the corresponding restriction.
        """
        return {}

    @property
    def defaults(self) -> dict[str, Any]:
        """
        The defaults for each user-input argument for the corresponding model.

        This is provided as a dictionary where the keys are the name of the user-input argument,
        and the values are the default for that argument. The default will be used when a particular
        argument is not passed in.

        Returns
        -------
        defaults
            A mapping to user argument name to the corresponding default value.
        """
        return {}


class InstrumentModel(ABC):
    """
    Abstract base class for containing the code that defines a resolution function.

    In other words, the code representation of a mathematical function that models the resolution of
    an INS instrument. E.g., if the resolution of an instrument can be modelled using a polynomial,
    a subclass of this class will be the implementation of a polynomial, and `ModelData` will be the
    set of coefficient values.

    This base class provides the basic template for all subclasses but does not impose significant
    restrictions. However, there are some expectations that are not expressed with code:

    - The ``__init__`` method must be implemented, and it must take the corresponding `ModelData`
      subclass as its first positional argument.
      - The ``__init__`` method of this class must be called via ``super()``
      - The subclass ``__init__`` method should take all user-choice parameters as arguments, i.e.
        anything that was decided at the time of the experiment, such as the initial energy or
        chopper frequency.
        - These parameters should be ``Optional`` wherever possible, with the defaults for each
          instrument in the corresponding yaml files.
      - Any number of any other parameters is allowed, though ``__init__`` must not accept the
        parameters that are passed in to ``__call__``.
      - Each subclass must accept ``**kwargs``.
      - The ``__init__`` method should perform as much of the computation as possible, i.e. any
        computation that does not involve the ``__call__`` parameters.
      - No reference to the `ModelData` should be kept.
    - The ``__call__`` method must be implemented.
      - It must take the parameters that the particular model uses as argument. These can be any
        combination of the energy transfer (i.e. frequencies), the momentum value (q), and the
        momentum vector (q-vectors).
      - It must also take ``*args`` and ``**kwargs``.
    - The `input` and `output` class variables must be given specific values.
    - The `data_class` class variable must be assigned to the corresponding `ModelData` subclass.
    - Any additional defined methods should be private, but feel free to use your discretion.

    Parameters
    ----------
    model_data
        The data associated with the model

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    output
        The output of the ``__call__`` method.
    data_class
        The `ModelData` subclass associated with this particular model.
    citation
    """
    input: ClassVar[int]
    output: ClassVar[int]

    data_class: ClassVar[type[ModelData]]

    def __init__(self, model_data: ModelData, **_):
        self._citation = model_data.citation

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'{type(self).__name__}(citation="{self.citation}")'

    @property
    def citation(self) -> str:
        """
        The citation for this model. Please use this to look up more details and cite the model.

        Returns
        -------
        citation
            The citation.
        """
        return self._citation
