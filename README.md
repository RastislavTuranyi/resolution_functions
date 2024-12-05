# Resolution Functions

Python library for working with resolution functions of inelastic neutron scattering (INS) 
instruments. This package exists to centralise all things related to resolution of INS instruments 
and make it easier to work with. It pools related code from existing projects, namely 
[AbINS](https://github.com/mantidproject/mantid/tree/main/scripts/abins) and 
[PyChop](https://github.com/mducle/pychop/tree/main), as well as implementing code published in 
literature. The main purposes are:

1. Provide one, central place implementing various models for INS instruments (resolution functions)
2. Provide a simple way to obtain the broadening at a given frequency, for a given instrument and settings
3. (in progress) Provide a way to apply broadening to a spectrum

## Quick Start

The package can be installed with pip (see Installation). To start, import the main `Instrument` 
class and get the instrument object of your choice:

```
>>> from resolution_functions import Instrument
>>> tosca = Instrument('TOSCA')
>>> tosca
Instrument(name='TOSCA', version='TOSCA', models=['AbINS', 'book', 'vision'])
```

To get the resolution function, call the `get_resolution_function` method, which returns a callable 
that can be called to get the broadening at specified frequencies. However, you will need to know 
which model you want to use, as well as any model-specific parameters.

```
>>> # The available models for a given instrument can be queried:
>>> tosca.available_models
['AbINS', 'book', 'vision']
>>> # There are multiple ways of querying the model-specific parameters, but the most comprehensive is
>>> tosca.get_model_signature('book')
<Signature (model_name: Optional[str] = 'book', *, detector_bank: Literal['Backward', 'Forward'] = 'Backward', _)>
>>> # Now we can get the resolution function
>>> book = tosca.get_resolution_function('book', detector_bank='Forward')
>>> book
<resolution_functions.models.tosca_book.ToscaBookModel object at 0x000000000>
>>> book(100)
0.81802604002035
>>> import numpy as np
>>> book(np.array([100, 200, 300]))
array([0.81802604, 1.34222267, 1.88255039])
```

## Installation

This package can be installed using pip, though it is not yet on PyPI, so it has to be installed directly from GitHub:

```
pip install git+https://github.com/pace-neutrons/resolution_functions.git
```

or from a local copy:

```
git clone https://github.com/pace-neutrons/resolution_functions.git
pip install resolution_functions
```


