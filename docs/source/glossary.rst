Glossary
========

This scientific library assumes from its users a large amount of knowledge and a familiarity with
a variety of terminology related to the topic. Furthermore, some amount of naming conventions have
been created here to encode all the necessary functionality in code. This page serves to briefly
explain and define all the terms used throughout the library, both the ones used as a standard in
the wider scientific literature, and the ones specified by this library. For more details, please
see the :doc:`theory` pages.


Glossary
--------

Most relevant terms
^^^^^^^^^^^^^^^^^^^

.. glossary::

    resolution
        The ability of an :term:`instrument` to show results in detail. The better the resolution,
        the higher the detail and the less broad the peaks.

    resolution function
        A mathematical function that models the :term:`resolution` of an :term:`instrument`, as a
        function of some physical property, such as energy or momentum.

    instrument
        A collection of physical devices used to perform an INS experiment. Different instruments are
        set up to specialise in different kinds of experiments, and so they consist of different devices
        placed at different distances and made of different materials. This results in different
        :term:`resolution` profiles.

    version
        A particular combination of parameters for an :term:`instrument`. Over time, instruments are
        upgraded by switching out and rearranging their components, resulting in a changed
        :term:`resolution` (hopefully improved!). For past and future compatibility, every time an
        upgrade significant enough to alter the :term:`resolution` is done, a new "version" is
        created for that :term:`instrument`.

    setting
        A set of :term:`options<option>` that can be made for a particular :term:`version` of a
        particular :term:`instrument`. This can be, for example, the choice of the
        :term:`Fermi chopper` parameters.

    option
        One of the options that can be chosen for a given :term:`setting`. This is tied to a
        particular :term:`version` of a particular :term:`instrument`. For example, if a
        :term:`Fermi chopper` is a :term:`setting`, the "option" is one of the choices that can be
        made for that :term:`Fermi chopper`.

    model
        A method to represent the :term:`resolution function`. There are different ways of
        approximating the :term:`resolution function`, each taking into account different levels of
        detail. The same model can often be used to describe multiple similar
        :term:`instruments<instrument>` or multiple :term:`versions<version>` of an
        :term:`instrument`, but in the context of this package, a model can only be accessed through
        a particular :term:`version` of an :term:`instrument`.


Neutron terminology
^^^^^^^^^^^^^^^^^^^

.. glossary::

    source
        A device that produces neutrons used for neutron experiments. In practice, this is usually
        either a nuclear reactor or spallation source (a particle accelerator).

    target
        A material used in spallation :term:`sources<source>`, which is hit by the particle produced
        from the accelerator. The collision emits neutrons which are used for the experiments.

    moderator
        A material used to slow down the neutrons produced by the collision with the :term:`target`.

    beam
        A collection of neutrons. In nuclear reactor :term:`sources<source>`, this is usually a
        continuous beam (somewhat similar to a laser beam), while in spallation
        :term:`sources<source>` this is usually a pulse (a short burst).

    chopper
        A rotating mechanical device designed to block the neutron :term:`beam` for some fraction of
        each revolution.

    Fermi chopper
        A :term:`chopper` designed to select a particular narrow slice of energies from the wide
        range of energies coming from the moderator. This kind of :term:`chopper` often has multiple
        openings, allowing the user to make a choice of which energies to use or how good a
        :term:`resolution` to obtain.

    disk chopper
        A type of :term:`chopper` shaped like a disk. Can be used for various purposes, such as
        improving :term:`resolution` or removing contamination, etc.

    sample
        The material being studied by the neutron experiment. Unless specified otherwise, in this
        library the term "sample" combines two things: the :term:`sample environment` and the sample
        itself. The latter is the stuff that the user brings to be studied, in :term:`INS` usually
        a single crystal or a powder.

    sample environment
        A device used to enclose the :term:`sample`, usually an aluminium "can". May consist of
        extra parts, such as a cooling system. Further parts may also be present, such as a vacuum
        pump or extra experimental devices like a Raman spectrometer, but these are generally not
        exposed to the neutron :term:`beam` and therefore shouldn't affect the :term:`resolution`.

    detector
        A device used used to detect the presence of a neutron. These are placed in an array such
        that the the neutrons that hit the :term:`sample`, hit the detectors, registering where and
        when the neutron hit the detector.


.. _abbreviations:

Abbreviations and Acronyms
--------------------------

.. glossary::

    INS
        Inelastic Neutron Scattering

    FWHM
        Full Width Half Maximum

    FWHH
        Full Width Half Height