Cyclic redundancy checks
########################

.. py:module:: amaranth.lib.crc

The :mod:`amaranth.lib.crc` module provides a cyclic redundancy check (CRC)
implementation in software and in hardware.

Many commonly used CRC algorithms are available with their parameters
predefined in the :class:`Catalog` class, while most other CRC algorithms can
be used by directly specifying their parameters in the :class:`Parameters` class.

.. autoclass:: Parameters

.. autoclass:: Processor

.. autoclass:: Predefined

.. autoclass:: Catalog
   :undoc-members:
