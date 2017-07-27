.. _discrete_eg:

Discrete Example
================


First step is to import dct,

.. code-block:: python

	>>> import dct

A :ref:`discrete` object can then be created,

::
	
	>>> example_disc = dct.disc()
	"Initialised with empty matrices, please specify using "setABC"."

The warning message appears because the discinous simulation object currently has no associated internal matrices. These can be set using `setABC()` or the object can be initialised by specifying the dimensions,

::

	>>> example_disc_2 = dct.disc(5,no=3,nu=4)

This defines a system with state vector of dimension 6, 3 inputs and 4 outputs. The A matrix is random and stable, B and C are just random. Another random matrix, or a specified matrix can be set instead,

::

	>>> example_disc_2.setA(dct.random_unit(5))
	>>> example_disc_2.setB(dct.random(5,4))
	
We can check the stability of the A matrix,

::

	>>> print(example_disc_2.is_stable())
	True

In the same way, the observability and discrollability can be checked. These retrun a boolean indicating discrollability or observability and also the associated gramian.

::

	>>> obs, w_o = example_disc_2.is_observable())
	>>> print(obs)
	True

The step and impulse response of the setup can be plotted,

::

	>>> example_disc_2.plot_step([0,50])

However there are multiple other plotting options, such as saving the data to a file, adding a grid and specifying inputs and outputs,

::

	>>> example_disc_2.plot_impulse([0,100], grid = True, filename = 'random_impulse.dat', inputs = [5])