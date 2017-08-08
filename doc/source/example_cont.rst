.. _continuous_eg:

Continuous Example
==================

First step is to import dct,

.. code-block:: python

	>>> import dct

A :ref:`continuous` object can then be created,

::
	
	>>> example_cont = dct.cont()
	"Initialised with empty matrices, please specify using "setABC"."

The warning message appears because the continous simulation object currently has no associated internal matrices. These can be set using `setABC()` or the object can be initialised by specifying the dimensions,

::

	>>> example_cont_2 = dct.cont(6,no=4,nu=3)

This defines a system with state vector of dimension 6, 3 inputs and 4 outputs. The A matrix is random and stable, B and C are just random. Another random matrix, or a specified matrix can be set instead,

::

	>>> example_cont_2.setA(dct.random_stable(6))
	>>> example_cont_2.setB(dct.random(6,3))
	

We can then verify the stability of the new A matrix,


::

	>>> print(example_cont_2.is_stable())
	True

In the same way, the observability and controllability can be checked. These retrun a boolean indicating controllability or observability and also the associated gramian.

::

	>>> obs, w_o = example_cont_2.is_observable())
	>>> print(obs)
	True

The step and impulse response of the setup can be plotted,

::

	>>> example_cont_2.plot_step([0,50])

However there are multiple other plotting options, such as saving the data to a file, adding a grid, altering the number of plot points and specifying inputs and outputs,

::

	>>> example_cont_2.plot_impulse([0,100],plot_points = 200, grid = True, filename = 'random_impulse.dat', inputs = [5])