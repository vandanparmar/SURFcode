.. _network_eg:

Network Example
===============


A network can be created and saved using the network creator. Alternatively,

.. code-block:: python

	>>> import dct
	>>> test_Adj = dct.generate_rand(10,0.4)

To import a network saved from the GUI,

::

	>>> test_Adj_2 = dct.load_from_file('test_graph.json')

A :ref:`network` object can then be created using this adjacency matrix,

:: 

	>>>  test_network = dct.network(test_Adj_2)

Which can then be turned into either a :ref:`continuous` object or a :ref:`discrete` object,

::

	>>> cont_network = test_network.generate_cont_sim()
	>>> disc_network = test_network.generate_disc_sim()

Where impulse and step reponses can be plotted,

::

	>>> cont_network.plot_impulse([0,50],grid=True)
	>>> disc_network.plot_step([0,10],output=[4,5])