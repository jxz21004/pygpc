
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_algorithms/plot_algorithm_mestaticprojection.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_mestaticprojection.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_mestaticprojection.py:


Algorithm: MEStaticProjection
=============================
:: _label_algorithm_ME_static_projection:

.. GENERATED FROM PYTHON SOURCE LINES 6-14

.. code-block:: default

    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():
    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/mestaticprojection'   # filename of output
    save_session_format = ".pkl"           # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








.. GENERATED FROM PYTHON SOURCE LINES 15-17

Loading the model and defining the problem
------------------------------------------

.. GENERATED FROM PYTHON SOURCE LINES 17-27

.. code-block:: default


    # define model
    model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    problem = pygpc.Problem(model, parameters)








.. GENERATED FROM PYTHON SOURCE LINES 28-30

Setting up the algorithm
------------------------

.. GENERATED FROM PYTHON SOURCE LINES 30-61

.. code-block:: default


    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [3, 3]
    options["order_max"] = 3
    options["interaction_order"] = 2
    options["n_cpu"] = 0
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_fwd"
    options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["qoi"] = "all"
    options["classifier"] = "learning"
    options["classifier_options"] = {"clusterer": "KMeans",
                                     "n_clusters": 2,
                                     "classifier": "MLPClassifier",
                                     "classifier_solver": "lbfgs"}
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["grid"] = pygpc.Random
    options["grid_options"] = {"seed": 1}
    options["n_grid"] = 1000
    options["adaptive_sampling"] = False

    # define algorithm
    algorithm = pygpc.MEStaticProjection(problem=problem, options=options)








.. GENERATED FROM PYTHON SOURCE LINES 62-64

Running the gpc
---------------

.. GENERATED FROM PYTHON SOURCE LINES 64-71

.. code-block:: default


    # Initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Creating initial grid (<class 'pygpc.Grid.Random'>) with n_grid=1000
    Determining gPC approximation for QOI #0:
    =========================================
    Performing 1000 simulations!
    It/Sub-it: 3/2 Performing simulation 0001 from 1000 [                                        ] 0.1%
    Total function evaluation: 0.00045680999755859375 sec
    It/Sub-it: 3/2 Performing simulation 0001 from 2000 [                                        ] 0.1%
    Gradient evaluation: 0.012847900390625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.278750752598564
    Determining gPC approximation for QOI #1:
    =========================================
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.2804633059391995




.. GENERATED FROM PYTHON SOURCE LINES 72-74

Postprocessing
--------------

.. GENERATED FROM PYTHON SOURCE LINES 74-87

.. code-block:: default


    # read session
    session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

    # Post-process gPC
    pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                 output_idx=None,
                                 calc_sobol=True,
                                 calc_global_sens=True,
                                 calc_pdf=True,
                                 algorithm="sampling",
                                 n_samples=1e3)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    > Loading gpc session object: tmp/mestaticprojection.pkl
    > Loading gpc coeffs: tmp/mestaticprojection.hdf5
    > Adding results to: tmp/mestaticprojection.hdf5




.. GENERATED FROM PYTHON SOURCE LINES 88-92

Validation
----------
Validate gPC vs original model function (2D-surface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. GENERATED FROM PYTHON SOURCE LINES 92-100

.. code-block:: default

    pygpc.validate_gpc_plot(session=session,
                            coeffs=coeffs,
                            random_vars=list(problem.parameters_random.keys()),
                            n_grid=[51, 51],
                            output_idx=[0],
                            fn_out=None,
                            folder=None,
                            n_cpu=session.n_cpu)



.. image-sg:: /auto_algorithms/images/sphx_glr_plot_algorithm_mestaticprojection_001.png
   :alt: Original model, gPC approximation, Difference (Original vs gPC)
   :srcset: /auto_algorithms/images/sphx_glr_plot_algorithm_mestaticprojection_001.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 101-103

Validate gPC vs original model function (Monte Carlo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. GENERATED FROM PYTHON SOURCE LINES 103-120

.. code-block:: default

    nrmsd = pygpc.validate_gpc_mc(session=session,
                                  coeffs=coeffs,
                                  n_samples=int(1e4),
                                  output_idx=[0],
                                  fn_out=None,
                                  folder=None,
                                  plot=True,
                                  n_cpu=session.n_cpu)

    print("> Maximum NRMSD (gpc vs original): {:.2}%".format(max(nrmsd)))

    # On Windows subprocesses will import (i.e. execute) the main module at start.
    # You need to insert an if __name__ == '__main__': guard in the main module to avoid
    # creating subprocesses recursively.
    #
    # if __name__ == '__main__':
    #     main()



.. image-sg:: /auto_algorithms/images/sphx_glr_plot_algorithm_mestaticprojection_002.png
   :alt: plot algorithm mestaticprojection
   :srcset: /auto_algorithms/images/sphx_glr_plot_algorithm_mestaticprojection_002.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    > Maximum NRMSD (gpc vs original): 0.26%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.394 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_mestaticprojection.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example


    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_algorithm_mestaticprojection.py <plot_algorithm_mestaticprojection.py>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_algorithm_mestaticprojection.ipynb <plot_algorithm_mestaticprojection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
