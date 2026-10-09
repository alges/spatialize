.. _result:

**************
Result classes
**************

The classes the results of the estimations and searches derive from.

Every result shows itself as a summary, in the manner of statistical packages. It tells what was estimated
and how, the data, the ensemble and the statistics of the estimate, or the best configurations of a
search. Printing a result, or its representation at a prompt, gives the summary as text. A Jupyter
notebook shows it as HTML, and ``rich`` prints it with colours in a terminal. ``show()`` displays it
in the look of the session setting ``display`` (:doc:`session`), and ``summary()`` returns it as an
object with the same three renderings. The estimation and search results, the simulations of
:func:`~spatialize.gs.ess.ess_sample`, the Pareto search and the posterior analysis share this
behaviour.

.. currentmodule:: spatialize.result

.. autoclass:: Summarised
   :members: summary, show

.. autoclass:: GridSearchResult
   :members:
   :undoc-members:

.. autoclass:: EstimationResult
   :members:
   :undoc-members:
