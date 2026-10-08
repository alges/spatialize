.. _development:

################
Developer guide
################

These pages describe how Spatialize is built inside, how to add a decoder of one's own and how
changes are tested. A user who wants a local model that Spatialize does not offer can write it in
Python, without compiling anything (:doc:`python_decoders`), or add it to the library in C++
(:doc:`cpp_decoders`).

.. toctree::
   :maxdepth: 2

   architecture
   python_decoders
   cpp_decoders
   testing
