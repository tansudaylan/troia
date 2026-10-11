Getting started
===============

Installation
------------

Install Troia from its repository root:

.. code-block:: console

   python -m pip install -e .

Set ``TROIA_PATH`` to the repository root for local workflow inputs and
products.

Compact-object signatures
-------------------------

Run the deterministic signature example from the repository root:

.. code-block:: console

   python examples/compact_object_signatures.py --typefileplot png

The example compares beaming, ellipsoidal variation, and self-lensing
predictions for explicitly chosen binary parameters. These curves are
predictions rather than detections. Use Pergamon to evaluate compact-object
feature populations and to combine detections with independently measured
recovery efficiencies.