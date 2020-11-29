

.. _dictionary_generating:

***********************
Generating a dictionary
***********************

We have trained several G2P models that are available for download (:ref:`pretrained_g2p`).

Use
===

To construct a pronunciation dictionary from your .lab or .TextGrid files, simply input:

.. code-block:: bash

    mfa g2p g2p_model_path input_path output_path

The argument ``g2p_model_path`` can either be a fully specified path to a G2P model you've trained previously
or one that you've downloaded via the :code:`mfa download g2p` command (see :ref:`pretrained_g2p`). The
``input_path`` argument can either be a text file of words to generate transcriptions for or a corpus directory that
will be inspected for text transcripts and a word list will be compiled and pronunciations generated.  The
``output_path`` argument is the full path to where the resulting pronunciation dictionary should be saved.

.. note::

   Generating pronunciations to supplement your existing pronunciation
   dictionary can be done by running the validation utility (see :ref:`running_the_validator`), and then use the path
   to the ``oovs_found.txt`` file that it generates.


Pronunciation dictionaries can also be generated from the orthographies of the words themselves, rather than relying on
a trained G2P model.  This functionality should be reserved for languages with transparent orthographies, close to 1-to-1
grapheme-to-phoneme mapping.

.. code-block:: bash

    mfa g2p input_path output_path


See :ref:`dict_generating_example` for an example of how to use G2P functionality with a premade example.




