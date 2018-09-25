import subprocess
import os
import random
import re

from ..helper import thirdparty_binary

from ..config import TEMP_DIR

from ..models import LanguageModel

from ..exceptions import G2PError


class NgramLmTrainer(object):
    """Train an ngram language model from a list of sentences

    Parameters
    ----------
    language: str
        the path and language code
    input_dict : str
        path to the pronunciation dictionary

    """
    def __init__(self, text, model_path, ngram_order, temp_directory=None):
        super(NgramLmTrainer, self).__init__()
        if not temp_directory:
            temp_directory = TEMP_DIR
        self.temp_directory = os.path.join(temp_directory, 'LM')
        self.ngram_order = ngram_order
        self.name, _ = os.path.splitext(os.path.basename(model_path))
        self.temp_directory = os.path.join(temp_directory, self.name)
        os.makedirs(self.temp_directory, exist_ok=True)
        self.model_path = model_path
        self.text = text

    def train(self):
        input_path = os.path.join(self.temp_directory, 'input.txt')
        with open(input_path, 'w', encoding='utf8') as f:
            for t in self.text:
                f.write(t)
                f.write('\n')

        sym_path = os.path.join(self.temp_directory, 'full.syms')
        far_path = os.path.join(self.temp_directory, 'full.far')
        cnts_path = os.path.join(self.temp_directory, 'full.cnts')
        mod_path = os.path.join(self.temp_directory, 'full.mod')
        arpa_path = os.path.join(self.temp_directory, 'full.arpa')
        fst_path = os.path.join(self.temp_directory, 'model.fst')

        ngramsymbols_proc = subprocess.Popen([thirdparty_binary('ngramsymbols'),
                                              input_path, sym_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
        stdout, stderr = ngramsymbols_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngramsymbols', stderr.decode('utf8')))

        farcompile_proc = subprocess.Popen([thirdparty_binary('farcompilestrings'),
                                            '--symbols=' + sym_path, '--keep_symbols=1',
                                            input_path, far_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        stdout, stderr = farcompile_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('farcompilestrings', stderr.decode('utf8')))

        ngramcount_proc = subprocess.Popen([thirdparty_binary('ngramcount'),
                                            '--order={}'.format(self.ngram_order), far_path, cnts_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
        stdout, stderr = ngramcount_proc.communicate()
        if stderr:
            raise G2PError('There was an error in {}: {}'.format('ngramcount', stderr.decode('utf8')))