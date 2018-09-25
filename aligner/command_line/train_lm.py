import argparse
import os
import traceback

from textgrid import TextGrid, IntervalTier

from aligner.lm.trainer import NgramLmTrainer

from aligner.corpus import Corpus, load_text, parse_transcription
from aligner.exceptions import ArgumentError
from aligner.config import TEMP_DIR

from aligner.command_line.align import fix_path, unfix_path


def train_lm(args):
    if not args.temp_directory:
        temp_dir = TEMP_DIR
    else:
        temp_dir = os.path.expanduser(args.temp_directory)
    if os.path.isfile(args.input_path):
        text = load_text(args.input_path).splitlines()
    else:
        corpus = Corpus(args.input_path, '')
        text = get_text_set(corpus)

    t = NgramLmTrainer(text, args.output_model_path, temp_directory=temp_dir)
    t.train()


def get_text_set(corpus):
    text = list(corpus.text_mapping.values())
    decode_error_files = []
    textgrid_read_errors = {}
    for file_path in corpus.transcriptions_without_wavs:
        if file_path.endswith('.lab'):
            try:
                text = load_text(file_path)
            except UnicodeDecodeError:
                decode_error_files.append(file_path)
                continue
            words = parse_transcription(text)
            text.append(' '.join(words))
        else:
            tg = TextGrid()
            try:
                tg.read(file_path)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                textgrid_read_errors[file_path] = '\n'.join(
                    traceback.format_exception(exc_type, exc_value, exc_traceback))
                continue
            for i, ti in enumerate(tg.tiers):
                if ti.name.lower() == 'notes':
                    continue
                if not isinstance(ti, IntervalTier):
                    continue
                for interval in ti:
                    text = interval.mark.lower().strip()
                    words = parse_transcription(text)
                    if not words:
                        continue
                    text.append(' '.join(words))

    if decode_error_files:
        print('WARNING: The following files were not able to be decoded using utf8:\n\n'
              '{}'.format('\n'.join(decode_error_files)))
    if textgrid_read_errors:
        print('WARNING: The following TextGrid files were not able to be read:\n\n'
              '{}'.format('\n'.join(textgrid_read_errors.keys())))
    print('Generating language model based on {} transcriptions found in the corpus...'.format(len(text)))
    return text


def validate(args):
    if not os.path.exists(args.input_path):
        raise (ArgumentError('Could not find the input path {}'.format(args.input_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a language model from a corpus directory or text file")

    parser.add_argument("input_path", help="Location of existing corpus or text file")

    parser.add_argument("output_model_path", help="Desired location of generated model")
    parser.add_argument('-t', '--temp_directory', type=str, default='',
                        help='Temporary directory root to use for LM training, default is ~/Documents/MFA')


    args = parser.parse_args()
    fix_path()
    validate(args)
    train_lm(args)
    unfix_path()
