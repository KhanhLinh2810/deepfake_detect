import argparse
import os
import roop.detect_deepfake
import signal
import roop.globals

from roop.utilities import has_image_extension

def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image or source video', dest='source_path')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.headless = roop.globals.source_path is not None


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    # if not roop.globals.headless:
    #     ui.update_status(message)
    return


def start() -> None:
    if not roop.detect_deepfake.pre_start():
        return
    # process image to image
    if has_image_extension(roop.globals.source_path):
        # process frame
        update_status('Progressing... Detect deepfake')
        roop.detect_deepfake.process_image(roop.globals.source_path)



def run() -> None:
    parse_args()
    if not roop.detect_deepfake.pre_check():
        return
    
    if roop.globals.headless:
        start()
    