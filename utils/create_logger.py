import os
import logging
import time

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def create_logger(output_path, cfg_name, main_file):

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%m-%d-%H-%M-%S'))
    head = '%(asctime)-15s %(message)s'
    filename=os.path.join(output_path, log_file)
    makedirs(filename)
    logging.basicConfig(filename=filename, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    with open(main_file, 'r') as f:
        logger.info(f.read())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
