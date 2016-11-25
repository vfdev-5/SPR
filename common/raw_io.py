
#
# Script
#

# Python
import logging
# Project
from raw_dataset import parse_line, get_user


def write_submission(submission_writer, test_file_reader,
                     target_labels, compute_predictions_func):
    """
    :param submission_writer:
    :param test_file_reader:
    :param target_labels: list of target labels
    :param compute_predictions_func:
        function that takes a parsed row as an entry and should return a list of predictions
        This list contains at most 7 indices of target labels
        e.g. [0, 1, 2] or [3, 1, 5, 6, 8, 10, 13] etc

    """
    logging.info("- Write submission")
    total = 0
    submission_writer.write("ncodpers,added_products\n")

    removed_rows = 0
    while True:
        line = test_file_reader.readline()[:-1]
        total += 1

        if line == '':
            break

        row = parse_line(line)

        # Write before row processing
        user = get_user(row)
        submission_writer.write(user + ',')

        predicted = compute_predictions_func(row)
        if predicted is None:
            removed_rows += 1
            logging.debug("--- Removed row : {}".format(row))
            submission_writer.write("\n")
            continue

        for p in predicted:
            submission_writer.write(target_labels[p] + ' ')

        if total % 1000000 == 0:
            logging.info('Read {} lines'.format(total))

        submission_writer.write("\n")

    logging.info("-- Removed rows : %s" % removed_rows)