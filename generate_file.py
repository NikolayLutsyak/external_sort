import string
import argparse
import numpy as np

from pathlib import Path

VOCABULARY = list(string.ascii_letters)


def generate_random_string(string_len):
    """Generates string of specified len"""
    chars = np.random.choice(VOCABULARY, size=string_len, replace=True)
    random_string = ''.join(chars)
    return random_string

def generate_text_file(file_path, num_lines, max_string_len):
    """Generates file with random strings

    Parameters
    ----------
    file_path : object
        Path to file, where data will be stored
    num_lines : int
        Number of lines in file
    max_string_len : int
        Maximum number of characters in string. 
        Number of chars in string is distributed uniformly
        from 1 to max_string_len.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True) 
    with open(file_path, 'w') as file:
        for current_line in range(num_lines):
            string_len = np.random.randint(1, max_string_len)
            current_string = generate_random_string(string_len) + '\n'
            file.write(current_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            """Generates file with random strings"""
        )
    )

    parser.add_argument(
        '--file-path', '-f',
        dest='file_path',
        help='Path to file, where data will be stored'
    )

    parser.add_argument(
        '--num-lines', '-n',
        dest='num_lines',
        type=int,
        help='Number of lines in file'
    )

    parser.add_argument(
        '--max-string-len', '-m',
        dest='max_string_len',
        default=None,
        type=int,
        help=("Maximum number of characters in line. Number of chars in "
              "line is distributed uniformly from 1 to max_string_len.")
    )

    args = parser.parse_args()

    generate_text_file(args.file_path, args.num_lines, args.max_string_len)
