import io
import os
import psutil
import heapq
import argparse

from pathlib import Path

import numpy as np


class ExternalMergeSort(object):
    """Class for sorting large files

    It uses the idea of classic in-memory merge sort.

    First, it splits the whole file on parts that can be processed in RAM.
    Each of these parts is sorted and placed in temp file 'split_{i}.txt'.
    Each part is sorted using heap sort in order to do it in-place.

    After that, we should merge all that files into one.
    It would be great to do it using only one pass throug this files:
        1. open all of them for read
        2. read next string from each file
        3. find min string (e.g. in file 'split_k.txt')
        4. write this line to final file
        5. read next line from 'split_k.txt'
        6. repeat 3-5 until EOF all files
    Third item is done using min-heap to do it in logarithmic time on the 
    number of files. 

    But we have one restriction. For each opened file we need buffer B to
    read information efficiently. So if number of 'split' files is really big,
    then B will be really small and reading and writing operations will be
    inefficient. So it means that we may need additional passes. 

    Let, M - memory limit, N - number of splitted files, B - buffer size,
    X - numer of simultaneosly opened files for reading.
    Then if we do merge using K passes, X = ⌈N^{1/K}⌉ and (X+1)B <= M.
    X+1 because we need another one buffer for writing the result file.
    Resolving this equations we get: 
    K = ⌈1/(log_{N}(M/B - 1))⌉. And B <=M/(X+1). It means that we can merge 
    our files in K passes and we also can update B to its upper bound in 
    order to increase efficiency.

    Parameters
    ----------
    memory_limit : int, optional, default=None
        Memory limit in bytes. If None all available ram will be used.
    buffer_size : int, optional, default=None
        Buffer size for reading and writing text files. If None default
        value from io will be used (io.DEFAULT_BUFFER_SIZE).
    update_buffer_size : bool, optional, default=True
        Whether to update buffer_size from it is specified values to its
        upper bound.
    """

    def __init__(self, memory_limit=None, buffer_size=None,
                 update_buffer_size=True):
        self.buffer_size = buffer_size or io.DEFAULT_BUFFER_SIZE
        self.memory_limit = memory_limit or psutil.virtual_memory().available
        self.update_buffer_size = update_buffer_size

    def _make_temp_dir(self):
        """Creates directory for temporary files"""
        temp_dir = Path(self.file_path.parent, self.file_path.name + '__tmp')
        temp_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = temp_dir

    def sort(self, file_path, save_path=None):
        """Sorts lines in file and saves result to another file

        Parameters
        ----------
        file_path : object
            Path to file that should be sorted
        save_path : object, default=None
            Path where to save the result. If None, then save_path is almost 
            the same with file_path except for suffix '__sorted'
        """
        self.file_path = Path(file_path)
        self._make_temp_dir()
        self._split()
        self._merge(save_path)

    def _make_tempfile_path(self, file_index, pass_index=None):
        """Returns path for temporary file"""
        if self.mode == 'split':
            p = self.temp_dir.joinpath(
                "{}_{}.txt".format(self.mode, file_index)
            )
        elif self.mode == 'merge':
            p = self.temp_dir.joinpath(
                "{}_{}_{}.txt".format(self.mode, pass_index, file_index)
            )
        else:
            raise ValueError('self.mode should be "merge" or "split"')
        return str(p.absolute())

    def _split(self):
        """Splits the initial file into several files which are sorted"""
        self.mode = 'split'
        with open(self.file_path, 'r') as f:
            block_index = 0
            block_lines = f.readlines(self.memory_limit)
            while block_lines:
                # use heap sort because it is made in-place
                heapq.heapify(block_lines)
                save_path = self._make_tempfile_path(block_index)
                with open(save_path, 'w') as temp_file:
                    for _ in range(len(block_lines)):
                        temp_file.write(heapq.heappop(block_lines))

                block_index += 1
                block_lines = f.readlines(self.memory_limit)

        self.num_split_files = block_index

    def _merge_files(self, files, save_path):
        """Merges specified files and save result

        After merging temporary files will be deleted.

        Parameters
        ----------
        files : list-like
            Paths to files that should be merged
        save_path : object
            Path for saving the result
        """
        opened_files = []
        for file in files:
            opened_files.append(open(file, buffering=self.buffer_size))

        with open(save_path, mode='w', buffering=self.buffer_size) as f:
            for line in heapq.merge(*opened_files):
                f.write(line)

        for file_stream, file_name in zip(opened_files, files):
            file_stream.close()
            os.remove(file_name)

    def _get_pass_params(self):
        """Computes parameters for merging

        Returns
        -------
        num_merges : int
            Number of passes through temporary files to merge them into
            final final
        num_files_to_merge : int
            Number of files to merge in one merge call
        buffer_size : int
            Updated buffer size
        """
        num_merges = int(np.ceil(
            1 /
            (np.log(self.memory_limit / self.buffer_size - 1) /
             np.log(self.num_split_files))
        ))
        num_files_to_merge = int(np.ceil(
            self.num_split_files ** (1 / num_merges)
        ))
        if self.update_buffer_size:
            buffer_size = int(
                self.memory_limit / (num_files_to_merge + 1)
            )
        else:
            buffer_size = self.buffer_size
        return num_merges, num_files_to_merge, buffer_size

    def _move_to_save_path(self, temp_path, save_path):
        """Moves final temporary file to dessired path and
        remove temp directory.

        Parameters
        ----------
        temp_path : object
            Path to result in temporary directory
        save_path : object
            Path for saving the result
        """

        if save_path is None:
            root, ext = os.path.splitext(self.file_path)
            save_path = root + '__sorted' + ext
        os.rename(temp_path, save_path)
        os.removedirs(str(self.temp_dir))

    def _merge(self, save_path):
        """Merge all split files and save result

        Parameters
        ----------
        save_path : object
            Path for saving the result
        """
        self.mode = 'merge'
        num_split_files = self.num_split_files
        if num_split_files == 1:
            self._move_to_save_path(
                Path(self.temp_dir, 'split_0.txt'),
                save_path
            )
            return

        num_merges, num_files_to_merge, buffer_size = self._get_pass_params()
        self.buffer_size = buffer_size

        for merge_index in range(num_merges):
            temp_files = list(map(str, self.temp_dir.iterdir()))
            num_split_files = len(temp_files)
            for start_index in range(0, num_split_files, num_files_to_merge):
                files_slice = slice(
                    start_index, start_index + num_files_to_merge)
                files_to_merge = temp_files[files_slice]

                file_index = int(np.ceil(start_index / num_files_to_merge))
                save_path_tmp = self._make_tempfile_path(
                    file_index, merge_index)
                self._merge_files(files_to_merge, save_path_tmp)

        self._move_to_save_path(save_path_tmp, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            ("Sorts string in file and write result to another file even if"
             " file can not be allocated in RAM")
        )
    )

    parser.add_argument(
        '--file-path', '-f',
        dest='file_path',
        help='Path to file that should be sorted'
    )

    parser.add_argument(
        '--save-path', '-s',
        dest='save_path',
        default=None,
        help=("Path where to save the result. If None, then save_path is"
              " almost the same with file_path except for suffix '__sorted'")
    )

    parser.add_argument(
        '--memory-limit', '-m',
        dest='memory_limit',
        default=None,
        type=int,
        help='Memory limit in bytes. If None all available RAM will be used.'
    )

    parser.add_argument(
        '--buffer-size', '-b',
        dest='buffer_size',
        default=None,
        type=int,
        help=('Buffer size for reading and writing text files. If None default'
              'value from io will be used (io.DEFAULT_BUFFER_SIZE).')
    )

    parser.add_argument(
        '--update-buffer-size', '-u',
        dest='update_buffer_size',
        default=True,
        type=bool,
        help=("Whether to update buffer_size from it is specified values to"
              "its upper bound.")
    )

    args = parser.parse_args()

    merge_sort = ExternalMergeSort(args.memory_limit, args.buffer_size,
                                   args.update_buffer_size)
    merge_sort.sort(args.file_path, args.save_path)
