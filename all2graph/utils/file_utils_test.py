import os
import shutil
import traceback
import numpy as np
import pandas as pd
import all2graph as ag


def test_iter_files():
    def test_dir():
        for path in ag.iter_files('./'):
            print(path)

    def test_list():
        for path in ag.iter_files(['../info', ['../json'], './file_utils.py']):
            print(path)

    def test_error():
        try:
            list(ag.iter_files('./haha'))
        except ValueError:
            traceback.print_exc()
            return
        raise ValueError('test error failed')

    def test_warning():
        for path in ag.iter_files(['./haha', './'], error=False):
            print(path)

    with ag.Timer('test_iter_files'):
        test_dir()
        test_list()
        test_error()
        test_warning()


def test_dataframe_chunk_iter():
    def test_error():
        try:
            for _ in ag.iter_csv('./', chunksize=64):
                pass
        except ValueError:
            traceback.print_exc()
            return
        raise ValueError('test_error failed')

    def test_warning():
        for _ in ag.iter_csv('file_utils.py', chunksize=64, error=False):
            pass

    def test_concat_chip():
        df = pd.DataFrame(np.random.random(10000).reshape(100, 100))
        directory = 'test_temp_files'
        if not os.path.exists(directory):
            os.mkdir(directory)
        num_lines = 0
        for i in ag.tqdm(range(1, 20)):
            new_df = pd.concat([df] * i)
            new_df.to_csv(os.path.join(directory, str(i)+'.csv'))
            num_lines += new_df.shape[0]
        print(num_lines)
        num_lines2 = []
        for chunk in ag.tqdm(ag.iter_csv(directory, chunksize=64, concat_chip=True)):
            num_lines2.append(chunk.shape[0])

        assert sum(num_lines2) == num_lines
        assert set(num_lines2[:-1]) == {64}
        shutil.rmtree(directory)
        print('test_concat_chip success')

    def test_recurse():
        df = pd.DataFrame(np.random.random(10000).reshape(100, 100))
        directory = 'test_temp_files'
        if not os.path.exists(directory):
            os.mkdir(directory)
        for i in ag.tqdm(range(1, 20)):
            new_df = pd.concat([df] * i)
            new_df.to_csv(os.path.join(directory, str(i)+'.csv'))
        for _ in ag.iter_csv(directory, chunksize=64):
            pass

        try:
            for _ in ag.iter_csv(directory, chunksize=64, recurse=False):
                pass
        except:
            traceback.print_exc()
            print('test_recurse success')
            return
        finally:
            shutil.rmtree(directory)
        raise Exception

    test_error()
    test_warning()
    test_concat_chip()
    test_recurse()


def test_split_csv():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    data = np.random.random(10000)
    df = pd.DataFrame({'uid': data, 'data': data})
    for chunk in ag.iter_csv(df, chunksize=1000):
        assert chunk.shape[0] == 1000

    meta_df = ag.split_csv(df, 'temp', chunksize=1000, sel_cols=['uid'], concat_chip=False)
    assert meta_df.shape[0] == df.shape[0], meta_df.shape
    assert 'uid' in meta_df
    shutil.rmtree('temp')


if __name__ == '__main__':
    test_iter_files()
    test_dataframe_chunk_iter()
    test_split_csv()
    print('all success')
