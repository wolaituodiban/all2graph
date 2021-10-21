import all2graph as ag
import traceback


def test_rename():
    op = ag.Rename('a', 'b')

    def test_not_dict():
        try:
            op([1])
        except Exception:
            traceback.print_exc()
            print('success')

    test_not_dict()


if __name__ == '__main__':
    test_rename()
    print('aha')
