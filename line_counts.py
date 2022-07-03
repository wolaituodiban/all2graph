import os


def line_counts(path, depth, test):
    lines = 0
    if os.path.isdir(path):
        for file in os.listdir(path):
            lines += line_counts(os.path.join(path, file), depth+1, test)
    elif path.endswith('.py') and (('test' in os.path.split(path)[-1]) == test):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() != '':
                    lines += 1
    if lines > 0:
        print('{}{}: {}'.format('\t' * depth, path, lines))
    return lines


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))
    print('source files')
    line_counts(dir_path, 0, test=False)
    print('test files')
    line_counts(dir_path, 0, test=True)
