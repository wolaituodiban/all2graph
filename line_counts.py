import os


def line_counts(path, depth):
    lines = 0
    if os.path.isdir(path):
        for file in os.listdir(path):
            lines += line_counts(os.path.join(path, file), depth+1)
    elif path.endswith('.py') and 'test' not in os.path.split(path)[-1]:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() != '':
                    lines += 1
    if lines > 0:
        print('{}{}: {}'.format('\t' * depth, path, lines))
    return lines


if __name__ == '__main__':
    dir_path = os.path.dirname(__file__)
    line_counts(dir_path, 0)
