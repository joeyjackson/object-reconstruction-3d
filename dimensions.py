import pickle
import matplotlib.pyplot as plt


def main():
    o = [
        'rubiks',
        'rubiks2',
        'spray',
        'clothespin',
        'hex'
    ]

    for name in o:
        print(name)
        for res in range(0, 6):
            print('\tres{}: '.format(res), end=' ')
            model = pickle.load(open('obj/' + name + str(res) + '.obj', 'rb'))
            model.dimensions()

if __name__ == '__main__':
    main()
