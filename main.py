import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--predict', action == 'store_false')
    args = parser.parse_args()
    
