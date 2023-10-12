import sys

def print_training_info(args, all=False):
    print('==================== Training Setting ====================')

    if all: print(args)
    else:
        try: print(f'Epoch: {args.epoch}')
        except: pass

        try: print(f'LR: {args.lr}')
        except: pass

        try: print(f'Batch size: {args.batch_size}')
        except: pass

        try: print(f'GPU ID: {args.gpuid}')
        except: pass

    print('==========================================================')

class StdRedirect:
    def __init__(self, filename):
        self.stream = sys.stdout
        self.file = open(filename,'w')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        pass

    def __del__(self):
        self.file.close()