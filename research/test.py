import multiprocessing as mp

from time  import sleep
import sys

def daemon():
    p = mp.current_process()
    print('Starting:', p.name, p.pid)
    sys.stdout.flush()
    sleep(3)
    print('Exiting:', p.name, p.pid)
    sys.stdout.flush()

def non_daemon():
    p = mp.current_process()
    print('Starting:', p.name, p.pid)
    sys.stdout.flush()
    print('Exiting:', p.name, p.pid)
    sys.stdout.flush()
    
if __name__ == '__main__':
    d = mp.Process(target=daemon, name='daemon')
    d.daemon = True
    n = mp.Process(target=non_daemon, name='non daemon')
    n.daemon = False
    d.start()
    sleep(1)
    n.start()
    d.join(5)
    n.join()