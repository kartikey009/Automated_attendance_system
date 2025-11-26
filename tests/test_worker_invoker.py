import threading, time, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from user_interface import Worker, _INVOKER

called = []

def on_done(result, exc):
    import threading
    called.append((result, exc, threading.current_thread().name))
    print('on_done invoked in thread:', threading.current_thread().name, 'result:', result, 'exc:', exc)

# test function: just returns 'ok' after a short sleep
def workfn():
    time.sleep(0.3)
    return 'ok'

w = Worker(workfn, args=(), on_done=on_done)
w.start()
# wait for worker thread to finish
w.join(timeout=2)
print('worker alive?', w.is_alive())
# If signal delivery requires an event loop, in many cases the slot may have been called synchronously.
print('called list:', called)
