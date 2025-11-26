import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import final_software_opencv as f
# ensure no classifier file
for fn in ('classifier_buffalo','classifier_buffalo.pkl'):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass
try:
    params=('classifier_buffalo','', '', '', '', '', '', '', '', '', '', '')
    f.recognize('i', params)
    print('No exception (unexpected)')
except Exception as e:
    print('Raised:', type(e).__name__, str(e))
