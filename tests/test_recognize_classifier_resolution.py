import os, pickle
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import final_software_opencv as f

clf = 'classifier_buffalo.pkl'
with open(clf,'wb') as fh:
    pickle.dump({'embeddings':{}, 'threshold':0.6}, fh)
print('classifier file created:', os.path.exists(clf))
params = ('classifier_buffalo', '', '', '', '', '', '', '', '', '', '', '')
res = f.recognize('i', params)
print('recognize returned:', res)
os.remove(clf)
print('cleanup done')
