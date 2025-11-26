import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import final_software_opencv as f

cases = [
    [10,20,50,60,0.9],
    [10,20,60,80,0.7,1],
    (0,0,10,10),
    [1.5,2.5,5.6,7.8,0.5],
    [100,200,300,400,0.99,2]
]

for c in cases:
    try:
        print('IN:', c, '-> OUT:', f._parse_box(c))
    except Exception as e:
        print('IN:', c, '-> ERROR:', e)
