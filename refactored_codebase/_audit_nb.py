"""Full audit of birdclef_refactored.ipynb: dump all code cells with line numbers."""
import json

nb_path = r'c:\Users\koust\Documents\Kaggle\Bird_AudioPred\refactored_codebase\birdclef_refactored.ipynb'
nb = json.load(open(nb_path, 'r', encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        print(f"\n{'='*80}")
        print(f"CELL {i} ({len(c['source'])} lines)")
        print(f"{'='*80}")
        for j, line in enumerate(src.splitlines()):
            print(f"  {j:4d}: {line}")
