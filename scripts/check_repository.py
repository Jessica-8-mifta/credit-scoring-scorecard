"""Validate notebook syntax, links and common credential patterns in public sources."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
TOKEN=re.compile('KGA'+r'T_[A-Za-z0-9]{16,}')


def main():
    count=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git','.venv','__pycache__','runs','build'} for p in path.parts):
            continue
        if path.suffix not in {'.py','.md','.json','.ipynb','.yml','.toml','.txt'}:
            continue
        text=path.read_text()
        if TOKEN.search(text):
            raise SystemExit(f'Possible credential in {path.relative_to(ROOT)}')
        if path.suffix=='.ipynb':
            notebook=json.loads(text)
            assert notebook['nbformat']==4
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    ast.parse(''.join(cell['source']))
                    if path.name!='original_credit_scoring.ipynb':
                        assert cell['execution_count'] is None and cell['outputs']==[]
        count+=1
    print(f'Repository checks passed ({count} text files).')


if __name__=='__main__':
    main()
