from PyQt5 import uic

with open('MainPy.py', 'w', encoding="utf-8") as fout:
   uic.compileUi('main.ui', fout)
print("ok")