import locale
import os
import subprocess
import sys
from distutils.ccompiler import new_compiler

c = new_compiler()
c.initialize()
print(c.cc)
f = os.popen(str(c.cc), 'r')
res = f.read()
print(res)
cc = subprocess.check_output(str(c.cc), stderr=subprocess.STDOUT, shell=True)
print(cc)
enc = os.device_encoding(sys.stdout.fileno()) or locale.getpreferredencoding()
print(enc)
print(res.decode(enc).partition('\n')[0].strip())
print(cc.decode(enc).partition('\n')[0].strip())
