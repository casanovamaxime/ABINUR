import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Define periodic hill function
# -----------------------------
xh=[]
yh=[]
 
c=0
with open('wallpts.txt') as f:
    lines = f.readlines()
    for it,el in enumerate(lines):
            c=c+1
            if c == 3:
                data=el.split()
                xh.append(float(data[0]))
                yh.append(float(data[1]))
                c=0
            
print(len(xh))
plt.plot(xh,yh)
plt.show()

for i in range(0,len(xh)):
    print("Point(",98+i,") = {",xh[len(xh)-i-1],",",yh[len(xh)-i-1],",","0",",","lc","};")
    #print("Line(",98+i,") = {",97+i,", ",98+i,"};")

print(len(xh))
