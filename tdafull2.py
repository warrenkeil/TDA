#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:00:45 2018

@author: warrenkeil
"""

 ###################   To get wd >>>>      os.getcwd()
 
 ##################   To  set wd >>>>      os.chdir('/Users/warrenkeil/documents') 
 
 
import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx


os.chdir('/Users/warrenkeil/documents/TDA') 
print('hi') 

cols = ['id', 'asin','title','group','salesrank','similar','categories','reviews','avgrating']
#index= list(range(1,548552))
index= list(range(1,201))
df = pd.DataFrame(index=index, columns=cols)
df = df.fillna(0) # with 0s rather than NaNs  



#file = open("amtest.txt" , "r")    # this has 10

file = open("amtest200.txt" , "r")    # this has 200

lines = file.readlines()


for line in lines:
    if line.startswith('\n'):
        print('hidfhoihwef')



for line in lines:
    if line.startswith('Id:'):
        u=  int(line[ line.index('Id:')+len('Id:') : line.rindex('\n') ])
        print('line is ', line, 'u is ', u )
        for i in range(line.index(line), line.rindex('\n')):
            print( lines[i] )
















#   This assigns the Id into df   DONE
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('Id:'):
        #print('id is',  int(lines[i][ lines[i].index('Id:')+len('Id:') : lines[i].rindex('\n') ]))
        #print('u is ' , u)
        #print('i is ',  i)
        df.iloc[u,0] = int(lines[i][ lines[i].index('Id:')+len('Id:') : lines[i].rindex('\n') ])
        u=u+1

    

print( ' id done' )

#   This assigns the asin into df   
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('ASIN:'):
        #print('id is',  int(lines[i][ lines[i].index('Id:')+len('Id:') : lines[i].rindex('\n') ]))
        #print('u is ' , u)
        #print('i is ',  i)
        df.iloc[u,1] =lines[i][ lines[i].index('ASIN:')+len('ASIN: ') : lines[i].rindex('\n') ]
        u=u+1

print( ' asin done' )


'''

#       This loop assigns title to df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  title:'):
        #print( lines[i][9:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,2] = lines[i][lines[i].index('  title:')+len('   title:') :lines[i].rindex('\n') ]
        u=u+1 
    
print( ' title done' ) 
    
'''


    


    
# This assigns group into df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  group:'):
        #print( lines[i][9:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,3] = lines[i][lines[i].index('  group:')+len('   group:') :lines[i].rindex('\n') ]
        u=u+1      

print( ' group done' )   
    
    
    




# This assigns salesrank into df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  salesrank:'):
        #print( lines[i][13:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,4] = int(lines[i][lines[i].index('  salesrank:')+len('   salesrank:') :lines[i].rindex('\n') ])
        u=u+1 
    
print( ' salesrank done' )   





'''
    
   # This assigns similar into df   [ use p.split and then p[0] = int(p[0])]
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  similar:'):
        #print( lines[i][11:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,5] = int(lines[i][lines[i].index('  similar:')+len('   similar:'):lines[i].index('  similar:')+len('   similar:')+2]) 
        #df.iloc[u,5] = p
        u=u+1  

print( ' similar done' )



'''


'''

# This assigns categories into df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  categories:'):
        #print( lines[i][13:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,6] = lines[i][lines[i].index('  categories:')+len('   categories:') :lines[i].rindex('\n') ]
        u=u+1 
        
print( ' categories done' )    
        
        
'''




# This assigns review into df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  reviews:'):
        #print( lines[i][13:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,7] =int(lines[i][lines[i].index('total:')+len('total:'):lines[i].index('downloaded')])
        u=u+1 
   
print( ' review done' )    
        





   # This assigns avg rating into df
u=0
for i in range(0,np.shape(lines)[0]):
    if lines[i].startswith('  reviews:'):
        #print( lines[i][13:len(lines[i])-1])
        #print('u is ' , u, 'i is', i)
        df.iloc[u,8] =float(lines[i][lines[i].index('avg rating:')+len('avg rating:'):lines[i].index('\n')])
        u=u+1     
         
print( ' avg rating done' )        
  


df.to_csv('dfmeta2k.csv')



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:26:23 2018

@author: warrenkeil
"""

#Diffusion Frechet Functions - Networks
import numpy
import pdb
import math




def lapl_mat(adjmat):
    """calculates the laplacian matrix from an adjacency matrix"""
    #initializing diagonal matrix
    rows=adjmat.shape[0]
    D=numpy.zeros([rows,rows])
    #filling diagonal matrix
    d_aux=adjmat.sum(0)
    for i in numpy.arange(0,rows):
        D[i,i]=d_aux[i]
        return D-adjmat
    
    
#lm = lapl_mat(a111)  
    
    
    
def discrete_laplacian_eigen(adjmat):
    """calculates the eigenfunctions and eigenvalues of the discrete laplacian given an adjacency matrix"""
    #creating laplacian matrix
    laplmat=lapl_mat(adjmat)
    #finding eigenvalues and eigenvectors of laplacian matrix
    evalues,evectors=numpy.linalg.eigh(laplmat)
    #normalizing eigenfunctions
    for i in range(1,evalues.shape[0]):
        evectors[:,i]=evectors[:,i]*math.pow(numpy.dot(evectors[:,i],evectors[:,i]),-0.5)
        return evalues, evectors



#dlm = discrete_laplacian_eigen(a111)


def diffusion_distance(ind1,ind2,evalues,evectors,t):
    """calculates the difussion distances (squared) between 2 nodes on the graph with the discrete laplacian"""
    aux1=(evectors[ind1,:]-evectors[ind2,:])**2
    aux2=numpy.exp(-2*t*evalues)
    return numpy.dot(aux1,aux2)


#dd = diffusion_distance(5,3,dlm[0], dlm[1],2)




def diff_dist_mat(evalues,evectors,t):
    """calculates the diffusion distance matrix between nodes using t for decay parameter of eigenvalues"""
    size=evalues.shape[0]
    dist_mat=numpy.zeros([size,size])
    for i in range(size):
        for j in range(i+1,size):
            dist_mat[i,j]=diffusion_distance(i,j,evalues,evectors,t)
            dist_mat[j,i]=dist_mat[i,j]
    return dist_mat


#ddm = diff_dist_mat(dlm[0],dlm[1],.1)




def dff_ntw(prob,dist_mat):
    """calculates the value of the diffusion Frechet function at each node given a probability vector on the nodes of the
    respective graph and its diffusion distances"""
    return numpy.dot(dist_mat,prob)



#dffntw = dff_ntw(p[0:200],ddm)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:53:09 2018

@author: warrenkeil
"""
 
import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx

print('hi') 


os.chdir('/Users/warrenkeil/documents/TDA') 

# file = open("amazon0302.txt" , "r")    

lines2 = pd.read_csv('amazon0302.txt', delim_whitespace=True)

# lines2.pop(0)

df200= pd.read_csv('dfmeta200.csv')

# dfmeta = pd.read_csv('dfmeta.csv') 


lines2[0:6]


def sigmoid(x):
  return 1 / (1 + math.exp(-(x-7)))

dfp = pd.read_csv('amazon0302.txt', delim_whitespace=True)

dfp[len(dfp.columns)]=0



#for i in range(0, dfp.shape[0]):
 #   if df200.iloc[dfp.iloc[i,0],4] == df200.iloc[dfp.iloc[i,1],4]:
  #      print( "c heck" )



for i in range(0, dfp.shape[0]):
    if dfp.iloc[i,0]<=199 and dfp.iloc[i,1]<=199:
        if df200.iloc[dfp.iloc[i,0],4] == df200.iloc[dfp.iloc[i,1],4]:
            #print( df200.iloc[dfp.iloc[i,0],4] == df200.iloc[dfp.iloc[i,1],4] )
            dfp.iloc[i,2] =max(.01, (.5)*((sigmoid(df200.iloc[dfp.iloc[i,0],8]))*df200.iloc[dfp.iloc[i,0],9]+(sigmoid(df200.iloc[dfp.iloc[i,1],8]))*df200.iloc[dfp.iloc[i,1],9]))         
        else:
            dfp.iloc[i,2]=.01
        
        
dfp.columns = ['a', 'b', 'c']

# iris.ix[~(iris['sepal length (cm)'] < 5)]


dfp = dfp[ dfp['a']<=199 ]

dfp = dfp[ dfp['b']<=199 ]

# dfp.to_csv('dfp.csv')
df200['rscore'] = 0

m = np.zeros((200,200))


for i in range(0,np.shape(dfp)[0]):
    m[dfp.iloc[i,0], dfp.iloc[i,1]] = dfp.iloc[i,2]




for i in range(0, 200):
    df200.iloc[i,10] = sigmoid(df200.iloc[i,8])*df200.iloc[i,9]       
        










#   ADD THIS EXTRA LINE TO DF

df200['rs'] = df200['rscore']


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:42:14 2018

@author: warrenkeil
"""

 
import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx


'''
print('hi') 


os.chdir('/Users/warrenkeil/documents/TDA') 

flow = d.fill_freudenthal(ddm)
rip = d.fill_rips(ddm,2,.3)

p = d.homology_persistence(flow)
dgms = d.init_diagrams(p, flow)



d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])

d.plot.plot_bars(dgms[0])
d.plot.plot_bars(dgms[1])

'''  







e0 = pd.DataFrame(np.zeros((773, 3)))
e0.columns = ['a','b','c']

u=0

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e0.iloc[k,0] = dfp.iloc[i,0]
        e0.iloc[k,1] = dfp.iloc[i,1]
        e0.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        

        
###########################   Loop for groups        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e0.iloc[i,0]),4] == df200.iloc[int(e0.iloc[i,0]),4]:
        e0.iloc[i,2]= e0.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 1 and  df200.iloc[int(e0.iloc[i,0]),10] <1) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 2 and  df200.iloc[int(e0.iloc[i,0]),10] <2) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 3 and  df200.iloc[int(e0.iloc[i,0]),10] <3) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 4 and  df200.iloc[int(e0.iloc[i,0]),10] <4) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 6 and  df200.iloc[int(e0.iloc[i,0]),10] <6) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 700000 and  df200.iloc[int(e0.iloc[i,0]),5] <700000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e0.iloc[i,0]),5] <1400000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e0.iloc[i,0]),5] <2100000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e0.iloc[i,0]),5] <2800000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e0.iloc[i,0]),5] <3500000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
    
    
    
    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:43:38 2018

@author: katieng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:42:14 2018

@author: warrenkeil
"""

 
import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx

'''
print('hi') 


os.chdir('/Users/warrenkeil/documents/TDA') 

flow = d.fill_freudenthal(ddm)
rip = d.fill_rips(ddm,2,.3)

p = d.homology_persistence(flow)
dgms = d.init_diagrams(p, flow)



d.plot.plot_diagram(dgms[0])
d.plot.plot_diagram(dgms[1])

d.plot.plot_bars(dgms[0])
d.plot.plot_bars(dgms[1])

'''
#os.chdir('/Users/katieng/documents/py')
#df200 = pd.read_csv('df200.csv')
#dfp = pd.read_csv('dfp.csv')



# e = 0

e0 = pd.DataFrame(np.zeros((773, 3)))
e0.columns = ['a','b','c']




u=0

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    

  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e0.iloc[k,0] = dfp.iloc[i,0]
        e0.iloc[k,1] = dfp.iloc[i,1]
        e0.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e0.iloc[i,0]),4] == df200.iloc[int(e0.iloc[i,0]),4]:
        e0.iloc[i,2]= e0.iloc[i,2]+.33
        

print('DDDDDDD')
###########################   Loop for review score        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 1 and  df200.iloc[int(e0.iloc[i,0]),10] <1) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 2 and  df200.iloc[int(e0.iloc[i,0]),10] <2) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 3 and  df200.iloc[int(e0.iloc[i,0]),10] <3) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 4 and  df200.iloc[int(e0.iloc[i,0]),10] <4) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 6 and  df200.iloc[int(e0.iloc[i,0]),10] <6) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))
     
    
    
print('ggggg')    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 700000 and  df200.iloc[int(e0.iloc[i,0]),5] < 700000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e0.iloc[i,0]),5] < 1400000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e0.iloc[i,0]),5] < 2100000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e0.iloc[i,0]),5] < 2800000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e0.iloc[i,0]),5] < 3500000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
    
    
e1 = pd.DataFrame(np.zeros((773, 3)))
e1.columns = ['a','b','c']




u=1

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e1.iloc[k,0] = dfp.iloc[i,0]
        e1.iloc[k,1] = dfp.iloc[i,1]
        e1.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e1.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e1.iloc[i,0]),4] == df200.iloc[int(e1.iloc[i,0]),4]:
        e1.iloc[i,2]= e1.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e1.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e1.iloc[i,0]),10] < 1 and  df200.iloc[int(e1.iloc[i,0]),10] <1) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e1.iloc[i,0]),10] < 2 and  df200.iloc[int(e1.iloc[i,0]),10] <2) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e1.iloc[i,0]),10] < 3 and  df200.iloc[int(e1.iloc[i,0]),10] <3) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e1.iloc[i,0]),10] < 4 and  df200.iloc[int(e1.iloc[i,0]),10] <4) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e1.iloc[i,0]),10] < 6 and  df200.iloc[int(e1.iloc[i,0]),10] <6) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e1.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e1.iloc[i,0]),5] < 700000 and  df200.iloc[int(e1.iloc[i,0]),5] < 700000) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e1.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e1.iloc[i,0]),5] < 1400000) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e1.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e1.iloc[i,0]),5] < 2100000) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e1.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e1.iloc[i,0]),5] < 2800000) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e1.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e1.iloc[i,0]),5] < 3500000) :
        e1.iloc[i,2]= e1.iloc[i,2]+(.33*(.2))
    
    
    
    
    e2 = pd.DataFrame(np.zeros((773, 3)))
e2.columns = ['a','b','c']




u=2

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e2.iloc[k,0] = dfp.iloc[i,0]
        e2.iloc[k,1] = dfp.iloc[i,1]
        e2.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e2.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e2.iloc[i,0]),4] == df200.iloc[int(e2.iloc[i,0]),4]:
        e2.iloc[i,2]= e2.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e2.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e2.iloc[i,0]),10] < 1 and  df200.iloc[int(e2.iloc[i,0]),10] <1) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e2.iloc[i,0]),10] < 2 and  df200.iloc[int(e2.iloc[i,0]),10] <2) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e2.iloc[i,0]),10] < 3 and  df200.iloc[int(e2.iloc[i,0]),10] <3) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e2.iloc[i,0]),10] < 4 and  df200.iloc[int(e2.iloc[i,0]),10] <4) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e2.iloc[i,0]),10] < 6 and  df200.iloc[int(e2.iloc[i,0]),10] <6) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e2.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e2.iloc[i,0]),5] < 700000 and  df200.iloc[int(e2.iloc[i,0]),5] < 700000) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e2.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e2.iloc[i,0]),5] < 1400000) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e2.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e2.iloc[i,0]),5] < 2100000) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e2.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e2.iloc[i,0]),5] < 2800000) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e2.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e2.iloc[i,0]),5] < 3500000) :
        e2.iloc[i,2]= e2.iloc[i,2]+(.33*(.2))
    
    
    
    
e3 = pd.DataFrame(np.zeros((773, 3)))
e3.columns = ['a','b','c']




u=3

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e3.iloc[k,0] = dfp.iloc[i,0]
        e3.iloc[k,1] = dfp.iloc[i,1]
        e3.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e3.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e3.iloc[i,0]),4] == df200.iloc[int(e3.iloc[i,0]),4]:
        e3.iloc[i,2]= e3.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e3.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e3.iloc[i,0]),10] < 1 and  df200.iloc[int(e3.iloc[i,0]),10] <1) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e3.iloc[i,0]),10] < 2 and  df200.iloc[int(e3.iloc[i,0]),10] <2) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e3.iloc[i,0]),10] < 3 and  df200.iloc[int(e3.iloc[i,0]),10] <3) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e3.iloc[i,0]),10] < 4 and  df200.iloc[int(e3.iloc[i,0]),10] <4) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e3.iloc[i,0]),10] < 6 and  df200.iloc[int(e3.iloc[i,0]),10] <6) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e3.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e3.iloc[i,0]),5] < 700000 and  df200.iloc[int(e3.iloc[i,0]),5] < 700000) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e3.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e3.iloc[i,0]),5] < 1400000) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e3.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e3.iloc[i,0]),5] < 2100000) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e3.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e3.iloc[i,0]),5] < 2800000) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e3.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e3.iloc[i,0]),5] < 3500000) :
        e3.iloc[i,2]= e3.iloc[i,2]+(.33*(.2))
    
e4 = pd.DataFrame(np.zeros((773, 3)))
e4.columns = ['a','b','c']




u=4

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e4.iloc[k,0] = dfp.iloc[i,0]
        e4.iloc[k,1] = dfp.iloc[i,1]
        e4.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e4.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e4.iloc[i,0]),4] == df200.iloc[int(e4.iloc[i,0]),4]:
        e4.iloc[i,2]= e4.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e4.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e4.iloc[i,0]),10] < 1 and  df200.iloc[int(e4.iloc[i,0]),10] <1) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e4.iloc[i,0]),10] < 2 and  df200.iloc[int(e4.iloc[i,0]),10] <2) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e4.iloc[i,0]),10] < 3 and  df200.iloc[int(e4.iloc[i,0]),10] <3) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e4.iloc[i,0]),10] < 4 and  df200.iloc[int(e4.iloc[i,0]),10] <4) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e4.iloc[i,0]),10] < 6 and  df200.iloc[int(e4.iloc[i,0]),10] <6) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e4.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e4.iloc[i,0]),5] < 700000 and  df200.iloc[int(e4.iloc[i,0]),5] < 700000) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e4.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e4.iloc[i,0]),5] < 1400000) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e4.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e4.iloc[i,0]),5] < 2100000) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e4.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e4.iloc[i,0]),5] < 2800000) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e4.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e4.iloc[i,0]),5] < 3500000) :
        e4.iloc[i,2]= e4.iloc[i,2]+(.33*(.2))
    
    
    
    
e5 = pd.DataFrame(np.zeros((773, 3)))
e5.columns = ['a','b','c']




u=5

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e5.iloc[k,0] = dfp.iloc[i,0]
        e5.iloc[k,1] = dfp.iloc[i,1]
        e5.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e5.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e5.iloc[i,0]),4] == df200.iloc[int(e5.iloc[i,0]),4]:
        e5.iloc[i,2]= e5.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e5.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e5.iloc[i,0]),10] < 1 and  df200.iloc[int(e5.iloc[i,0]),10] <1) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e5.iloc[i,0]),10] < 2 and  df200.iloc[int(e5.iloc[i,0]),10] <2) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e5.iloc[i,0]),10] < 3 and  df200.iloc[int(e5.iloc[i,0]),10] <3) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e5.iloc[i,0]),10] < 4 and  df200.iloc[int(e5.iloc[i,0]),10] <4) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e5.iloc[i,0]),10] < 6 and  df200.iloc[int(e5.iloc[i,0]),10] <6) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e5.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e5.iloc[i,0]),5] < 700000 and  df200.iloc[int(e5.iloc[i,0]),5] < 700000) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e5.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e5.iloc[i,0]),5] < 1400000) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e5.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e5.iloc[i,0]),5] < 2100000) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e5.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e5.iloc[i,0]),5] < 2800000) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e5.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e5.iloc[i,0]),5] < 3500000) :
        e5.iloc[i,2]= e5.iloc[i,2]+(.33*(.2))
    
e6 = pd.DataFrame(np.zeros((773, 3)))
e6.columns = ['a','b','c']




u=6

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e6.iloc[k,0] = dfp.iloc[i,0]
        e6.iloc[k,1] = dfp.iloc[i,1]
        e6.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e6.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e6.iloc[i,0]),4] == df200.iloc[int(e6.iloc[i,0]),4]:
        e6.iloc[i,2]= e6.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e6.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e6.iloc[i,0]),10] < 1 and  df200.iloc[int(e6.iloc[i,0]),10] <1) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e6.iloc[i,0]),10] < 2 and  df200.iloc[int(e6.iloc[i,0]),10] <2) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e6.iloc[i,0]),10] < 3 and  df200.iloc[int(e6.iloc[i,0]),10] <3) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e6.iloc[i,0]),10] < 4 and  df200.iloc[int(e6.iloc[i,0]),10] <4) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e6.iloc[i,0]),10] < 6 and  df200.iloc[int(e6.iloc[i,0]),10] <6) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e6.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e6.iloc[i,0]),5] < 700000 and  df200.iloc[int(e6.iloc[i,0]),5] < 700000) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e6.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e6.iloc[i,0]),5] < 1400000) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e6.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e6.iloc[i,0]),5] < 2100000) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e6.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e6.iloc[i,0]),5] < 2800000) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e6.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e6.iloc[i,0]),5] < 3500000) :
        e6.iloc[i,2]= e6.iloc[i,2]+(.33*(.2))
    
    

e7 = pd.DataFrame(np.zeros((773, 3)))
e7.columns = ['a','b','c']




u=7

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e7.iloc[k,0] = dfp.iloc[i,0]
        e7.iloc[k,1] = dfp.iloc[i,1]
        e7.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e7.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e7.iloc[i,0]),4] == df200.iloc[int(e7.iloc[i,0]),4]:
        e7.iloc[i,2]= e7.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e7.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e7.iloc[i,0]),10] < 1 and  df200.iloc[int(e7.iloc[i,0]),10] <1) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e7.iloc[i,0]),10] < 2 and  df200.iloc[int(e7.iloc[i,0]),10] <2) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e7.iloc[i,0]),10] < 3 and  df200.iloc[int(e7.iloc[i,0]),10] <3) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e7.iloc[i,0]),10] < 4 and  df200.iloc[int(e7.iloc[i,0]),10] <4) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e7.iloc[i,0]),10] < 6 and  df200.iloc[int(e7.iloc[i,0]),10] <6) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e7.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e7.iloc[i,0]),5] < 700000 and  df200.iloc[int(e7.iloc[i,0]),5] < 700000) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e7.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e7.iloc[i,0]),5] < 1400000) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e7.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e7.iloc[i,0]),5] < 2100000) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e7.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e7.iloc[i,0]),5] < 2800000) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e7.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e7.iloc[i,0]),5] < 3500000) :
        e7.iloc[i,2]= e7.iloc[i,2]+(.33*(.2))
    
    
e8 = pd.DataFrame(np.zeros((773, 3)))
e8.columns = ['a','b','c']




u=8

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e8.iloc[k,0] = dfp.iloc[i,0]
        e8.iloc[k,1] = dfp.iloc[i,1]
        e8.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e8.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e8.iloc[i,0]),4] == df200.iloc[int(e8.iloc[i,0]),4]:
        e8.iloc[i,2]= e8.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e8.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e8.iloc[i,0]),10] < 1 and  df200.iloc[int(e8.iloc[i,0]),10] <1) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e8.iloc[i,0]),10] < 2 and  df200.iloc[int(e8.iloc[i,0]),10] <2) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e8.iloc[i,0]),10] < 3 and  df200.iloc[int(e8.iloc[i,0]),10] <3) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e8.iloc[i,0]),10] < 4 and  df200.iloc[int(e8.iloc[i,0]),10] <4) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e8.iloc[i,0]),10] < 6 and  df200.iloc[int(e8.iloc[i,0]),10] <6) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e8.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e8.iloc[i,0]),5] < 700000 and  df200.iloc[int(e8.iloc[i,0]),5] < 700000) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e8.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e8.iloc[i,0]),5] < 1400000) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e8.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e8.iloc[i,0]),5] < 2100000) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e8.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e8.iloc[i,0]),5] < 2800000) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e8.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e8.iloc[i,0]),5] < 3500000) :
        e8.iloc[i,2]= e8.iloc[i,2]+(.33*(.2))
    

e9 = pd.DataFrame(np.zeros((773, 3)))
e9.columns = ['a','b','c']




u=9

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e9.iloc[k,0] = dfp.iloc[i,0]
        e9.iloc[k,1] = dfp.iloc[i,1]
        e9.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e9.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e9.iloc[i,0]),4] == df200.iloc[int(e9.iloc[i,0]),4]:
        e9.iloc[i,2]= e9.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e9.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e9.iloc[i,0]),10] < 1 and  df200.iloc[int(e9.iloc[i,0]),10] <1) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e9.iloc[i,0]),10] < 2 and  df200.iloc[int(e9.iloc[i,0]),10] <2) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e9.iloc[i,0]),10] < 3 and  df200.iloc[int(e9.iloc[i,0]),10] <3) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e9.iloc[i,0]),10] < 4 and  df200.iloc[int(e9.iloc[i,0]),10] <4) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e9.iloc[i,0]),10] < 6 and  df200.iloc[int(e9.iloc[i,0]),10] <6) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e9.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e9.iloc[i,0]),5] < 700000 and  df200.iloc[int(e9.iloc[i,0]),5] < 700000) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e9.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e9.iloc[i,0]),5] < 1400000) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e9.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e9.iloc[i,0]),5] < 2100000) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e9.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e9.iloc[i,0]),5] < 2800000) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e9.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e9.iloc[i,0]),5] < 3500000) :
        e9.iloc[i,2]= e9.iloc[i,2]+(.33*(.2))
    
    
    
    
e10 = pd.DataFrame(np.zeros((773, 3)))
e10.columns = ['a','b','c']




u=10

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e10.iloc[k,0] = dfp.iloc[i,0]
        e10.iloc[k,1] = dfp.iloc[i,1]
        e10.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e10.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e10.iloc[i,0]),4] == df200.iloc[int(e10.iloc[i,0]),4]:
        e10.iloc[i,2]= e10.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e10.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e10.iloc[i,0]),10] < 1 and  df200.iloc[int(e10.iloc[i,0]),10] <1) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e10.iloc[i,0]),10] < 2 and  df200.iloc[int(e10.iloc[i,0]),10] <2) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e10.iloc[i,0]),10] < 3 and  df200.iloc[int(e10.iloc[i,0]),10] <3) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e10.iloc[i,0]),10] < 4 and  df200.iloc[int(e10.iloc[i,0]),10] <4) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e10.iloc[i,0]),10] < 6 and  df200.iloc[int(e10.iloc[i,0]),10] <6) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e10.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e10.iloc[i,0]),5] < 700000 and  df200.iloc[int(e10.iloc[i,0]),5] < 700000) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e10.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e10.iloc[i,0]),5] < 1400000) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e10.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e10.iloc[i,0]),5] < 2100000) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e10.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e10.iloc[i,0]),5] < 2800000) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e10.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e10.iloc[i,0]),5] < 3500000) :
        e10.iloc[i,2]= e10.iloc[i,2]+(.33*(.2))
    
    
    
    
e11 = pd.DataFrame(np.zeros((773, 3)))
e11.columns = ['a','b','c']




u=11

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e11.iloc[k,0] = dfp.iloc[i,0]
        e11.iloc[k,1] = dfp.iloc[i,1]
        e11.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e11.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e11.iloc[i,0]),4] == df200.iloc[int(e11.iloc[i,0]),4]:
        e11.iloc[i,2]= e11.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e11.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e11.iloc[i,0]),10] < 1 and  df200.iloc[int(e11.iloc[i,0]),10] <1) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e11.iloc[i,0]),10] < 2 and  df200.iloc[int(e11.iloc[i,0]),10] <2) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e11.iloc[i,0]),10] < 3 and  df200.iloc[int(e11.iloc[i,0]),10] <3) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e11.iloc[i,0]),10] < 4 and  df200.iloc[int(e11.iloc[i,0]),10] <4) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e11.iloc[i,0]),10] < 6 and  df200.iloc[int(e11.iloc[i,0]),10] <6) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e11.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e11.iloc[i,0]),5] < 700000 and  df200.iloc[int(e11.iloc[i,0]),5] < 700000) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e11.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e11.iloc[i,0]),5] < 1400000) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e11.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e11.iloc[i,0]),5] < 2100000) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e11.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e11.iloc[i,0]),5] < 2800000) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e11.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e11.iloc[i,0]),5] < 3500000) :
        e11.iloc[i,2]= e11.iloc[i,2]+(.33*(.2))
    
    
e12 = pd.DataFrame(np.zeros((773, 3)))
e12.columns = ['a','b','c']




u=12

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e12.iloc[k,0] = dfp.iloc[i,0]
        e12.iloc[k,1] = dfp.iloc[i,1]
        e12.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e12.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e12.iloc[i,0]),4] == df200.iloc[int(e12.iloc[i,0]),4]:
        e12.iloc[i,2]= e12.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e12.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e12.iloc[i,0]),10] < 1 and  df200.iloc[int(e12.iloc[i,0]),10] <1) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e12.iloc[i,0]),10] < 2 and  df200.iloc[int(e12.iloc[i,0]),10] <2) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e12.iloc[i,0]),10] < 3 and  df200.iloc[int(e12.iloc[i,0]),10] <3) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e12.iloc[i,0]),10] < 4 and  df200.iloc[int(e12.iloc[i,0]),10] <4) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e12.iloc[i,0]),10] < 6 and  df200.iloc[int(e12.iloc[i,0]),10] <6) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e12.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e12.iloc[i,0]),5] < 700000 and  df200.iloc[int(e12.iloc[i,0]),5] < 700000) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e12.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e12.iloc[i,0]),5] < 1400000) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e12.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e12.iloc[i,0]),5] < 2100000) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e12.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e12.iloc[i,0]),5] < 2800000) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e12.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e12.iloc[i,0]),5] < 3500000) :
        e12.iloc[i,2]= e12.iloc[i,2]+(.33*(.2))
    
    
    
    
e13 = pd.DataFrame(np.zeros((773, 3)))
e13.columns = ['a','b','c']




u=13

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e13.iloc[k,0] = dfp.iloc[i,0]
        e13.iloc[k,1] = dfp.iloc[i,1]
        e13.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e13.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e13.iloc[i,0]),4] == df200.iloc[int(e13.iloc[i,0]),4]:
        e13.iloc[i,2]= e13.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e13.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e13.iloc[i,0]),10] < 1 and  df200.iloc[int(e13.iloc[i,0]),10] <1) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e13.iloc[i,0]),10] < 2 and  df200.iloc[int(e13.iloc[i,0]),10] <2) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e13.iloc[i,0]),10] < 3 and  df200.iloc[int(e13.iloc[i,0]),10] <3) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e13.iloc[i,0]),10] < 4 and  df200.iloc[int(e13.iloc[i,0]),10] <4) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e13.iloc[i,0]),10] < 6 and  df200.iloc[int(e13.iloc[i,0]),10] <6) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e13.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e13.iloc[i,0]),5] < 700000 and  df200.iloc[int(e13.iloc[i,0]),5] < 700000) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e13.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e13.iloc[i,0]),5] < 1400000) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e13.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e13.iloc[i,0]),5] < 2100000) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e13.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e13.iloc[i,0]),5] < 2800000) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e13.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e13.iloc[i,0]),5] < 3500000) :
        e13.iloc[i,2]= e13.iloc[i,2]+(.33*(.2))
    
    
    
    
e14 = pd.DataFrame(np.zeros((773, 3)))
e14.columns = ['a','b','c']




u=14

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e14.iloc[k,0] = dfp.iloc[i,0]
        e14.iloc[k,1] = dfp.iloc[i,1]
        e14.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e14.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e14.iloc[i,0]),4] == df200.iloc[int(e14.iloc[i,0]),4]:
        e14.iloc[i,2]= e14.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e14.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e14.iloc[i,0]),10] < 1 and  df200.iloc[int(e14.iloc[i,0]),10] <1) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e14.iloc[i,0]),10] < 2 and  df200.iloc[int(e14.iloc[i,0]),10] <2) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e14.iloc[i,0]),10] < 3 and  df200.iloc[int(e14.iloc[i,0]),10] <3) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e14.iloc[i,0]),10] < 4 and  df200.iloc[int(e14.iloc[i,0]),10] <4) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e14.iloc[i,0]),10] < 6 and  df200.iloc[int(e14.iloc[i,0]),10] <6) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e14.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e14.iloc[i,0]),5] < 700000 and  df200.iloc[int(e14.iloc[i,0]),5] < 700000) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e14.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e14.iloc[i,0]),5] < 1400000) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e14.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e14.iloc[i,0]),5] < 2100000) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e14.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e14.iloc[i,0]),5] < 2800000) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e14.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e14.iloc[i,0]),5] < 3500000) :
        e14.iloc[i,2]= e14.iloc[i,2]+(.33*(.2))
    
    
    
    
e15 = pd.DataFrame(np.zeros((773, 3)))
e15.columns = ['a','b','c']




u=15

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e15.iloc[k,0] = dfp.iloc[i,0]
        e15.iloc[k,1] = dfp.iloc[i,1]
        e15.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e15.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e15.iloc[i,0]),4] == df200.iloc[int(e15.iloc[i,0]),4]:
        e15.iloc[i,2]= e15.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e15.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e15.iloc[i,0]),10] < 1 and  df200.iloc[int(e15.iloc[i,0]),10] <1) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e15.iloc[i,0]),10] < 2 and  df200.iloc[int(e15.iloc[i,0]),10] <2) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e15.iloc[i,0]),10] < 3 and  df200.iloc[int(e15.iloc[i,0]),10] <3) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e15.iloc[i,0]),10] < 4 and  df200.iloc[int(e15.iloc[i,0]),10] <4) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e15.iloc[i,0]),10] < 6 and  df200.iloc[int(e15.iloc[i,0]),10] <6) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e15.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e15.iloc[i,0]),5] < 700000 and  df200.iloc[int(e15.iloc[i,0]),5] < 700000) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e15.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e15.iloc[i,0]),5] < 1400000) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e15.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e15.iloc[i,0]),5] < 2100000) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e15.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e15.iloc[i,0]),5] < 2800000) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e15.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e15.iloc[i,0]),5] < 3500000) :
        e15.iloc[i,2]= e15.iloc[i,2]+(.33*(.2))
    
    
    
    
e16 = pd.DataFrame(np.zeros((773, 3)))
e16.columns = ['a','b','c']




u=16

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e16.iloc[k,0] = dfp.iloc[i,0]
        e16.iloc[k,1] = dfp.iloc[i,1]
        e16.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e16.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e16.iloc[i,0]),4] == df200.iloc[int(e16.iloc[i,0]),4]:
        e16.iloc[i,2]= e16.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e16.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e16.iloc[i,0]),10] < 1 and  df200.iloc[int(e16.iloc[i,0]),10] <1) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e16.iloc[i,0]),10] < 2 and  df200.iloc[int(e16.iloc[i,0]),10] <2) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e16.iloc[i,0]),10] < 3 and  df200.iloc[int(e16.iloc[i,0]),10] <3) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e16.iloc[i,0]),10] < 4 and  df200.iloc[int(e16.iloc[i,0]),10] <4) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e16.iloc[i,0]),10] < 6 and  df200.iloc[int(e16.iloc[i,0]),10] <6) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e16.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e16.iloc[i,0]),5] < 700000 and  df200.iloc[int(e16.iloc[i,0]),5] < 700000) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e16.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e16.iloc[i,0]),5] < 1400000) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e16.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e16.iloc[i,0]),5] < 2100000) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e16.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e16.iloc[i,0]),5] < 2800000) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e16.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e16.iloc[i,0]),5] < 3500000) :
        e16.iloc[i,2]= e16.iloc[i,2]+(.33*(.2))
    
    
    
    
e17 = pd.DataFrame(np.zeros((773, 3)))
e17.columns = ['a','b','c']




u=17

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e17.iloc[k,0] = dfp.iloc[i,0]
        e17.iloc[k,1] = dfp.iloc[i,1]
        e17.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e17.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e17.iloc[i,0]),4] == df200.iloc[int(e17.iloc[i,0]),4]:
        e17.iloc[i,2]= e17.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e17.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e17.iloc[i,0]),10] < 1 and  df200.iloc[int(e17.iloc[i,0]),10] <1) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e17.iloc[i,0]),10] < 2 and  df200.iloc[int(e17.iloc[i,0]),10] <2) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e17.iloc[i,0]),10] < 3 and  df200.iloc[int(e17.iloc[i,0]),10] <3) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e17.iloc[i,0]),10] < 4 and  df200.iloc[int(e17.iloc[i,0]),10] <4) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e17.iloc[i,0]),10] < 6 and  df200.iloc[int(e17.iloc[i,0]),10] <6) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e17.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e17.iloc[i,0]),5] < 700000 and  df200.iloc[int(e17.iloc[i,0]),5] < 700000) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e17.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e17.iloc[i,0]),5] < 1400000) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e17.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e17.iloc[i,0]),5] < 2100000) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e17.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e17.iloc[i,0]),5] < 2800000) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e17.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e17.iloc[i,0]),5] < 3500000) :
        e17.iloc[i,2]= e17.iloc[i,2]+(.33*(.2))
    
    
    
    
e18 = pd.DataFrame(np.zeros((773, 3)))
e18.columns = ['a','b','c']




u=18

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e18.iloc[k,0] = dfp.iloc[i,0]
        e18.iloc[k,1] = dfp.iloc[i,1]
        e18.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e18.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e18.iloc[i,0]),4] == df200.iloc[int(e18.iloc[i,0]),4]:
        e18.iloc[i,2]= e18.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e18.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e18.iloc[i,0]),10] < 1 and  df200.iloc[int(e18.iloc[i,0]),10] <1) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e18.iloc[i,0]),10] < 2 and  df200.iloc[int(e18.iloc[i,0]),10] <2) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e18.iloc[i,0]),10] < 3 and  df200.iloc[int(e18.iloc[i,0]),10] <3) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e18.iloc[i,0]),10] < 4 and  df200.iloc[int(e18.iloc[i,0]),10] <4) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e18.iloc[i,0]),10] < 6 and  df200.iloc[int(e18.iloc[i,0]),10] <6) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e18.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e18.iloc[i,0]),5] < 700000 and  df200.iloc[int(e18.iloc[i,0]),5] < 700000) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e18.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e18.iloc[i,0]),5] < 1400000) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e18.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e18.iloc[i,0]),5] < 2100000) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e18.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e18.iloc[i,0]),5] < 2800000) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e18.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e18.iloc[i,0]),5] < 3500000) :
        e18.iloc[i,2]= e18.iloc[i,2]+(.33*(.2))
    
    
    
e19 = pd.DataFrame(np.zeros((773, 3)))
e19.columns = ['a','b','c']




u=19

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e19.iloc[k,0] = dfp.iloc[i,0]
        e19.iloc[k,1] = dfp.iloc[i,1]
        e19.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e19.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e19.iloc[i,0]),4] == df200.iloc[int(e19.iloc[i,0]),4]:
        e19.iloc[i,2]= e19.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e19.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e19.iloc[i,0]),10] < 1 and  df200.iloc[int(e19.iloc[i,0]),10] <1) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e19.iloc[i,0]),10] < 2 and  df200.iloc[int(e19.iloc[i,0]),10] <2) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e19.iloc[i,0]),10] < 3 and  df200.iloc[int(e19.iloc[i,0]),10] <3) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e19.iloc[i,0]),10] < 4 and  df200.iloc[int(e19.iloc[i,0]),10] <4) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e19.iloc[i,0]),10] < 6 and  df200.iloc[int(e19.iloc[i,0]),10] <6) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e19.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e19.iloc[i,0]),5] < 700000 and  df200.iloc[int(e19.iloc[i,0]),5] < 700000) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e19.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e19.iloc[i,0]),5] < 1400000) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e19.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e19.iloc[i,0]),5] < 2100000) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e19.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e19.iloc[i,0]),5] < 2800000) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e19.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e19.iloc[i,0]),5] < 3500000) :
        e19.iloc[i,2]= e19.iloc[i,2]+(.33*(.2))
    
    
    
    
e20 = pd.DataFrame(np.zeros((773, 3)))
e20.columns = ['a','b','c']




u=20

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e20.iloc[k,0] = dfp.iloc[i,0]
        e20.iloc[k,1] = dfp.iloc[i,1]
        e20.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e20.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e20.iloc[i,0]),4] == df200.iloc[int(e20.iloc[i,0]),4]:
        e20.iloc[i,2]= e20.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e20.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e20.iloc[i,0]),10] < 1 and  df200.iloc[int(e20.iloc[i,0]),10] <1) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e20.iloc[i,0]),10] < 2 and  df200.iloc[int(e20.iloc[i,0]),10] <2) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e20.iloc[i,0]),10] < 3 and  df200.iloc[int(e20.iloc[i,0]),10] <3) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e20.iloc[i,0]),10] < 4 and  df200.iloc[int(e20.iloc[i,0]),10] <4) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e20.iloc[i,0]),10] < 6 and  df200.iloc[int(e20.iloc[i,0]),10] <6) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e20.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e20.iloc[i,0]),5] < 700000 and  df200.iloc[int(e20.iloc[i,0]),5] < 700000) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e20.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e20.iloc[i,0]),5] < 1400000) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e20.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e20.iloc[i,0]),5] < 2100000) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e20.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e20.iloc[i,0]),5] < 2800000) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e20.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e20.iloc[i,0]),5] < 3500000) :
        e20.iloc[i,2]= e20.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e21 = pd.DataFrame(np.zeros((773, 3)))
e21.columns = ['a','b','c']




u=21

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e21.iloc[k,0] = dfp.iloc[i,0]
        e21.iloc[k,1] = dfp.iloc[i,1]
        e21.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e21.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e21.iloc[i,0]),4] == df200.iloc[int(e21.iloc[i,0]),4]:
        e21.iloc[i,2]= e21.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e21.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e21.iloc[i,0]),10] < 1 and  df200.iloc[int(e21.iloc[i,0]),10] <1) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e21.iloc[i,0]),10] < 2 and  df200.iloc[int(e21.iloc[i,0]),10] <2) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e21.iloc[i,0]),10] < 3 and  df200.iloc[int(e21.iloc[i,0]),10] <3) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e21.iloc[i,0]),10] < 4 and  df200.iloc[int(e21.iloc[i,0]),10] <4) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e21.iloc[i,0]),10] < 6 and  df200.iloc[int(e21.iloc[i,0]),10] <6) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e21.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e21.iloc[i,0]),5] < 700000 and  df200.iloc[int(e21.iloc[i,0]),5] < 700000) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e21.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e21.iloc[i,0]),5] < 1400000) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e21.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e21.iloc[i,0]),5] < 2100000) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e21.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e21.iloc[i,0]),5] < 2800000) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e21.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e21.iloc[i,0]),5] < 3500000) :
        e21.iloc[i,2]= e21.iloc[i,2]+(.33*(.2))
    
    
    
    
e22 = pd.DataFrame(np.zeros((773, 3)))
e22.columns = ['a','b','c']




u=22

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e22.iloc[k,0] = dfp.iloc[i,0]
        e22.iloc[k,1] = dfp.iloc[i,1]
        e22.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e22.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e22.iloc[i,0]),4] == df200.iloc[int(e22.iloc[i,0]),4]:
        e22.iloc[i,2]= e22.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e22.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e22.iloc[i,0]),10] < 1 and  df200.iloc[int(e22.iloc[i,0]),10] <1) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e22.iloc[i,0]),10] < 2 and  df200.iloc[int(e22.iloc[i,0]),10] <2) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e22.iloc[i,0]),10] < 3 and  df200.iloc[int(e22.iloc[i,0]),10] <3) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e22.iloc[i,0]),10] < 4 and  df200.iloc[int(e22.iloc[i,0]),10] <4) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e22.iloc[i,0]),10] < 6 and  df200.iloc[int(e22.iloc[i,0]),10] <6) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e22.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e22.iloc[i,0]),5] < 700000 and  df200.iloc[int(e22.iloc[i,0]),5] < 700000) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e22.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e22.iloc[i,0]),5] < 1400000) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e22.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e22.iloc[i,0]),5] < 2100000) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e22.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e22.iloc[i,0]),5] < 2800000) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e22.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e22.iloc[i,0]),5] < 3500000) :
        e22.iloc[i,2]= e22.iloc[i,2]+(.33*(.2))
    
    
    
    
e23 = pd.DataFrame(np.zeros((773, 3)))
e23.columns = ['a','b','c']




u=23

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e23.iloc[k,0] = dfp.iloc[i,0]
        e23.iloc[k,1] = dfp.iloc[i,1]
        e23.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e23.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e23.iloc[i,0]),4] == df200.iloc[int(e23.iloc[i,0]),4]:
        e23.iloc[i,2]= e23.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e23.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e23.iloc[i,0]),10] < 1 and  df200.iloc[int(e23.iloc[i,0]),10] <1) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e23.iloc[i,0]),10] < 2 and  df200.iloc[int(e23.iloc[i,0]),10] <2) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e23.iloc[i,0]),10] < 3 and  df200.iloc[int(e23.iloc[i,0]),10] <3) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e23.iloc[i,0]),10] < 4 and  df200.iloc[int(e23.iloc[i,0]),10] <4) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e23.iloc[i,0]),10] < 6 and  df200.iloc[int(e23.iloc[i,0]),10] <6) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e23.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e23.iloc[i,0]),5] < 700000 and  df200.iloc[int(e23.iloc[i,0]),5] < 700000) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e23.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e23.iloc[i,0]),5] < 1400000) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e23.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e23.iloc[i,0]),5] < 2100000) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e23.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e23.iloc[i,0]),5] < 2800000) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e23.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e23.iloc[i,0]),5] < 3500000) :
        e23.iloc[i,2]= e23.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e24 = pd.DataFrame(np.zeros((773, 3)))
e24.columns = ['a','b','c']




u=24

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e24.iloc[k,0] = dfp.iloc[i,0]
        e24.iloc[k,1] = dfp.iloc[i,1]
        e24.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e24.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e24.iloc[i,0]),4] == df200.iloc[int(e24.iloc[i,0]),4]:
        e24.iloc[i,2]= e24.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e24.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e24.iloc[i,0]),10] < 1 and  df200.iloc[int(e24.iloc[i,0]),10] <1) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e24.iloc[i,0]),10] < 2 and  df200.iloc[int(e24.iloc[i,0]),10] <2) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e24.iloc[i,0]),10] < 3 and  df200.iloc[int(e24.iloc[i,0]),10] <3) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e24.iloc[i,0]),10] < 4 and  df200.iloc[int(e24.iloc[i,0]),10] <4) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e24.iloc[i,0]),10] < 6 and  df200.iloc[int(e24.iloc[i,0]),10] <6) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e24.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e24.iloc[i,0]),5] < 700000 and  df200.iloc[int(e24.iloc[i,0]),5] < 700000) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e24.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e24.iloc[i,0]),5] < 1400000) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e24.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e24.iloc[i,0]),5] < 2100000) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e24.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e24.iloc[i,0]),5] < 2800000) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e24.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e24.iloc[i,0]),5] < 3500000) :
        e24.iloc[i,2]= e24.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e25 = pd.DataFrame(np.zeros((773, 3)))
e25.columns = ['a','b','c']




u=25

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e25.iloc[k,0] = dfp.iloc[i,0]
        e25.iloc[k,1] = dfp.iloc[i,1]
        e25.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e25.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e25.iloc[i,0]),4] == df200.iloc[int(e25.iloc[i,0]),4]:
        e25.iloc[i,2]= e25.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e25.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e25.iloc[i,0]),10] < 1 and  df200.iloc[int(e25.iloc[i,0]),10] <1) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e25.iloc[i,0]),10] < 2 and  df200.iloc[int(e25.iloc[i,0]),10] <2) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e25.iloc[i,0]),10] < 3 and  df200.iloc[int(e25.iloc[i,0]),10] <3) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e25.iloc[i,0]),10] < 4 and  df200.iloc[int(e25.iloc[i,0]),10] <4) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e25.iloc[i,0]),10] < 6 and  df200.iloc[int(e25.iloc[i,0]),10] <6) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e25.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e25.iloc[i,0]),5] < 700000 and  df200.iloc[int(e25.iloc[i,0]),5] < 700000) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e25.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e25.iloc[i,0]),5] < 1400000) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e25.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e25.iloc[i,0]),5] < 2100000) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e25.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e25.iloc[i,0]),5] < 2800000) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e25.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e25.iloc[i,0]),5] < 3500000) :
        e25.iloc[i,2]= e25.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e26 = pd.DataFrame(np.zeros((773, 3)))
e26.columns = ['a','b','c']




u=26

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e26.iloc[k,0] = dfp.iloc[i,0]
        e26.iloc[k,1] = dfp.iloc[i,1]
        e26.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e26.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e26.iloc[i,0]),4] == df200.iloc[int(e26.iloc[i,0]),4]:
        e26.iloc[i,2]= e26.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e26.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e26.iloc[i,0]),10] < 1 and  df200.iloc[int(e26.iloc[i,0]),10] <1) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e26.iloc[i,0]),10] < 2 and  df200.iloc[int(e26.iloc[i,0]),10] <2) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e26.iloc[i,0]),10] < 3 and  df200.iloc[int(e26.iloc[i,0]),10] <3) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e26.iloc[i,0]),10] < 4 and  df200.iloc[int(e26.iloc[i,0]),10] <4) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e26.iloc[i,0]),10] < 6 and  df200.iloc[int(e26.iloc[i,0]),10] <6) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e26.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e26.iloc[i,0]),5] < 700000 and  df200.iloc[int(e26.iloc[i,0]),5] < 700000) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e26.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e26.iloc[i,0]),5] < 1400000) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e26.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e26.iloc[i,0]),5] < 2100000) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e26.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e26.iloc[i,0]),5] < 2800000) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e26.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e26.iloc[i,0]),5] < 3500000) :
        e26.iloc[i,2]= e26.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e27 = pd.DataFrame(np.zeros((773, 3)))
e27.columns = ['a','b','c']




u=27

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e27.iloc[k,0] = dfp.iloc[i,0]
        e27.iloc[k,1] = dfp.iloc[i,1]
        e27.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e27.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e27.iloc[i,0]),4] == df200.iloc[int(e27.iloc[i,0]),4]:
        e27.iloc[i,2]= e27.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e27.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e27.iloc[i,0]),10] < 1 and  df200.iloc[int(e27.iloc[i,0]),10] <1) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e27.iloc[i,0]),10] < 2 and  df200.iloc[int(e27.iloc[i,0]),10] <2) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e27.iloc[i,0]),10] < 3 and  df200.iloc[int(e27.iloc[i,0]),10] <3) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e27.iloc[i,0]),10] < 4 and  df200.iloc[int(e27.iloc[i,0]),10] <4) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e27.iloc[i,0]),10] < 6 and  df200.iloc[int(e27.iloc[i,0]),10] <6) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e27.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e27.iloc[i,0]),5] < 700000 and  df200.iloc[int(e27.iloc[i,0]),5] < 700000) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e27.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e27.iloc[i,0]),5] < 1400000) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e27.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e27.iloc[i,0]),5] < 2100000) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e27.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e27.iloc[i,0]),5] < 2800000) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e27.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e27.iloc[i,0]),5] < 3500000) :
        e27.iloc[i,2]= e27.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e28 = pd.DataFrame(np.zeros((773, 3)))
e28.columns = ['a','b','c']




u=28

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e28.iloc[k,0] = dfp.iloc[i,0]
        e28.iloc[k,1] = dfp.iloc[i,1]
        e28.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e28.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e28.iloc[i,0]),4] == df200.iloc[int(e28.iloc[i,0]),4]:
        e28.iloc[i,2]= e28.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e28.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e28.iloc[i,0]),10] < 1 and  df200.iloc[int(e28.iloc[i,0]),10] <1) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e28.iloc[i,0]),10] < 2 and  df200.iloc[int(e28.iloc[i,0]),10] <2) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e28.iloc[i,0]),10] < 3 and  df200.iloc[int(e28.iloc[i,0]),10] <3) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e28.iloc[i,0]),10] < 4 and  df200.iloc[int(e28.iloc[i,0]),10] <4) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e28.iloc[i,0]),10] < 6 and  df200.iloc[int(e28.iloc[i,0]),10] <6) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e28.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e28.iloc[i,0]),5] < 700000 and  df200.iloc[int(e28.iloc[i,0]),5] < 700000) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e28.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e28.iloc[i,0]),5] < 1400000) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e28.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e28.iloc[i,0]),5] < 2100000) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e28.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e28.iloc[i,0]),5] < 2800000) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e28.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e28.iloc[i,0]),5] < 3500000) :
        e28.iloc[i,2]= e28.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e29 = pd.DataFrame(np.zeros((773, 3)))
e29.columns = ['a','b','c']




u=29

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e29.iloc[k,0] = dfp.iloc[i,0]
        e29.iloc[k,1] = dfp.iloc[i,1]
        e29.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e29.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e29.iloc[i,0]),4] == df200.iloc[int(e29.iloc[i,0]),4]:
        e29.iloc[i,2]= e29.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e29.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e29.iloc[i,0]),10] < 1 and  df200.iloc[int(e29.iloc[i,0]),10] <1) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e29.iloc[i,0]),10] < 2 and  df200.iloc[int(e29.iloc[i,0]),10] <2) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e29.iloc[i,0]),10] < 3 and  df200.iloc[int(e29.iloc[i,0]),10] <3) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e29.iloc[i,0]),10] < 4 and  df200.iloc[int(e29.iloc[i,0]),10] <4) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e29.iloc[i,0]),10] < 6 and  df200.iloc[int(e29.iloc[i,0]),10] <6) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e29.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e29.iloc[i,0]),5] < 700000 and  df200.iloc[int(e29.iloc[i,0]),5] < 700000) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e29.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e29.iloc[i,0]),5] < 1400000) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e29.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e29.iloc[i,0]),5] < 2100000) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e29.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e29.iloc[i,0]),5] < 2800000) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e29.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e29.iloc[i,0]),5] < 3500000) :
        e29.iloc[i,2]= e29.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e30 = pd.DataFrame(np.zeros((773, 3)))
e30.columns = ['a','b','c']




u=30

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e30.iloc[k,0] = dfp.iloc[i,0]
        e30.iloc[k,1] = dfp.iloc[i,1]
        e30.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e30.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e30.iloc[i,0]),4] == df200.iloc[int(e30.iloc[i,0]),4]:
        e30.iloc[i,2]= e30.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e30.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e30.iloc[i,0]),10] < 1 and  df200.iloc[int(e30.iloc[i,0]),10] <1) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e30.iloc[i,0]),10] < 2 and  df200.iloc[int(e30.iloc[i,0]),10] <2) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e30.iloc[i,0]),10] < 3 and  df200.iloc[int(e30.iloc[i,0]),10] <3) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e30.iloc[i,0]),10] < 4 and  df200.iloc[int(e30.iloc[i,0]),10] <4) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e30.iloc[i,0]),10] < 6 and  df200.iloc[int(e30.iloc[i,0]),10] <6) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e30.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e30.iloc[i,0]),5] < 700000 and  df200.iloc[int(e30.iloc[i,0]),5] < 700000) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e30.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e30.iloc[i,0]),5] < 1400000) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e30.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e30.iloc[i,0]),5] < 2100000) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e30.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e30.iloc[i,0]),5] < 2800000) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e30.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e30.iloc[i,0]),5] < 3500000) :
        e30.iloc[i,2]= e30.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e31 = pd.DataFrame(np.zeros((773, 3)))
e31.columns = ['a','b','c']




u=31

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e31.iloc[k,0] = dfp.iloc[i,0]
        e31.iloc[k,1] = dfp.iloc[i,1]
        e31.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e31.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e31.iloc[i,0]),4] == df200.iloc[int(e31.iloc[i,0]),4]:
        e31.iloc[i,2]= e31.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e31.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e31.iloc[i,0]),10] < 1 and  df200.iloc[int(e31.iloc[i,0]),10] <1) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e31.iloc[i,0]),10] < 2 and  df200.iloc[int(e31.iloc[i,0]),10] <2) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e31.iloc[i,0]),10] < 3 and  df200.iloc[int(e31.iloc[i,0]),10] <3) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e31.iloc[i,0]),10] < 4 and  df200.iloc[int(e31.iloc[i,0]),10] <4) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e31.iloc[i,0]),10] < 6 and  df200.iloc[int(e31.iloc[i,0]),10] <6) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e31.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e31.iloc[i,0]),5] < 700000 and  df200.iloc[int(e31.iloc[i,0]),5] < 700000) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e31.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e31.iloc[i,0]),5] < 1400000) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e31.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e31.iloc[i,0]),5] < 2100000) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e31.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e31.iloc[i,0]),5] < 2800000) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e31.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e31.iloc[i,0]),5] < 3500000) :
        e31.iloc[i,2]= e31.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e32 = pd.DataFrame(np.zeros((773, 3)))
e32.columns = ['a','b','c']




u=32

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e32.iloc[k,0] = dfp.iloc[i,0]
        e32.iloc[k,1] = dfp.iloc[i,1]
        e32.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e32.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e32.iloc[i,0]),4] == df200.iloc[int(e32.iloc[i,0]),4]:
        e32.iloc[i,2]= e32.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e32.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e32.iloc[i,0]),10] < 1 and  df200.iloc[int(e32.iloc[i,0]),10] <1) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e32.iloc[i,0]),10] < 2 and  df200.iloc[int(e32.iloc[i,0]),10] <2) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e32.iloc[i,0]),10] < 3 and  df200.iloc[int(e32.iloc[i,0]),10] <3) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e32.iloc[i,0]),10] < 4 and  df200.iloc[int(e32.iloc[i,0]),10] <4) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e32.iloc[i,0]),10] < 6 and  df200.iloc[int(e32.iloc[i,0]),10] <6) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e32.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e32.iloc[i,0]),5] < 700000 and  df200.iloc[int(e32.iloc[i,0]),5] < 700000) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e32.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e32.iloc[i,0]),5] < 1400000) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e32.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e32.iloc[i,0]),5] < 2100000) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e32.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e32.iloc[i,0]),5] < 2800000) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e32.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e32.iloc[i,0]),5] < 3500000) :
        e32.iloc[i,2]= e32.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e33 = pd.DataFrame(np.zeros((773, 3)))
e33.columns = ['a','b','c']




u=33

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e33.iloc[k,0] = dfp.iloc[i,0]
        e33.iloc[k,1] = dfp.iloc[i,1]
        e33.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e33.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e33.iloc[i,0]),4] == df200.iloc[int(e33.iloc[i,0]),4]:
        e33.iloc[i,2]= e33.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e33.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e33.iloc[i,0]),10] < 1 and  df200.iloc[int(e33.iloc[i,0]),10] <1) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e33.iloc[i,0]),10] < 2 and  df200.iloc[int(e33.iloc[i,0]),10] <2) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e33.iloc[i,0]),10] < 3 and  df200.iloc[int(e33.iloc[i,0]),10] <3) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e33.iloc[i,0]),10] < 4 and  df200.iloc[int(e33.iloc[i,0]),10] <4) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e33.iloc[i,0]),10] < 6 and  df200.iloc[int(e33.iloc[i,0]),10] <6) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e33.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e33.iloc[i,0]),5] < 700000 and  df200.iloc[int(e33.iloc[i,0]),5] < 700000) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e33.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e33.iloc[i,0]),5] < 1400000) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e33.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e33.iloc[i,0]),5] < 2100000) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e33.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e33.iloc[i,0]),5] < 2800000) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e33.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e33.iloc[i,0]),5] < 3500000) :
        e33.iloc[i,2]= e33.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e34 = pd.DataFrame(np.zeros((773, 3)))
e34.columns = ['a','b','c']




u=34

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e34.iloc[k,0] = dfp.iloc[i,0]
        e34.iloc[k,1] = dfp.iloc[i,1]
        e34.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e34.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e34.iloc[i,0]),4] == df200.iloc[int(e34.iloc[i,0]),4]:
        e34.iloc[i,2]= e34.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e34.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e34.iloc[i,0]),10] < 1 and  df200.iloc[int(e34.iloc[i,0]),10] <1) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e34.iloc[i,0]),10] < 2 and  df200.iloc[int(e34.iloc[i,0]),10] <2) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e34.iloc[i,0]),10] < 3 and  df200.iloc[int(e34.iloc[i,0]),10] <3) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e34.iloc[i,0]),10] < 4 and  df200.iloc[int(e34.iloc[i,0]),10] <4) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e34.iloc[i,0]),10] < 6 and  df200.iloc[int(e34.iloc[i,0]),10] <6) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e34.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e34.iloc[i,0]),5] < 700000 and  df200.iloc[int(e34.iloc[i,0]),5] < 700000) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e34.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e34.iloc[i,0]),5] < 1400000) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e34.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e34.iloc[i,0]),5] < 2100000) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e34.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e34.iloc[i,0]),5] < 2800000) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e34.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e34.iloc[i,0]),5] < 3500000) :
        e34.iloc[i,2]= e34.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e35 = pd.DataFrame(np.zeros((773, 3)))
e35.columns = ['a','b','c']




u=35

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e35.iloc[k,0] = dfp.iloc[i,0]
        e35.iloc[k,1] = dfp.iloc[i,1]
        e35.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e35.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e35.iloc[i,0]),4] == df200.iloc[int(e35.iloc[i,0]),4]:
        e35.iloc[i,2]= e35.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e35.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e35.iloc[i,0]),10] < 1 and  df200.iloc[int(e35.iloc[i,0]),10] <1) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e35.iloc[i,0]),10] < 2 and  df200.iloc[int(e35.iloc[i,0]),10] <2) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e35.iloc[i,0]),10] < 3 and  df200.iloc[int(e35.iloc[i,0]),10] <3) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e35.iloc[i,0]),10] < 4 and  df200.iloc[int(e35.iloc[i,0]),10] <4) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e35.iloc[i,0]),10] < 6 and  df200.iloc[int(e35.iloc[i,0]),10] <6) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e35.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e35.iloc[i,0]),5] < 700000 and  df200.iloc[int(e35.iloc[i,0]),5] < 700000) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e35.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e35.iloc[i,0]),5] < 1400000) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e35.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e35.iloc[i,0]),5] < 2100000) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e35.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e35.iloc[i,0]),5] < 2800000) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e35.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e35.iloc[i,0]),5] < 3500000) :
        e35.iloc[i,2]= e35.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e36 = pd.DataFrame(np.zeros((773, 3)))
e36.columns = ['a','b','c']




u=36

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e36.iloc[k,0] = dfp.iloc[i,0]
        e36.iloc[k,1] = dfp.iloc[i,1]
        e36.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e36.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e36.iloc[i,0]),4] == df200.iloc[int(e36.iloc[i,0]),4]:
        e36.iloc[i,2]= e36.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e36.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e36.iloc[i,0]),10] < 1 and  df200.iloc[int(e36.iloc[i,0]),10] <1) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e36.iloc[i,0]),10] < 2 and  df200.iloc[int(e36.iloc[i,0]),10] <2) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e36.iloc[i,0]),10] < 3 and  df200.iloc[int(e36.iloc[i,0]),10] <3) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e36.iloc[i,0]),10] < 4 and  df200.iloc[int(e36.iloc[i,0]),10] <4) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e36.iloc[i,0]),10] < 6 and  df200.iloc[int(e36.iloc[i,0]),10] <6) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e36.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e36.iloc[i,0]),5] < 700000 and  df200.iloc[int(e36.iloc[i,0]),5] < 700000) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e36.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e36.iloc[i,0]),5] < 1400000) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e36.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e36.iloc[i,0]),5] < 2100000) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e36.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e36.iloc[i,0]),5] < 2800000) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e36.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e36.iloc[i,0]),5] < 3500000) :
        e36.iloc[i,2]= e36.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e37 = pd.DataFrame(np.zeros((773, 3)))
e37.columns = ['a','b','c']




u=37

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e37.iloc[k,0] = dfp.iloc[i,0]
        e37.iloc[k,1] = dfp.iloc[i,1]
        e37.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e37.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e37.iloc[i,0]),4] == df200.iloc[int(e37.iloc[i,0]),4]:
        e37.iloc[i,2]= e37.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e37.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e37.iloc[i,0]),10] < 1 and  df200.iloc[int(e37.iloc[i,0]),10] <1) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e37.iloc[i,0]),10] < 2 and  df200.iloc[int(e37.iloc[i,0]),10] <2) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e37.iloc[i,0]),10] < 3 and  df200.iloc[int(e37.iloc[i,0]),10] <3) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e37.iloc[i,0]),10] < 4 and  df200.iloc[int(e37.iloc[i,0]),10] <4) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e37.iloc[i,0]),10] < 6 and  df200.iloc[int(e37.iloc[i,0]),10] <6) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e37.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e37.iloc[i,0]),5] < 700000 and  df200.iloc[int(e37.iloc[i,0]),5] < 700000) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e37.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e37.iloc[i,0]),5] < 1400000) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e37.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e37.iloc[i,0]),5] < 2100000) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e37.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e37.iloc[i,0]),5] < 2800000) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e37.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e37.iloc[i,0]),5] < 3500000) :
        e37.iloc[i,2]= e37.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e38 = pd.DataFrame(np.zeros((773, 3)))
e38.columns = ['a','b','c']




u=38

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e38.iloc[k,0] = dfp.iloc[i,0]
        e38.iloc[k,1] = dfp.iloc[i,1]
        e38.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e38.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e38.iloc[i,0]),4] == df200.iloc[int(e38.iloc[i,0]),4]:
        e38.iloc[i,2]= e38.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e38.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e38.iloc[i,0]),10] < 1 and  df200.iloc[int(e38.iloc[i,0]),10] <1) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e38.iloc[i,0]),10] < 2 and  df200.iloc[int(e38.iloc[i,0]),10] <2) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e38.iloc[i,0]),10] < 3 and  df200.iloc[int(e38.iloc[i,0]),10] <3) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e38.iloc[i,0]),10] < 4 and  df200.iloc[int(e38.iloc[i,0]),10] <4) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e38.iloc[i,0]),10] < 6 and  df200.iloc[int(e38.iloc[i,0]),10] <6) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e38.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e38.iloc[i,0]),5] < 700000 and  df200.iloc[int(e38.iloc[i,0]),5] < 700000) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e38.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e38.iloc[i,0]),5] < 1400000) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e38.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e38.iloc[i,0]),5] < 2100000) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e38.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e38.iloc[i,0]),5] < 2800000) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e38.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e38.iloc[i,0]),5] < 3500000) :
        e38.iloc[i,2]= e38.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e39 = pd.DataFrame(np.zeros((773, 3)))
e39.columns = ['a','b','c']




u=39

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e39.iloc[k,0] = dfp.iloc[i,0]
        e39.iloc[k,1] = dfp.iloc[i,1]
        e39.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e39.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e39.iloc[i,0]),4] == df200.iloc[int(e39.iloc[i,0]),4]:
        e39.iloc[i,2]= e39.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e39.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e39.iloc[i,0]),10] < 1 and  df200.iloc[int(e39.iloc[i,0]),10] <1) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e39.iloc[i,0]),10] < 2 and  df200.iloc[int(e39.iloc[i,0]),10] <2) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e39.iloc[i,0]),10] < 3 and  df200.iloc[int(e39.iloc[i,0]),10] <3) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e39.iloc[i,0]),10] < 4 and  df200.iloc[int(e39.iloc[i,0]),10] <4) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e39.iloc[i,0]),10] < 6 and  df200.iloc[int(e39.iloc[i,0]),10] <6) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e39.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e39.iloc[i,0]),5] < 700000 and  df200.iloc[int(e39.iloc[i,0]),5] < 700000) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e39.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e39.iloc[i,0]),5] < 1400000) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e39.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e39.iloc[i,0]),5] < 2100000) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e39.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e39.iloc[i,0]),5] < 2800000) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e39.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e39.iloc[i,0]),5] < 3500000) :
        e39.iloc[i,2]= e39.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e40 = pd.DataFrame(np.zeros((773, 3)))
e40.columns = ['a','b','c']




u=40

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e40.iloc[k,0] = dfp.iloc[i,0]
        e40.iloc[k,1] = dfp.iloc[i,1]
        e40.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e40.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e40.iloc[i,0]),4] == df200.iloc[int(e40.iloc[i,0]),4]:
        e40.iloc[i,2]= e40.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e40.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e40.iloc[i,0]),10] < 1 and  df200.iloc[int(e40.iloc[i,0]),10] <1) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e40.iloc[i,0]),10] < 2 and  df200.iloc[int(e40.iloc[i,0]),10] <2) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e40.iloc[i,0]),10] < 3 and  df200.iloc[int(e40.iloc[i,0]),10] <3) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e40.iloc[i,0]),10] < 4 and  df200.iloc[int(e40.iloc[i,0]),10] <4) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e40.iloc[i,0]),10] < 6 and  df200.iloc[int(e40.iloc[i,0]),10] <6) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e40.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e40.iloc[i,0]),5] < 700000 and  df200.iloc[int(e40.iloc[i,0]),5] < 700000) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e40.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e40.iloc[i,0]),5] < 1400000) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e40.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e40.iloc[i,0]),5] < 2100000) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e40.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e40.iloc[i,0]),5] < 2800000) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e40.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e40.iloc[i,0]),5] < 3500000) :
        e40.iloc[i,2]= e40.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e41 = pd.DataFrame(np.zeros((773, 3)))
e41.columns = ['a','b','c']




u=41

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e41.iloc[k,0] = dfp.iloc[i,0]
        e41.iloc[k,1] = dfp.iloc[i,1]
        e41.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e41.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e41.iloc[i,0]),4] == df200.iloc[int(e41.iloc[i,0]),4]:
        e41.iloc[i,2]= e41.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e41.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e41.iloc[i,0]),10] < 1 and  df200.iloc[int(e41.iloc[i,0]),10] <1) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e41.iloc[i,0]),10] < 2 and  df200.iloc[int(e41.iloc[i,0]),10] <2) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e41.iloc[i,0]),10] < 3 and  df200.iloc[int(e41.iloc[i,0]),10] <3) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e41.iloc[i,0]),10] < 4 and  df200.iloc[int(e41.iloc[i,0]),10] <4) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e41.iloc[i,0]),10] < 6 and  df200.iloc[int(e41.iloc[i,0]),10] <6) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e41.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e41.iloc[i,0]),5] < 700000 and  df200.iloc[int(e41.iloc[i,0]),5] < 700000) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e41.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e41.iloc[i,0]),5] < 1400000) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e41.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e41.iloc[i,0]),5] < 2100000) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e41.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e41.iloc[i,0]),5] < 2800000) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e41.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e41.iloc[i,0]),5] < 3500000) :
        e41.iloc[i,2]= e41.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e42 = pd.DataFrame(np.zeros((773, 3)))
e42.columns = ['a','b','c']




u=42

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e42.iloc[k,0] = dfp.iloc[i,0]
        e42.iloc[k,1] = dfp.iloc[i,1]
        e42.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e42.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e42.iloc[i,0]),4] == df200.iloc[int(e42.iloc[i,0]),4]:
        e42.iloc[i,2]= e42.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e42.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e42.iloc[i,0]),10] < 1 and  df200.iloc[int(e42.iloc[i,0]),10] <1) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e42.iloc[i,0]),10] < 2 and  df200.iloc[int(e42.iloc[i,0]),10] <2) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e42.iloc[i,0]),10] < 3 and  df200.iloc[int(e42.iloc[i,0]),10] <3) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e42.iloc[i,0]),10] < 4 and  df200.iloc[int(e42.iloc[i,0]),10] <4) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e42.iloc[i,0]),10] < 6 and  df200.iloc[int(e42.iloc[i,0]),10] <6) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e42.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e42.iloc[i,0]),5] < 700000 and  df200.iloc[int(e42.iloc[i,0]),5] < 700000) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e42.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e42.iloc[i,0]),5] < 1400000) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e42.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e42.iloc[i,0]),5] < 2100000) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e42.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e42.iloc[i,0]),5] < 2800000) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e42.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e42.iloc[i,0]),5] < 3500000) :
        e42.iloc[i,2]= e42.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e43 = pd.DataFrame(np.zeros((773, 3)))
e43.columns = ['a','b','c']




u=43

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e43.iloc[k,0] = dfp.iloc[i,0]
        e43.iloc[k,1] = dfp.iloc[i,1]
        e43.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e43.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e43.iloc[i,0]),4] == df200.iloc[int(e43.iloc[i,0]),4]:
        e43.iloc[i,2]= e43.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e43.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e43.iloc[i,0]),10] < 1 and  df200.iloc[int(e43.iloc[i,0]),10] <1) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e43.iloc[i,0]),10] < 2 and  df200.iloc[int(e43.iloc[i,0]),10] <2) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e43.iloc[i,0]),10] < 3 and  df200.iloc[int(e43.iloc[i,0]),10] <3) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e43.iloc[i,0]),10] < 4 and  df200.iloc[int(e43.iloc[i,0]),10] <4) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e43.iloc[i,0]),10] < 6 and  df200.iloc[int(e43.iloc[i,0]),10] <6) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e43.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e43.iloc[i,0]),5] < 700000 and  df200.iloc[int(e43.iloc[i,0]),5] < 700000) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e43.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e43.iloc[i,0]),5] < 1400000) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e43.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e43.iloc[i,0]),5] < 2100000) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e43.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e43.iloc[i,0]),5] < 2800000) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e43.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e43.iloc[i,0]),5] < 3500000) :
        e43.iloc[i,2]= e43.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e44 = pd.DataFrame(np.zeros((773, 3)))
e44.columns = ['a','b','c']




u=44

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e44.iloc[k,0] = dfp.iloc[i,0]
        e44.iloc[k,1] = dfp.iloc[i,1]
        e44.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e44.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e44.iloc[i,0]),4] == df200.iloc[int(e44.iloc[i,0]),4]:
        e44.iloc[i,2]= e44.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e44.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e44.iloc[i,0]),10] < 1 and  df200.iloc[int(e44.iloc[i,0]),10] <1) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e44.iloc[i,0]),10] < 2 and  df200.iloc[int(e44.iloc[i,0]),10] <2) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e44.iloc[i,0]),10] < 3 and  df200.iloc[int(e44.iloc[i,0]),10] <3) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e44.iloc[i,0]),10] < 4 and  df200.iloc[int(e44.iloc[i,0]),10] <4) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e44.iloc[i,0]),10] < 6 and  df200.iloc[int(e44.iloc[i,0]),10] <6) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e44.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e44.iloc[i,0]),5] < 700000 and  df200.iloc[int(e44.iloc[i,0]),5] < 700000) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e44.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e44.iloc[i,0]),5] < 1400000) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e44.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e44.iloc[i,0]),5] < 2100000) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e44.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e44.iloc[i,0]),5] < 2800000) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e44.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e44.iloc[i,0]),5] < 3500000) :
        e44.iloc[i,2]= e44.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e45 = pd.DataFrame(np.zeros((773, 3)))
e45.columns = ['a','b','c']




u=45

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e45.iloc[k,0] = dfp.iloc[i,0]
        e45.iloc[k,1] = dfp.iloc[i,1]
        e45.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e45.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e45.iloc[i,0]),4] == df200.iloc[int(e45.iloc[i,0]),4]:
        e45.iloc[i,2]= e45.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e45.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e45.iloc[i,0]),10] < 1 and  df200.iloc[int(e45.iloc[i,0]),10] <1) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e45.iloc[i,0]),10] < 2 and  df200.iloc[int(e45.iloc[i,0]),10] <2) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e45.iloc[i,0]),10] < 3 and  df200.iloc[int(e45.iloc[i,0]),10] <3) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e45.iloc[i,0]),10] < 4 and  df200.iloc[int(e45.iloc[i,0]),10] <4) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e45.iloc[i,0]),10] < 6 and  df200.iloc[int(e45.iloc[i,0]),10] <6) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e45.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e45.iloc[i,0]),5] < 700000 and  df200.iloc[int(e45.iloc[i,0]),5] < 700000) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e45.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e45.iloc[i,0]),5] < 1400000) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e45.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e45.iloc[i,0]),5] < 2100000) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e45.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e45.iloc[i,0]),5] < 2800000) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e45.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e45.iloc[i,0]),5] < 3500000) :
        e45.iloc[i,2]= e45.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e46 = pd.DataFrame(np.zeros((773, 3)))
e46.columns = ['a','b','c']




u=46

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e46.iloc[k,0] = dfp.iloc[i,0]
        e46.iloc[k,1] = dfp.iloc[i,1]
        e46.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e46.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e46.iloc[i,0]),4] == df200.iloc[int(e46.iloc[i,0]),4]:
        e46.iloc[i,2]= e46.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e46.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e46.iloc[i,0]),10] < 1 and  df200.iloc[int(e46.iloc[i,0]),10] <1) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e46.iloc[i,0]),10] < 2 and  df200.iloc[int(e46.iloc[i,0]),10] <2) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e46.iloc[i,0]),10] < 3 and  df200.iloc[int(e46.iloc[i,0]),10] <3) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e46.iloc[i,0]),10] < 4 and  df200.iloc[int(e46.iloc[i,0]),10] <4) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e46.iloc[i,0]),10] < 6 and  df200.iloc[int(e46.iloc[i,0]),10] <6) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e46.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e46.iloc[i,0]),5] < 700000 and  df200.iloc[int(e46.iloc[i,0]),5] < 700000) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e46.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e46.iloc[i,0]),5] < 1400000) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e46.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e46.iloc[i,0]),5] < 2100000) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e46.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e46.iloc[i,0]),5] < 2800000) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e46.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e46.iloc[i,0]),5] < 3500000) :
        e46.iloc[i,2]= e46.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e47 = pd.DataFrame(np.zeros((773, 3)))
e47.columns = ['a','b','c']




u=47

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e47.iloc[k,0] = dfp.iloc[i,0]
        e47.iloc[k,1] = dfp.iloc[i,1]
        e47.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e47.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e47.iloc[i,0]),4] == df200.iloc[int(e47.iloc[i,0]),4]:
        e47.iloc[i,2]= e47.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e47.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e47.iloc[i,0]),10] < 1 and  df200.iloc[int(e47.iloc[i,0]),10] <1) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e47.iloc[i,0]),10] < 2 and  df200.iloc[int(e47.iloc[i,0]),10] <2) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e47.iloc[i,0]),10] < 3 and  df200.iloc[int(e47.iloc[i,0]),10] <3) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e47.iloc[i,0]),10] < 4 and  df200.iloc[int(e47.iloc[i,0]),10] <4) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e47.iloc[i,0]),10] < 6 and  df200.iloc[int(e47.iloc[i,0]),10] <6) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e47.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e47.iloc[i,0]),5] < 700000 and  df200.iloc[int(e47.iloc[i,0]),5] < 700000) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e47.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e47.iloc[i,0]),5] < 1400000) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e47.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e47.iloc[i,0]),5] < 2100000) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e47.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e47.iloc[i,0]),5] < 2800000) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e47.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e47.iloc[i,0]),5] < 3500000) :
        e47.iloc[i,2]= e47.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e48 = pd.DataFrame(np.zeros((773, 3)))
e48.columns = ['a','b','c']




u=48

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e48.iloc[k,0] = dfp.iloc[i,0]
        e48.iloc[k,1] = dfp.iloc[i,1]
        e48.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e48.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e48.iloc[i,0]),4] == df200.iloc[int(e48.iloc[i,0]),4]:
        e48.iloc[i,2]= e48.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e48.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e48.iloc[i,0]),10] < 1 and  df200.iloc[int(e48.iloc[i,0]),10] <1) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e48.iloc[i,0]),10] < 2 and  df200.iloc[int(e48.iloc[i,0]),10] <2) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e48.iloc[i,0]),10] < 3 and  df200.iloc[int(e48.iloc[i,0]),10] <3) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e48.iloc[i,0]),10] < 4 and  df200.iloc[int(e48.iloc[i,0]),10] <4) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e48.iloc[i,0]),10] < 6 and  df200.iloc[int(e48.iloc[i,0]),10] <6) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e48.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e48.iloc[i,0]),5] < 700000 and  df200.iloc[int(e48.iloc[i,0]),5] < 700000) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e48.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e48.iloc[i,0]),5] < 1400000) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e48.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e48.iloc[i,0]),5] < 2100000) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e48.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e48.iloc[i,0]),5] < 2800000) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e48.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e48.iloc[i,0]),5] < 3500000) :
        e48.iloc[i,2]= e48.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e49 = pd.DataFrame(np.zeros((773, 3)))
e49.columns = ['a','b','c']




u=49

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e49.iloc[k,0] = dfp.iloc[i,0]
        e49.iloc[k,1] = dfp.iloc[i,1]
        e49.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e49.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e49.iloc[i,0]),4] == df200.iloc[int(e49.iloc[i,0]),4]:
        e49.iloc[i,2]= e49.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e49.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e49.iloc[i,0]),10] < 1 and  df200.iloc[int(e49.iloc[i,0]),10] <1) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e49.iloc[i,0]),10] < 2 and  df200.iloc[int(e49.iloc[i,0]),10] <2) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e49.iloc[i,0]),10] < 3 and  df200.iloc[int(e49.iloc[i,0]),10] <3) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e49.iloc[i,0]),10] < 4 and  df200.iloc[int(e49.iloc[i,0]),10] <4) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e49.iloc[i,0]),10] < 6 and  df200.iloc[int(e49.iloc[i,0]),10] <6) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e49.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e49.iloc[i,0]),5] < 700000 and  df200.iloc[int(e49.iloc[i,0]),5] < 700000) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e49.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e49.iloc[i,0]),5] < 1400000) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e49.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e49.iloc[i,0]),5] < 2100000) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e49.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e49.iloc[i,0]),5] < 2800000) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e49.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e49.iloc[i,0]),5] < 3500000) :
        e49.iloc[i,2]= e49.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e50 = pd.DataFrame(np.zeros((773, 3)))
e50.columns = ['a','b','c']




u=50

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e50.iloc[k,0] = dfp.iloc[i,0]
        e50.iloc[k,1] = dfp.iloc[i,1]
        e50.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e50.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e50.iloc[i,0]),4] == df200.iloc[int(e50.iloc[i,0]),4]:
        e50.iloc[i,2]= e50.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e50.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e50.iloc[i,0]),10] < 1 and  df200.iloc[int(e50.iloc[i,0]),10] <1) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e50.iloc[i,0]),10] < 2 and  df200.iloc[int(e50.iloc[i,0]),10] <2) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e50.iloc[i,0]),10] < 3 and  df200.iloc[int(e50.iloc[i,0]),10] <3) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e50.iloc[i,0]),10] < 4 and  df200.iloc[int(e50.iloc[i,0]),10] <4) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e50.iloc[i,0]),10] < 6 and  df200.iloc[int(e50.iloc[i,0]),10] <6) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e50.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e50.iloc[i,0]),5] < 700000 and  df200.iloc[int(e50.iloc[i,0]),5] < 700000) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e50.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e50.iloc[i,0]),5] < 1400000) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e50.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e50.iloc[i,0]),5] < 2100000) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e50.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e50.iloc[i,0]),5] < 2800000) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e50.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e50.iloc[i,0]),5] < 3500000) :
        e50.iloc[i,2]= e50.iloc[i,2]+(.33*(.2))
    
    
    
    
e51 = pd.DataFrame(np.zeros((773, 3)))
e51.columns = ['a','b','c']




u=51

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e51.iloc[k,0] = dfp.iloc[i,0]
        e51.iloc[k,1] = dfp.iloc[i,1]
        e51.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e51.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e51.iloc[i,0]),4] == df200.iloc[int(e51.iloc[i,0]),4]:
        e51.iloc[i,2]= e51.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e51.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e51.iloc[i,0]),10] < 1 and  df200.iloc[int(e51.iloc[i,0]),10] <1) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e51.iloc[i,0]),10] < 2 and  df200.iloc[int(e51.iloc[i,0]),10] <2) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e51.iloc[i,0]),10] < 3 and  df200.iloc[int(e51.iloc[i,0]),10] <3) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e51.iloc[i,0]),10] < 4 and  df200.iloc[int(e51.iloc[i,0]),10] <4) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e51.iloc[i,0]),10] < 6 and  df200.iloc[int(e51.iloc[i,0]),10] <6) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e51.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e51.iloc[i,0]),5] < 700000 and  df200.iloc[int(e51.iloc[i,0]),5] < 700000) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e51.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e51.iloc[i,0]),5] < 1400000) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e51.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e51.iloc[i,0]),5] < 2100000) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e51.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e51.iloc[i,0]),5] < 2800000) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e51.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e51.iloc[i,0]),5] < 3500000) :
        e51.iloc[i,2]= e51.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e52 = pd.DataFrame(np.zeros((773, 3)))
e52.columns = ['a','b','c']




u=52

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e52.iloc[k,0] = dfp.iloc[i,0]
        e52.iloc[k,1] = dfp.iloc[i,1]
        e52.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e52.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e52.iloc[i,0]),4] == df200.iloc[int(e52.iloc[i,0]),4]:
        e52.iloc[i,2]= e52.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e52.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e52.iloc[i,0]),10] < 1 and  df200.iloc[int(e52.iloc[i,0]),10] <1) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e52.iloc[i,0]),10] < 2 and  df200.iloc[int(e52.iloc[i,0]),10] <2) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e52.iloc[i,0]),10] < 3 and  df200.iloc[int(e52.iloc[i,0]),10] <3) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e52.iloc[i,0]),10] < 4 and  df200.iloc[int(e52.iloc[i,0]),10] <4) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e52.iloc[i,0]),10] < 6 and  df200.iloc[int(e52.iloc[i,0]),10] <6) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e52.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e52.iloc[i,0]),5] < 700000 and  df200.iloc[int(e52.iloc[i,0]),5] < 700000) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e52.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e52.iloc[i,0]),5] < 1400000) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e52.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e52.iloc[i,0]),5] < 2100000) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e52.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e52.iloc[i,0]),5] < 2800000) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e52.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e52.iloc[i,0]),5] < 3500000) :
        e52.iloc[i,2]= e52.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e53 = pd.DataFrame(np.zeros((773, 3)))
e53.columns = ['a','b','c']




u=53

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e53.iloc[k,0] = dfp.iloc[i,0]
        e53.iloc[k,1] = dfp.iloc[i,1]
        e53.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e53.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e53.iloc[i,0]),4] == df200.iloc[int(e53.iloc[i,0]),4]:
        e53.iloc[i,2]= e53.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e53.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e53.iloc[i,0]),10] < 1 and  df200.iloc[int(e53.iloc[i,0]),10] <1) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e53.iloc[i,0]),10] < 2 and  df200.iloc[int(e53.iloc[i,0]),10] <2) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e53.iloc[i,0]),10] < 3 and  df200.iloc[int(e53.iloc[i,0]),10] <3) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e53.iloc[i,0]),10] < 4 and  df200.iloc[int(e53.iloc[i,0]),10] <4) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e53.iloc[i,0]),10] < 6 and  df200.iloc[int(e53.iloc[i,0]),10] <6) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e53.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e53.iloc[i,0]),5] < 700000 and  df200.iloc[int(e53.iloc[i,0]),5] < 700000) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e53.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e53.iloc[i,0]),5] < 1400000) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e53.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e53.iloc[i,0]),5] < 2100000) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e53.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e53.iloc[i,0]),5] < 2800000) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e53.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e53.iloc[i,0]),5] < 3500000) :
        e53.iloc[i,2]= e53.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e54 = pd.DataFrame(np.zeros((773, 3)))
e54.columns = ['a','b','c']




u=54

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e54.iloc[k,0] = dfp.iloc[i,0]
        e54.iloc[k,1] = dfp.iloc[i,1]
        e54.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e54.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e54.iloc[i,0]),4] == df200.iloc[int(e54.iloc[i,0]),4]:
        e54.iloc[i,2]= e54.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e54.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e54.iloc[i,0]),10] < 1 and  df200.iloc[int(e54.iloc[i,0]),10] <1) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e54.iloc[i,0]),10] < 2 and  df200.iloc[int(e54.iloc[i,0]),10] <2) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e54.iloc[i,0]),10] < 3 and  df200.iloc[int(e54.iloc[i,0]),10] <3) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e54.iloc[i,0]),10] < 4 and  df200.iloc[int(e54.iloc[i,0]),10] <4) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e54.iloc[i,0]),10] < 6 and  df200.iloc[int(e54.iloc[i,0]),10] <6) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e54.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e54.iloc[i,0]),5] < 700000 and  df200.iloc[int(e54.iloc[i,0]),5] < 700000) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e54.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e54.iloc[i,0]),5] < 1400000) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e54.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e54.iloc[i,0]),5] < 2100000) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e54.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e54.iloc[i,0]),5] < 2800000) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e54.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e54.iloc[i,0]),5] < 3500000) :
        e54.iloc[i,2]= e54.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e55 = pd.DataFrame(np.zeros((773, 3)))
e55.columns = ['a','b','c']




u=55

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e55.iloc[k,0] = dfp.iloc[i,0]
        e55.iloc[k,1] = dfp.iloc[i,1]
        e55.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e55.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e55.iloc[i,0]),4] == df200.iloc[int(e55.iloc[i,0]),4]:
        e55.iloc[i,2]= e55.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e55.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e55.iloc[i,0]),10] < 1 and  df200.iloc[int(e55.iloc[i,0]),10] <1) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e55.iloc[i,0]),10] < 2 and  df200.iloc[int(e55.iloc[i,0]),10] <2) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e55.iloc[i,0]),10] < 3 and  df200.iloc[int(e55.iloc[i,0]),10] <3) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e55.iloc[i,0]),10] < 4 and  df200.iloc[int(e55.iloc[i,0]),10] <4) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e55.iloc[i,0]),10] < 6 and  df200.iloc[int(e55.iloc[i,0]),10] <6) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e55.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e55.iloc[i,0]),5] < 700000 and  df200.iloc[int(e55.iloc[i,0]),5] < 700000) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e55.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e55.iloc[i,0]),5] < 1400000) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e55.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e55.iloc[i,0]),5] < 2100000) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e55.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e55.iloc[i,0]),5] < 2800000) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e55.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e55.iloc[i,0]),5] < 3500000) :
        e55.iloc[i,2]= e55.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e56 = pd.DataFrame(np.zeros((773, 3)))
e56.columns = ['a','b','c']




u=56

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e56.iloc[k,0] = dfp.iloc[i,0]
        e56.iloc[k,1] = dfp.iloc[i,1]
        e56.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e56.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e56.iloc[i,0]),4] == df200.iloc[int(e56.iloc[i,0]),4]:
        e56.iloc[i,2]= e56.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e56.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e56.iloc[i,0]),10] < 1 and  df200.iloc[int(e56.iloc[i,0]),10] <1) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e56.iloc[i,0]),10] < 2 and  df200.iloc[int(e56.iloc[i,0]),10] <2) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e56.iloc[i,0]),10] < 3 and  df200.iloc[int(e56.iloc[i,0]),10] <3) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e56.iloc[i,0]),10] < 4 and  df200.iloc[int(e56.iloc[i,0]),10] <4) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e56.iloc[i,0]),10] < 6 and  df200.iloc[int(e56.iloc[i,0]),10] <6) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e56.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e56.iloc[i,0]),5] < 700000 and  df200.iloc[int(e56.iloc[i,0]),5] < 700000) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e56.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e56.iloc[i,0]),5] < 1400000) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e56.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e56.iloc[i,0]),5] < 2100000) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e56.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e56.iloc[i,0]),5] < 2800000) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e56.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e56.iloc[i,0]),5] < 3500000) :
        e56.iloc[i,2]= e56.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e57 = pd.DataFrame(np.zeros((773, 3)))
e57.columns = ['a','b','c']




u=57

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e57.iloc[k,0] = dfp.iloc[i,0]
        e57.iloc[k,1] = dfp.iloc[i,1]
        e57.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e57.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e57.iloc[i,0]),4] == df200.iloc[int(e57.iloc[i,0]),4]:
        e57.iloc[i,2]= e57.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e57.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e57.iloc[i,0]),10] < 1 and  df200.iloc[int(e57.iloc[i,0]),10] <1) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e57.iloc[i,0]),10] < 2 and  df200.iloc[int(e57.iloc[i,0]),10] <2) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e57.iloc[i,0]),10] < 3 and  df200.iloc[int(e57.iloc[i,0]),10] <3) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e57.iloc[i,0]),10] < 4 and  df200.iloc[int(e57.iloc[i,0]),10] <4) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e57.iloc[i,0]),10] < 6 and  df200.iloc[int(e57.iloc[i,0]),10] <6) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e57.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e57.iloc[i,0]),5] < 700000 and  df200.iloc[int(e57.iloc[i,0]),5] < 700000) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e57.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e57.iloc[i,0]),5] < 1400000) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e57.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e57.iloc[i,0]),5] < 2100000) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e57.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e57.iloc[i,0]),5] < 2800000) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e57.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e57.iloc[i,0]),5] < 3500000) :
        e57.iloc[i,2]= e57.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e58 = pd.DataFrame(np.zeros((773, 3)))
e58.columns = ['a','b','c']




u=58

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e58.iloc[k,0] = dfp.iloc[i,0]
        e58.iloc[k,1] = dfp.iloc[i,1]
        e58.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e58.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e58.iloc[i,0]),4] == df200.iloc[int(e58.iloc[i,0]),4]:
        e58.iloc[i,2]= e58.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e58.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e58.iloc[i,0]),10] < 1 and  df200.iloc[int(e58.iloc[i,0]),10] <1) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e58.iloc[i,0]),10] < 2 and  df200.iloc[int(e58.iloc[i,0]),10] <2) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e58.iloc[i,0]),10] < 3 and  df200.iloc[int(e58.iloc[i,0]),10] <3) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e58.iloc[i,0]),10] < 4 and  df200.iloc[int(e58.iloc[i,0]),10] <4) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e58.iloc[i,0]),10] < 6 and  df200.iloc[int(e58.iloc[i,0]),10] <6) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e58.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e58.iloc[i,0]),5] < 700000 and  df200.iloc[int(e58.iloc[i,0]),5] < 700000) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e58.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e58.iloc[i,0]),5] < 1400000) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e58.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e58.iloc[i,0]),5] < 2100000) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e58.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e58.iloc[i,0]),5] < 2800000) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e58.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e58.iloc[i,0]),5] < 3500000) :
        e58.iloc[i,2]= e58.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e59 = pd.DataFrame(np.zeros((773, 3)))
e59.columns = ['a','b','c']




u=59

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e59.iloc[k,0] = dfp.iloc[i,0]
        e59.iloc[k,1] = dfp.iloc[i,1]
        e59.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e59.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e59.iloc[i,0]),4] == df200.iloc[int(e59.iloc[i,0]),4]:
        e59.iloc[i,2]= e59.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e59.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e59.iloc[i,0]),10] < 1 and  df200.iloc[int(e59.iloc[i,0]),10] <1) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e59.iloc[i,0]),10] < 2 and  df200.iloc[int(e59.iloc[i,0]),10] <2) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e59.iloc[i,0]),10] < 3 and  df200.iloc[int(e59.iloc[i,0]),10] <3) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e59.iloc[i,0]),10] < 4 and  df200.iloc[int(e59.iloc[i,0]),10] <4) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e59.iloc[i,0]),10] < 6 and  df200.iloc[int(e59.iloc[i,0]),10] <6) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e59.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e59.iloc[i,0]),5] < 700000 and  df200.iloc[int(e59.iloc[i,0]),5] < 700000) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e59.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e59.iloc[i,0]),5] < 1400000) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e59.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e59.iloc[i,0]),5] < 2100000) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e59.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e59.iloc[i,0]),5] < 2800000) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e59.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e59.iloc[i,0]),5] < 3500000) :
        e59.iloc[i,2]= e59.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e60 = pd.DataFrame(np.zeros((773, 3)))
e60.columns = ['a','b','c']




u=60

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e60.iloc[k,0] = dfp.iloc[i,0]
        e60.iloc[k,1] = dfp.iloc[i,1]
        e60.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e60.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e60.iloc[i,0]),4] == df200.iloc[int(e60.iloc[i,0]),4]:
        e60.iloc[i,2]= e60.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e60.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e60.iloc[i,0]),10] < 1 and  df200.iloc[int(e60.iloc[i,0]),10] <1) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e60.iloc[i,0]),10] < 2 and  df200.iloc[int(e60.iloc[i,0]),10] <2) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e60.iloc[i,0]),10] < 3 and  df200.iloc[int(e60.iloc[i,0]),10] <3) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e60.iloc[i,0]),10] < 4 and  df200.iloc[int(e60.iloc[i,0]),10] <4) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e60.iloc[i,0]),10] < 6 and  df200.iloc[int(e60.iloc[i,0]),10] <6) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e60.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e60.iloc[i,0]),5] < 700000 and  df200.iloc[int(e60.iloc[i,0]),5] < 700000) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e60.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e60.iloc[i,0]),5] < 1400000) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e60.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e60.iloc[i,0]),5] < 2100000) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e60.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e60.iloc[i,0]),5] < 2800000) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e60.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e60.iloc[i,0]),5] < 3500000) :
        e60.iloc[i,2]= e60.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e61 = pd.DataFrame(np.zeros((773, 3)))
e61.columns = ['a','b','c']




u=61

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e61.iloc[k,0] = dfp.iloc[i,0]
        e61.iloc[k,1] = dfp.iloc[i,1]
        e61.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e61.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e61.iloc[i,0]),4] == df200.iloc[int(e61.iloc[i,0]),4]:
        e61.iloc[i,2]= e61.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e61.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e61.iloc[i,0]),10] < 1 and  df200.iloc[int(e61.iloc[i,0]),10] <1) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e61.iloc[i,0]),10] < 2 and  df200.iloc[int(e61.iloc[i,0]),10] <2) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e61.iloc[i,0]),10] < 3 and  df200.iloc[int(e61.iloc[i,0]),10] <3) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e61.iloc[i,0]),10] < 4 and  df200.iloc[int(e61.iloc[i,0]),10] <4) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e61.iloc[i,0]),10] < 6 and  df200.iloc[int(e61.iloc[i,0]),10] <6) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e61.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e61.iloc[i,0]),5] < 700000 and  df200.iloc[int(e61.iloc[i,0]),5] < 700000) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e61.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e61.iloc[i,0]),5] < 1400000) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e61.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e61.iloc[i,0]),5] < 2100000) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e61.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e61.iloc[i,0]),5] < 2800000) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e61.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e61.iloc[i,0]),5] < 3500000) :
        e61.iloc[i,2]= e61.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e62 = pd.DataFrame(np.zeros((773, 3)))
e62.columns = ['a','b','c']




u=62

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e62.iloc[k,0] = dfp.iloc[i,0]
        e62.iloc[k,1] = dfp.iloc[i,1]
        e62.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e62.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e62.iloc[i,0]),4] == df200.iloc[int(e62.iloc[i,0]),4]:
        e62.iloc[i,2]= e62.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e62.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e62.iloc[i,0]),10] < 1 and  df200.iloc[int(e62.iloc[i,0]),10] <1) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e62.iloc[i,0]),10] < 2 and  df200.iloc[int(e62.iloc[i,0]),10] <2) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e62.iloc[i,0]),10] < 3 and  df200.iloc[int(e62.iloc[i,0]),10] <3) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e62.iloc[i,0]),10] < 4 and  df200.iloc[int(e62.iloc[i,0]),10] <4) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e62.iloc[i,0]),10] < 6 and  df200.iloc[int(e62.iloc[i,0]),10] <6) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e62.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e62.iloc[i,0]),5] < 700000 and  df200.iloc[int(e62.iloc[i,0]),5] < 700000) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e62.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e62.iloc[i,0]),5] < 1400000) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e62.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e62.iloc[i,0]),5] < 2100000) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e62.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e62.iloc[i,0]),5] < 2800000) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e62.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e62.iloc[i,0]),5] < 3500000) :
        e62.iloc[i,2]= e62.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e63 = pd.DataFrame(np.zeros((773, 3)))
e63.columns = ['a','b','c']




u=63

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e63.iloc[k,0] = dfp.iloc[i,0]
        e63.iloc[k,1] = dfp.iloc[i,1]
        e63.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e63.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e63.iloc[i,0]),4] == df200.iloc[int(e63.iloc[i,0]),4]:
        e63.iloc[i,2]= e63.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e63.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e63.iloc[i,0]),10] < 1 and  df200.iloc[int(e63.iloc[i,0]),10] <1) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e63.iloc[i,0]),10] < 2 and  df200.iloc[int(e63.iloc[i,0]),10] <2) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e63.iloc[i,0]),10] < 3 and  df200.iloc[int(e63.iloc[i,0]),10] <3) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e63.iloc[i,0]),10] < 4 and  df200.iloc[int(e63.iloc[i,0]),10] <4) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e63.iloc[i,0]),10] < 6 and  df200.iloc[int(e63.iloc[i,0]),10] <6) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e63.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e63.iloc[i,0]),5] < 700000 and  df200.iloc[int(e63.iloc[i,0]),5] < 700000) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e63.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e63.iloc[i,0]),5] < 1400000) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e63.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e63.iloc[i,0]),5] < 2100000) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e63.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e63.iloc[i,0]),5] < 2800000) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e63.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e63.iloc[i,0]),5] < 3500000) :
        e63.iloc[i,2]= e63.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e64 = pd.DataFrame(np.zeros((773, 3)))
e64.columns = ['a','b','c']




u=64

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e64.iloc[k,0] = dfp.iloc[i,0]
        e64.iloc[k,1] = dfp.iloc[i,1]
        e64.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e64.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e64.iloc[i,0]),4] == df200.iloc[int(e64.iloc[i,0]),4]:
        e64.iloc[i,2]= e64.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e64.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e64.iloc[i,0]),10] < 1 and  df200.iloc[int(e64.iloc[i,0]),10] <1) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e64.iloc[i,0]),10] < 2 and  df200.iloc[int(e64.iloc[i,0]),10] <2) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e64.iloc[i,0]),10] < 3 and  df200.iloc[int(e64.iloc[i,0]),10] <3) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e64.iloc[i,0]),10] < 4 and  df200.iloc[int(e64.iloc[i,0]),10] <4) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e64.iloc[i,0]),10] < 6 and  df200.iloc[int(e64.iloc[i,0]),10] <6) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e64.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e64.iloc[i,0]),5] < 700000 and  df200.iloc[int(e64.iloc[i,0]),5] < 700000) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e64.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e64.iloc[i,0]),5] < 1400000) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e64.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e64.iloc[i,0]),5] < 2100000) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e64.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e64.iloc[i,0]),5] < 2800000) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e64.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e64.iloc[i,0]),5] < 3500000) :
        e64.iloc[i,2]= e64.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e65 = pd.DataFrame(np.zeros((773, 3)))
e65.columns = ['a','b','c']




u=65

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e65.iloc[k,0] = dfp.iloc[i,0]
        e65.iloc[k,1] = dfp.iloc[i,1]
        e65.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e65.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e65.iloc[i,0]),4] == df200.iloc[int(e65.iloc[i,0]),4]:
        e65.iloc[i,2]= e65.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e65.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e65.iloc[i,0]),10] < 1 and  df200.iloc[int(e65.iloc[i,0]),10] <1) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e65.iloc[i,0]),10] < 2 and  df200.iloc[int(e65.iloc[i,0]),10] <2) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e65.iloc[i,0]),10] < 3 and  df200.iloc[int(e65.iloc[i,0]),10] <3) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e65.iloc[i,0]),10] < 4 and  df200.iloc[int(e65.iloc[i,0]),10] <4) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e65.iloc[i,0]),10] < 6 and  df200.iloc[int(e65.iloc[i,0]),10] <6) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e65.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e65.iloc[i,0]),5] < 700000 and  df200.iloc[int(e65.iloc[i,0]),5] < 700000) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e65.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e65.iloc[i,0]),5] < 1400000) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e65.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e65.iloc[i,0]),5] < 2100000) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e65.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e65.iloc[i,0]),5] < 2800000) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e65.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e65.iloc[i,0]),5] < 3500000) :
        e65.iloc[i,2]= e65.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e66 = pd.DataFrame(np.zeros((773, 3)))
e66.columns = ['a','b','c']




u=66

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e66.iloc[k,0] = dfp.iloc[i,0]
        e66.iloc[k,1] = dfp.iloc[i,1]
        e66.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e66.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e66.iloc[i,0]),4] == df200.iloc[int(e66.iloc[i,0]),4]:
        e66.iloc[i,2]= e66.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e66.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e66.iloc[i,0]),10] < 1 and  df200.iloc[int(e66.iloc[i,0]),10] <1) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e66.iloc[i,0]),10] < 2 and  df200.iloc[int(e66.iloc[i,0]),10] <2) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e66.iloc[i,0]),10] < 3 and  df200.iloc[int(e66.iloc[i,0]),10] <3) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e66.iloc[i,0]),10] < 4 and  df200.iloc[int(e66.iloc[i,0]),10] <4) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e66.iloc[i,0]),10] < 6 and  df200.iloc[int(e66.iloc[i,0]),10] <6) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e66.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e66.iloc[i,0]),5] < 700000 and  df200.iloc[int(e66.iloc[i,0]),5] < 700000) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e66.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e66.iloc[i,0]),5] < 1400000) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e66.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e66.iloc[i,0]),5] < 2100000) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e66.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e66.iloc[i,0]),5] < 2800000) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e66.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e66.iloc[i,0]),5] < 3500000) :
        e66.iloc[i,2]= e66.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e67 = pd.DataFrame(np.zeros((773, 3)))
e67.columns = ['a','b','c']




u=67

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e67.iloc[k,0] = dfp.iloc[i,0]
        e67.iloc[k,1] = dfp.iloc[i,1]
        e67.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e67.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e67.iloc[i,0]),4] == df200.iloc[int(e67.iloc[i,0]),4]:
        e67.iloc[i,2]= e67.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e67.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e67.iloc[i,0]),10] < 1 and  df200.iloc[int(e67.iloc[i,0]),10] <1) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e67.iloc[i,0]),10] < 2 and  df200.iloc[int(e67.iloc[i,0]),10] <2) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e67.iloc[i,0]),10] < 3 and  df200.iloc[int(e67.iloc[i,0]),10] <3) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e67.iloc[i,0]),10] < 4 and  df200.iloc[int(e67.iloc[i,0]),10] <4) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e67.iloc[i,0]),10] < 6 and  df200.iloc[int(e67.iloc[i,0]),10] <6) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e67.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e67.iloc[i,0]),5] < 700000 and  df200.iloc[int(e67.iloc[i,0]),5] < 700000) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e67.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e67.iloc[i,0]),5] < 1400000) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e67.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e67.iloc[i,0]),5] < 2100000) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e67.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e67.iloc[i,0]),5] < 2800000) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e67.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e67.iloc[i,0]),5] < 3500000) :
        e67.iloc[i,2]= e67.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e68 = pd.DataFrame(np.zeros((773, 3)))
e68.columns = ['a','b','c']




u=68

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e68.iloc[k,0] = dfp.iloc[i,0]
        e68.iloc[k,1] = dfp.iloc[i,1]
        e68.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e68.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e68.iloc[i,0]),4] == df200.iloc[int(e68.iloc[i,0]),4]:
        e68.iloc[i,2]= e68.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e68.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e68.iloc[i,0]),10] < 1 and  df200.iloc[int(e68.iloc[i,0]),10] <1) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e68.iloc[i,0]),10] < 2 and  df200.iloc[int(e68.iloc[i,0]),10] <2) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e68.iloc[i,0]),10] < 3 and  df200.iloc[int(e68.iloc[i,0]),10] <3) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e68.iloc[i,0]),10] < 4 and  df200.iloc[int(e68.iloc[i,0]),10] <4) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e68.iloc[i,0]),10] < 6 and  df200.iloc[int(e68.iloc[i,0]),10] <6) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e68.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e68.iloc[i,0]),5] < 700000 and  df200.iloc[int(e68.iloc[i,0]),5] < 700000) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e68.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e68.iloc[i,0]),5] < 1400000) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e68.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e68.iloc[i,0]),5] < 2100000) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e68.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e68.iloc[i,0]),5] < 2800000) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e68.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e68.iloc[i,0]),5] < 3500000) :
        e68.iloc[i,2]= e68.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e69 = pd.DataFrame(np.zeros((773, 3)))
e69.columns = ['a','b','c']




u=69

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e69.iloc[k,0] = dfp.iloc[i,0]
        e69.iloc[k,1] = dfp.iloc[i,1]
        e69.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e69.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e69.iloc[i,0]),4] == df200.iloc[int(e69.iloc[i,0]),4]:
        e69.iloc[i,2]= e69.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e69.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e69.iloc[i,0]),10] < 1 and  df200.iloc[int(e69.iloc[i,0]),10] <1) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e69.iloc[i,0]),10] < 2 and  df200.iloc[int(e69.iloc[i,0]),10] <2) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e69.iloc[i,0]),10] < 3 and  df200.iloc[int(e69.iloc[i,0]),10] <3) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e69.iloc[i,0]),10] < 4 and  df200.iloc[int(e69.iloc[i,0]),10] <4) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e69.iloc[i,0]),10] < 6 and  df200.iloc[int(e69.iloc[i,0]),10] <6) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e69.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e69.iloc[i,0]),5] < 700000 and  df200.iloc[int(e69.iloc[i,0]),5] < 700000) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e69.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e69.iloc[i,0]),5] < 1400000) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e69.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e69.iloc[i,0]),5] < 2100000) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e69.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e69.iloc[i,0]),5] < 2800000) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e69.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e69.iloc[i,0]),5] < 3500000) :
        e69.iloc[i,2]= e69.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e70 = pd.DataFrame(np.zeros((773, 3)))
e70.columns = ['a','b','c']




u=70

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e70.iloc[k,0] = dfp.iloc[i,0]
        e70.iloc[k,1] = dfp.iloc[i,1]
        e70.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e70.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e70.iloc[i,0]),4] == df200.iloc[int(e70.iloc[i,0]),4]:
        e70.iloc[i,2]= e70.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e70.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e70.iloc[i,0]),10] < 1 and  df200.iloc[int(e70.iloc[i,0]),10] <1) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e70.iloc[i,0]),10] < 2 and  df200.iloc[int(e70.iloc[i,0]),10] <2) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e70.iloc[i,0]),10] < 3 and  df200.iloc[int(e70.iloc[i,0]),10] <3) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e70.iloc[i,0]),10] < 4 and  df200.iloc[int(e70.iloc[i,0]),10] <4) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e70.iloc[i,0]),10] < 6 and  df200.iloc[int(e70.iloc[i,0]),10] <6) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e70.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e70.iloc[i,0]),5] < 700000 and  df200.iloc[int(e70.iloc[i,0]),5] < 700000) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e70.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e70.iloc[i,0]),5] < 1400000) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e70.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e70.iloc[i,0]),5] < 2100000) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e70.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e70.iloc[i,0]),5] < 2800000) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e70.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e70.iloc[i,0]),5] < 3500000) :
        e70.iloc[i,2]= e70.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e71 = pd.DataFrame(np.zeros((773, 3)))
e71.columns = ['a','b','c']




u=71

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e71.iloc[k,0] = dfp.iloc[i,0]
        e71.iloc[k,1] = dfp.iloc[i,1]
        e71.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e71.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e71.iloc[i,0]),4] == df200.iloc[int(e71.iloc[i,0]),4]:
        e71.iloc[i,2]= e71.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e71.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e71.iloc[i,0]),10] < 1 and  df200.iloc[int(e71.iloc[i,0]),10] <1) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e71.iloc[i,0]),10] < 2 and  df200.iloc[int(e71.iloc[i,0]),10] <2) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e71.iloc[i,0]),10] < 3 and  df200.iloc[int(e71.iloc[i,0]),10] <3) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e71.iloc[i,0]),10] < 4 and  df200.iloc[int(e71.iloc[i,0]),10] <4) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e71.iloc[i,0]),10] < 6 and  df200.iloc[int(e71.iloc[i,0]),10] <6) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e71.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e71.iloc[i,0]),5] < 700000 and  df200.iloc[int(e71.iloc[i,0]),5] < 700000) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e71.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e71.iloc[i,0]),5] < 1400000) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e71.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e71.iloc[i,0]),5] < 2100000) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e71.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e71.iloc[i,0]),5] < 2800000) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e71.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e71.iloc[i,0]),5] < 3500000) :
        e71.iloc[i,2]= e71.iloc[i,2]+(.33*(.2))
    
    
    
e72 = pd.DataFrame(np.zeros((773, 3)))
e72.columns = ['a','b','c']




u=72

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e72.iloc[k,0] = dfp.iloc[i,0]
        e72.iloc[k,1] = dfp.iloc[i,1]
        e72.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e72.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e72.iloc[i,0]),4] == df200.iloc[int(e72.iloc[i,0]),4]:
        e72.iloc[i,2]= e72.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e72.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e72.iloc[i,0]),10] < 1 and  df200.iloc[int(e72.iloc[i,0]),10] <1) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e72.iloc[i,0]),10] < 2 and  df200.iloc[int(e72.iloc[i,0]),10] <2) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e72.iloc[i,0]),10] < 3 and  df200.iloc[int(e72.iloc[i,0]),10] <3) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e72.iloc[i,0]),10] < 4 and  df200.iloc[int(e72.iloc[i,0]),10] <4) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e72.iloc[i,0]),10] < 6 and  df200.iloc[int(e72.iloc[i,0]),10] <6) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e72.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e72.iloc[i,0]),5] < 700000 and  df200.iloc[int(e72.iloc[i,0]),5] < 700000) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e72.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e72.iloc[i,0]),5] < 1400000) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e72.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e72.iloc[i,0]),5] < 2100000) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e72.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e72.iloc[i,0]),5] < 2800000) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e72.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e72.iloc[i,0]),5] < 3500000) :
        e72.iloc[i,2]= e72.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e73 = pd.DataFrame(np.zeros((773, 3)))
e73.columns = ['a','b','c']




u=73

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e73.iloc[k,0] = dfp.iloc[i,0]
        e73.iloc[k,1] = dfp.iloc[i,1]
        e73.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e73.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e73.iloc[i,0]),4] == df200.iloc[int(e73.iloc[i,0]),4]:
        e73.iloc[i,2]= e73.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e73.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e73.iloc[i,0]),10] < 1 and  df200.iloc[int(e73.iloc[i,0]),10] <1) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e73.iloc[i,0]),10] < 2 and  df200.iloc[int(e73.iloc[i,0]),10] <2) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e73.iloc[i,0]),10] < 3 and  df200.iloc[int(e73.iloc[i,0]),10] <3) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e73.iloc[i,0]),10] < 4 and  df200.iloc[int(e73.iloc[i,0]),10] <4) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e73.iloc[i,0]),10] < 6 and  df200.iloc[int(e73.iloc[i,0]),10] <6) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e73.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e73.iloc[i,0]),5] < 700000 and  df200.iloc[int(e73.iloc[i,0]),5] < 700000) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e73.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e73.iloc[i,0]),5] < 1400000) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e73.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e73.iloc[i,0]),5] < 2100000) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e73.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e73.iloc[i,0]),5] < 2800000) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e73.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e73.iloc[i,0]),5] < 3500000) :
        e73.iloc[i,2]= e73.iloc[i,2]+(.33*(.2))
    
    
    
e74 = pd.DataFrame(np.zeros((773, 3)))
e74.columns = ['a','b','c']




u=74

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e74.iloc[k,0] = dfp.iloc[i,0]
        e74.iloc[k,1] = dfp.iloc[i,1]
        e74.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e74.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e74.iloc[i,0]),4] == df200.iloc[int(e74.iloc[i,0]),4]:
        e74.iloc[i,2]= e74.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e74.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e74.iloc[i,0]),10] < 1 and  df200.iloc[int(e74.iloc[i,0]),10] <1) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e74.iloc[i,0]),10] < 2 and  df200.iloc[int(e74.iloc[i,0]),10] <2) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e74.iloc[i,0]),10] < 3 and  df200.iloc[int(e74.iloc[i,0]),10] <3) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e74.iloc[i,0]),10] < 4 and  df200.iloc[int(e74.iloc[i,0]),10] <4) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e74.iloc[i,0]),10] < 6 and  df200.iloc[int(e74.iloc[i,0]),10] <6) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e74.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e74.iloc[i,0]),5] < 700000 and  df200.iloc[int(e74.iloc[i,0]),5] < 700000) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e74.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e74.iloc[i,0]),5] < 1400000) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e74.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e74.iloc[i,0]),5] < 2100000) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e74.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e74.iloc[i,0]),5] < 2800000) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e74.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e74.iloc[i,0]),5] < 3500000) :
        e74.iloc[i,2]= e74.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e75 = pd.DataFrame(np.zeros((773, 3)))
e75.columns = ['a','b','c']




u=75

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e75.iloc[k,0] = dfp.iloc[i,0]
        e75.iloc[k,1] = dfp.iloc[i,1]
        e75.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e75.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e75.iloc[i,0]),4] == df200.iloc[int(e75.iloc[i,0]),4]:
        e75.iloc[i,2]= e75.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e75.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e75.iloc[i,0]),10] < 1 and  df200.iloc[int(e75.iloc[i,0]),10] <1) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e75.iloc[i,0]),10] < 2 and  df200.iloc[int(e75.iloc[i,0]),10] <2) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e75.iloc[i,0]),10] < 3 and  df200.iloc[int(e75.iloc[i,0]),10] <3) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e75.iloc[i,0]),10] < 4 and  df200.iloc[int(e75.iloc[i,0]),10] <4) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e75.iloc[i,0]),10] < 6 and  df200.iloc[int(e75.iloc[i,0]),10] <6) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e75.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e75.iloc[i,0]),5] < 700000 and  df200.iloc[int(e75.iloc[i,0]),5] < 700000) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e75.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e75.iloc[i,0]),5] < 1400000) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e75.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e75.iloc[i,0]),5] < 2100000) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e75.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e75.iloc[i,0]),5] < 2800000) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e75.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e75.iloc[i,0]),5] < 3500000) :
        e75.iloc[i,2]= e75.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e76 = pd.DataFrame(np.zeros((773, 3)))
e76.columns = ['a','b','c']




u=76

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e76.iloc[k,0] = dfp.iloc[i,0]
        e76.iloc[k,1] = dfp.iloc[i,1]
        e76.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e76.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e76.iloc[i,0]),4] == df200.iloc[int(e76.iloc[i,0]),4]:
        e76.iloc[i,2]= e76.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e76.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e76.iloc[i,0]),10] < 1 and  df200.iloc[int(e76.iloc[i,0]),10] <1) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e76.iloc[i,0]),10] < 2 and  df200.iloc[int(e76.iloc[i,0]),10] <2) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e76.iloc[i,0]),10] < 3 and  df200.iloc[int(e76.iloc[i,0]),10] <3) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e76.iloc[i,0]),10] < 4 and  df200.iloc[int(e76.iloc[i,0]),10] <4) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e76.iloc[i,0]),10] < 6 and  df200.iloc[int(e76.iloc[i,0]),10] <6) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e76.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e76.iloc[i,0]),5] < 700000 and  df200.iloc[int(e76.iloc[i,0]),5] < 700000) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e76.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e76.iloc[i,0]),5] < 1400000) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e76.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e76.iloc[i,0]),5] < 2100000) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e76.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e76.iloc[i,0]),5] < 2800000) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e76.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e76.iloc[i,0]),5] < 3500000) :
        e76.iloc[i,2]= e76.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e77 = pd.DataFrame(np.zeros((773, 3)))
e77.columns = ['a','b','c']




u=77

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e77.iloc[k,0] = dfp.iloc[i,0]
        e77.iloc[k,1] = dfp.iloc[i,1]
        e77.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e77.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e77.iloc[i,0]),4] == df200.iloc[int(e77.iloc[i,0]),4]:
        e77.iloc[i,2]= e77.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e77.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e77.iloc[i,0]),10] < 1 and  df200.iloc[int(e77.iloc[i,0]),10] <1) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e77.iloc[i,0]),10] < 2 and  df200.iloc[int(e77.iloc[i,0]),10] <2) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e77.iloc[i,0]),10] < 3 and  df200.iloc[int(e77.iloc[i,0]),10] <3) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e77.iloc[i,0]),10] < 4 and  df200.iloc[int(e77.iloc[i,0]),10] <4) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e77.iloc[i,0]),10] < 6 and  df200.iloc[int(e77.iloc[i,0]),10] <6) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e77.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e77.iloc[i,0]),5] < 700000 and  df200.iloc[int(e77.iloc[i,0]),5] < 700000) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e77.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e77.iloc[i,0]),5] < 1400000) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e77.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e77.iloc[i,0]),5] < 2100000) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e77.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e77.iloc[i,0]),5] < 2800000) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e77.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e77.iloc[i,0]),5] < 3500000) :
        e77.iloc[i,2]= e77.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e78 = pd.DataFrame(np.zeros((773, 3)))
e78.columns = ['a','b','c']




u=78

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e78.iloc[k,0] = dfp.iloc[i,0]
        e78.iloc[k,1] = dfp.iloc[i,1]
        e78.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e78.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e78.iloc[i,0]),4] == df200.iloc[int(e78.iloc[i,0]),4]:
        e78.iloc[i,2]= e78.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e78.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e78.iloc[i,0]),10] < 1 and  df200.iloc[int(e78.iloc[i,0]),10] <1) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e78.iloc[i,0]),10] < 2 and  df200.iloc[int(e78.iloc[i,0]),10] <2) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e78.iloc[i,0]),10] < 3 and  df200.iloc[int(e78.iloc[i,0]),10] <3) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e78.iloc[i,0]),10] < 4 and  df200.iloc[int(e78.iloc[i,0]),10] <4) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e78.iloc[i,0]),10] < 6 and  df200.iloc[int(e78.iloc[i,0]),10] <6) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e78.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e78.iloc[i,0]),5] < 700000 and  df200.iloc[int(e78.iloc[i,0]),5] < 700000) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e78.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e78.iloc[i,0]),5] < 1400000) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e78.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e78.iloc[i,0]),5] < 2100000) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e78.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e78.iloc[i,0]),5] < 2800000) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e78.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e78.iloc[i,0]),5] < 3500000) :
        e78.iloc[i,2]= e78.iloc[i,2]+(.33*(.2))
    
    
    
    
e79 = pd.DataFrame(np.zeros((773, 3)))
e79.columns = ['a','b','c']




u=79

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e79.iloc[k,0] = dfp.iloc[i,0]
        e79.iloc[k,1] = dfp.iloc[i,1]
        e79.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e79.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e79.iloc[i,0]),4] == df200.iloc[int(e79.iloc[i,0]),4]:
        e79.iloc[i,2]= e79.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e79.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e79.iloc[i,0]),10] < 1 and  df200.iloc[int(e79.iloc[i,0]),10] <1) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e79.iloc[i,0]),10] < 2 and  df200.iloc[int(e79.iloc[i,0]),10] <2) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e79.iloc[i,0]),10] < 3 and  df200.iloc[int(e79.iloc[i,0]),10] <3) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e79.iloc[i,0]),10] < 4 and  df200.iloc[int(e79.iloc[i,0]),10] <4) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e79.iloc[i,0]),10] < 6 and  df200.iloc[int(e79.iloc[i,0]),10] <6) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e79.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e79.iloc[i,0]),5] < 700000 and  df200.iloc[int(e79.iloc[i,0]),5] < 700000) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e79.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e79.iloc[i,0]),5] < 1400000) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e79.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e79.iloc[i,0]),5] < 2100000) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e79.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e79.iloc[i,0]),5] < 2800000) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e79.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e79.iloc[i,0]),5] < 3500000) :
        e79.iloc[i,2]= e79.iloc[i,2]+(.33*(.2))
    
    
    
    
e80 = pd.DataFrame(np.zeros((773, 3)))
e80.columns = ['a','b','c']




u=80

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e80.iloc[k,0] = dfp.iloc[i,0]
        e80.iloc[k,1] = dfp.iloc[i,1]
        e80.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e80.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e80.iloc[i,0]),4] == df200.iloc[int(e80.iloc[i,0]),4]:
        e80.iloc[i,2]= e80.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e80.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e80.iloc[i,0]),10] < 1 and  df200.iloc[int(e80.iloc[i,0]),10] <1) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e80.iloc[i,0]),10] < 2 and  df200.iloc[int(e80.iloc[i,0]),10] <2) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e80.iloc[i,0]),10] < 3 and  df200.iloc[int(e80.iloc[i,0]),10] <3) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e80.iloc[i,0]),10] < 4 and  df200.iloc[int(e80.iloc[i,0]),10] <4) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e80.iloc[i,0]),10] < 6 and  df200.iloc[int(e80.iloc[i,0]),10] <6) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e80.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e80.iloc[i,0]),5] < 700000 and  df200.iloc[int(e80.iloc[i,0]),5] < 700000) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e80.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e80.iloc[i,0]),5] < 1400000) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e80.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e80.iloc[i,0]),5] < 2100000) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e80.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e80.iloc[i,0]),5] < 2800000) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e80.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e80.iloc[i,0]),5] < 3500000) :
        e80.iloc[i,2]= e80.iloc[i,2]+(.33*(.2))
    
    
    
    
e81 = pd.DataFrame(np.zeros((773, 3)))
e81.columns = ['a','b','c']




u=81

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e81.iloc[k,0] = dfp.iloc[i,0]
        e81.iloc[k,1] = dfp.iloc[i,1]
        e81.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e81.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e81.iloc[i,0]),4] == df200.iloc[int(e81.iloc[i,0]),4]:
        e81.iloc[i,2]= e81.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e81.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e81.iloc[i,0]),10] < 1 and  df200.iloc[int(e81.iloc[i,0]),10] <1) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e81.iloc[i,0]),10] < 2 and  df200.iloc[int(e81.iloc[i,0]),10] <2) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e81.iloc[i,0]),10] < 3 and  df200.iloc[int(e81.iloc[i,0]),10] <3) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e81.iloc[i,0]),10] < 4 and  df200.iloc[int(e81.iloc[i,0]),10] <4) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e81.iloc[i,0]),10] < 6 and  df200.iloc[int(e81.iloc[i,0]),10] <6) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e81.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e81.iloc[i,0]),5] < 700000 and  df200.iloc[int(e81.iloc[i,0]),5] < 700000) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e81.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e81.iloc[i,0]),5] < 1400000) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e81.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e81.iloc[i,0]),5] < 2100000) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e81.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e81.iloc[i,0]),5] < 2800000) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e81.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e81.iloc[i,0]),5] < 3500000) :
        e81.iloc[i,2]= e81.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e82 = pd.DataFrame(np.zeros((773, 3)))
e82.columns = ['a','b','c']




u=82

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e82.iloc[k,0] = dfp.iloc[i,0]
        e82.iloc[k,1] = dfp.iloc[i,1]
        e82.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e82.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e82.iloc[i,0]),4] == df200.iloc[int(e82.iloc[i,0]),4]:
        e82.iloc[i,2]= e82.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e82.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e82.iloc[i,0]),10] < 1 and  df200.iloc[int(e82.iloc[i,0]),10] <1) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e82.iloc[i,0]),10] < 2 and  df200.iloc[int(e82.iloc[i,0]),10] <2) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e82.iloc[i,0]),10] < 3 and  df200.iloc[int(e82.iloc[i,0]),10] <3) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e82.iloc[i,0]),10] < 4 and  df200.iloc[int(e82.iloc[i,0]),10] <4) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e82.iloc[i,0]),10] < 6 and  df200.iloc[int(e82.iloc[i,0]),10] <6) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e82.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e82.iloc[i,0]),5] < 700000 and  df200.iloc[int(e82.iloc[i,0]),5] < 700000) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e82.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e82.iloc[i,0]),5] < 1400000) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e82.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e82.iloc[i,0]),5] < 2100000) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e82.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e82.iloc[i,0]),5] < 2800000) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e82.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e82.iloc[i,0]),5] < 3500000) :
        e82.iloc[i,2]= e82.iloc[i,2]+(.33*(.2))
    
    
    
    
e83 = pd.DataFrame(np.zeros((773, 3)))
e83.columns = ['a','b','c']




u=83

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e83.iloc[k,0] = dfp.iloc[i,0]
        e83.iloc[k,1] = dfp.iloc[i,1]
        e83.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e83.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e83.iloc[i,0]),4] == df200.iloc[int(e83.iloc[i,0]),4]:
        e83.iloc[i,2]= e83.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e83.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e83.iloc[i,0]),10] < 1 and  df200.iloc[int(e83.iloc[i,0]),10] <1) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e83.iloc[i,0]),10] < 2 and  df200.iloc[int(e83.iloc[i,0]),10] <2) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e83.iloc[i,0]),10] < 3 and  df200.iloc[int(e83.iloc[i,0]),10] <3) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e83.iloc[i,0]),10] < 4 and  df200.iloc[int(e83.iloc[i,0]),10] <4) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e83.iloc[i,0]),10] < 6 and  df200.iloc[int(e83.iloc[i,0]),10] <6) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e83.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e83.iloc[i,0]),5] < 700000 and  df200.iloc[int(e83.iloc[i,0]),5] < 700000) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e83.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e83.iloc[i,0]),5] < 1400000) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e83.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e83.iloc[i,0]),5] < 2100000) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e83.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e83.iloc[i,0]),5] < 2800000) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e83.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e83.iloc[i,0]),5] < 3500000) :
        e83.iloc[i,2]= e83.iloc[i,2]+(.33*(.2))
    
    
    
e84 = pd.DataFrame(np.zeros((773, 3)))   
e84.columns = ['a','b','c']




u=84

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e84.iloc[k,0] = dfp.iloc[i,0]
        e84.iloc[k,1] = dfp.iloc[i,1]
        e84.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e84.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e84.iloc[i,0]),4] == df200.iloc[int(e84.iloc[i,0]),4]:
        e84.iloc[i,2]= e84.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e84.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e84.iloc[i,0]),10] < 1 and  df200.iloc[int(e84.iloc[i,0]),10] <1) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e84.iloc[i,0]),10] < 2 and  df200.iloc[int(e84.iloc[i,0]),10] <2) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e84.iloc[i,0]),10] < 3 and  df200.iloc[int(e84.iloc[i,0]),10] <3) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e84.iloc[i,0]),10] < 4 and  df200.iloc[int(e84.iloc[i,0]),10] <4) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e84.iloc[i,0]),10] < 6 and  df200.iloc[int(e84.iloc[i,0]),10] <6) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e84.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e84.iloc[i,0]),5] < 700000 and  df200.iloc[int(e84.iloc[i,0]),5] < 700000) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e84.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e84.iloc[i,0]),5] < 1400000) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e84.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e84.iloc[i,0]),5] < 2100000) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e84.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e84.iloc[i,0]),5] < 2800000) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e84.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e84.iloc[i,0]),5] < 3500000) :
        e84.iloc[i,2]= e84.iloc[i,2]+(.33*(.2))
    
    
    
    
e85 = pd.DataFrame(np.zeros((773, 3)))
e85.columns = ['a','b','c']




u=85

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e85.iloc[k,0] = dfp.iloc[i,0]
        e85.iloc[k,1] = dfp.iloc[i,1]
        e85.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e85.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e85.iloc[i,0]),4] == df200.iloc[int(e85.iloc[i,0]),4]:
        e85.iloc[i,2]= e85.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e85.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e85.iloc[i,0]),10] < 1 and  df200.iloc[int(e85.iloc[i,0]),10] <1) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e85.iloc[i,0]),10] < 2 and  df200.iloc[int(e85.iloc[i,0]),10] <2) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e85.iloc[i,0]),10] < 3 and  df200.iloc[int(e85.iloc[i,0]),10] <3) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e85.iloc[i,0]),10] < 4 and  df200.iloc[int(e85.iloc[i,0]),10] <4) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e85.iloc[i,0]),10] < 6 and  df200.iloc[int(e85.iloc[i,0]),10] <6) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e85.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e85.iloc[i,0]),5] < 700000 and  df200.iloc[int(e85.iloc[i,0]),5] < 700000) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e85.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e85.iloc[i,0]),5] < 1400000) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e85.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e85.iloc[i,0]),5] < 2100000) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e85.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e85.iloc[i,0]),5] < 2800000) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e85.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e85.iloc[i,0]),5] < 3500000) :
        e85.iloc[i,2]= e85.iloc[i,2]+(.33*(.2))
    
    
    
    
e86 = pd.DataFrame(np.zeros((773, 3)))
e86.columns = ['a','b','c']




u=86

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e86.iloc[k,0] = dfp.iloc[i,0]
        e86.iloc[k,1] = dfp.iloc[i,1]
        e86.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e86.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e86.iloc[i,0]),4] == df200.iloc[int(e86.iloc[i,0]),4]:
        e86.iloc[i,2]= e86.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e86.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e86.iloc[i,0]),10] < 1 and  df200.iloc[int(e86.iloc[i,0]),10] <1) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e86.iloc[i,0]),10] < 2 and  df200.iloc[int(e86.iloc[i,0]),10] <2) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e86.iloc[i,0]),10] < 3 and  df200.iloc[int(e86.iloc[i,0]),10] <3) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e86.iloc[i,0]),10] < 4 and  df200.iloc[int(e86.iloc[i,0]),10] <4) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e86.iloc[i,0]),10] < 6 and  df200.iloc[int(e86.iloc[i,0]),10] <6) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e86.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e86.iloc[i,0]),5] < 700000 and  df200.iloc[int(e86.iloc[i,0]),5] < 700000) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e86.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e86.iloc[i,0]),5] < 1400000) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e86.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e86.iloc[i,0]),5] < 2100000) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e86.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e86.iloc[i,0]),5] < 2800000) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e86.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e86.iloc[i,0]),5] < 3500000) :
        e86.iloc[i,2]= e86.iloc[i,2]+(.33*(.2))
    
    
    
    
e87 = pd.DataFrame(np.zeros((773, 3)))
e87.columns = ['a','b','c']




u=87

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e87.iloc[k,0] = dfp.iloc[i,0]
        e87.iloc[k,1] = dfp.iloc[i,1]
        e87.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e87.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e87.iloc[i,0]),4] == df200.iloc[int(e87.iloc[i,0]),4]:
        e87.iloc[i,2]= e87.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e87.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e87.iloc[i,0]),10] < 1 and  df200.iloc[int(e87.iloc[i,0]),10] <1) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e87.iloc[i,0]),10] < 2 and  df200.iloc[int(e87.iloc[i,0]),10] <2) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e87.iloc[i,0]),10] < 3 and  df200.iloc[int(e87.iloc[i,0]),10] <3) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e87.iloc[i,0]),10] < 4 and  df200.iloc[int(e87.iloc[i,0]),10] <4) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e87.iloc[i,0]),10] < 6 and  df200.iloc[int(e87.iloc[i,0]),10] <6) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e87.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e87.iloc[i,0]),5] < 700000 and  df200.iloc[int(e87.iloc[i,0]),5] < 700000) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e87.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e87.iloc[i,0]),5] < 1400000) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e87.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e87.iloc[i,0]),5] < 2100000) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e87.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e87.iloc[i,0]),5] < 2800000) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e87.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e87.iloc[i,0]),5] < 3500000) :
        e87.iloc[i,2]= e87.iloc[i,2]+(.33*(.2))
    
    
    
    
e88 = pd.DataFrame(np.zeros((773, 3)))
e88.columns = ['a','b','c']




u=88

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e88.iloc[k,0] = dfp.iloc[i,0]
        e88.iloc[k,1] = dfp.iloc[i,1]
        e88.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e88.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e88.iloc[i,0]),4] == df200.iloc[int(e88.iloc[i,0]),4]:
        e88.iloc[i,2]= e88.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e88.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e88.iloc[i,0]),10] < 1 and  df200.iloc[int(e88.iloc[i,0]),10] <1) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e88.iloc[i,0]),10] < 2 and  df200.iloc[int(e88.iloc[i,0]),10] <2) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e88.iloc[i,0]),10] < 3 and  df200.iloc[int(e88.iloc[i,0]),10] <3) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e88.iloc[i,0]),10] < 4 and  df200.iloc[int(e88.iloc[i,0]),10] <4) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e88.iloc[i,0]),10] < 6 and  df200.iloc[int(e88.iloc[i,0]),10] <6) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e88.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e88.iloc[i,0]),5] < 700000 and  df200.iloc[int(e88.iloc[i,0]),5] < 700000) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e88.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e88.iloc[i,0]),5] < 1400000) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e88.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e88.iloc[i,0]),5] < 2100000) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e88.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e88.iloc[i,0]),5] < 2800000) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e88.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e88.iloc[i,0]),5] < 3500000) :
        e88.iloc[i,2]= e88.iloc[i,2]+(.33*(.2))
    
    
    
    
e89 = pd.DataFrame(np.zeros((773, 3)))
e89.columns = ['a','b','c']




u=89

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e89.iloc[k,0] = dfp.iloc[i,0]
        e89.iloc[k,1] = dfp.iloc[i,1]
        e89.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e89.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e89.iloc[i,0]),4] == df200.iloc[int(e89.iloc[i,0]),4]:
        e89.iloc[i,2]= e89.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e89.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e89.iloc[i,0]),10] < 1 and  df200.iloc[int(e89.iloc[i,0]),10] <1) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e89.iloc[i,0]),10] < 2 and  df200.iloc[int(e89.iloc[i,0]),10] <2) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e89.iloc[i,0]),10] < 3 and  df200.iloc[int(e89.iloc[i,0]),10] <3) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e89.iloc[i,0]),10] < 4 and  df200.iloc[int(e89.iloc[i,0]),10] <4) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e89.iloc[i,0]),10] < 6 and  df200.iloc[int(e89.iloc[i,0]),10] <6) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e89.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e89.iloc[i,0]),5] < 700000 and  df200.iloc[int(e89.iloc[i,0]),5] < 700000) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e89.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e89.iloc[i,0]),5] < 1400000) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e89.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e89.iloc[i,0]),5] < 2100000) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e89.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e89.iloc[i,0]),5] < 2800000) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e89.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e89.iloc[i,0]),5] < 3500000) :
        e89.iloc[i,2]= e89.iloc[i,2]+(.33*(.2))
    
    
    
    
e90 = pd.DataFrame(np.zeros((773, 3)))
e90.columns = ['a','b','c']




u=90

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e90.iloc[k,0] = dfp.iloc[i,0]
        e90.iloc[k,1] = dfp.iloc[i,1]
        e90.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e90.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e90.iloc[i,0]),4] == df200.iloc[int(e90.iloc[i,0]),4]:
        e90.iloc[i,2]= e90.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e90.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e90.iloc[i,0]),10] < 1 and  df200.iloc[int(e90.iloc[i,0]),10] <1) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e90.iloc[i,0]),10] < 2 and  df200.iloc[int(e90.iloc[i,0]),10] <2) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e90.iloc[i,0]),10] < 3 and  df200.iloc[int(e90.iloc[i,0]),10] <3) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e90.iloc[i,0]),10] < 4 and  df200.iloc[int(e90.iloc[i,0]),10] <4) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e90.iloc[i,0]),10] < 6 and  df200.iloc[int(e90.iloc[i,0]),10] <6) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e90.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e90.iloc[i,0]),5] < 700000 and  df200.iloc[int(e90.iloc[i,0]),5] < 700000) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e90.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e90.iloc[i,0]),5] < 1400000) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e90.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e90.iloc[i,0]),5] < 2100000) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e90.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e90.iloc[i,0]),5] < 2800000) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e90.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e90.iloc[i,0]),5] < 3500000) :
        e90.iloc[i,2]= e90.iloc[i,2]+(.33*(.2))
    
    
    
    
e91 = pd.DataFrame(np.zeros((773, 3)))
e91.columns = ['a','b','c']




u=91

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e91.iloc[k,0] = dfp.iloc[i,0]
        e91.iloc[k,1] = dfp.iloc[i,1]
        e91.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e91.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e91.iloc[i,0]),4] == df200.iloc[int(e91.iloc[i,0]),4]:
        e91.iloc[i,2]= e91.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e91.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e91.iloc[i,0]),10] < 1 and  df200.iloc[int(e91.iloc[i,0]),10] <1) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e91.iloc[i,0]),10] < 2 and  df200.iloc[int(e91.iloc[i,0]),10] <2) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e91.iloc[i,0]),10] < 3 and  df200.iloc[int(e91.iloc[i,0]),10] <3) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e91.iloc[i,0]),10] < 4 and  df200.iloc[int(e91.iloc[i,0]),10] <4) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e91.iloc[i,0]),10] < 6 and  df200.iloc[int(e91.iloc[i,0]),10] <6) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e91.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e91.iloc[i,0]),5] < 700000 and  df200.iloc[int(e91.iloc[i,0]),5] < 700000) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e91.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e91.iloc[i,0]),5] < 1400000) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e91.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e91.iloc[i,0]),5] < 2100000) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e91.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e91.iloc[i,0]),5] < 2800000) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e91.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e91.iloc[i,0]),5] < 3500000) :
        e91.iloc[i,2]= e91.iloc[i,2]+(.33*(.2))
    
    
    
    
    
e92 = pd.DataFrame(np.zeros((773, 3)))
e92.columns = ['a','b','c']




u=92

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e92.iloc[k,0] = dfp.iloc[i,0]
        e92.iloc[k,1] = dfp.iloc[i,1]
        e92.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e92.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e92.iloc[i,0]),4] == df200.iloc[int(e92.iloc[i,0]),4]:
        e92.iloc[i,2]= e92.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e92.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e92.iloc[i,0]),10] < 1 and  df200.iloc[int(e92.iloc[i,0]),10] <1) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e92.iloc[i,0]),10] < 2 and  df200.iloc[int(e92.iloc[i,0]),10] <2) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e92.iloc[i,0]),10] < 3 and  df200.iloc[int(e92.iloc[i,0]),10] <3) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e92.iloc[i,0]),10] < 4 and  df200.iloc[int(e92.iloc[i,0]),10] <4) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e92.iloc[i,0]),10] < 6 and  df200.iloc[int(e92.iloc[i,0]),10] <6) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e92.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e92.iloc[i,0]),5] < 700000 and  df200.iloc[int(e92.iloc[i,0]),5] < 700000) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e92.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e92.iloc[i,0]),5] < 1400000) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e92.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e92.iloc[i,0]),5] < 2100000) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e92.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e92.iloc[i,0]),5] < 2800000) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e92.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e92.iloc[i,0]),5] < 3500000) :
        e92.iloc[i,2]= e92.iloc[i,2]+(.33*(.2))
    
    
    
    
e93 = pd.DataFrame(np.zeros((773, 3)))
e93.columns = ['a','b','c']




u=93

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e93.iloc[k,0] = dfp.iloc[i,0]
        e93.iloc[k,1] = dfp.iloc[i,1]
        e93.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e93.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e93.iloc[i,0]),4] == df200.iloc[int(e93.iloc[i,0]),4]:
        e93.iloc[i,2]= e93.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e93.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e93.iloc[i,0]),10] < 1 and  df200.iloc[int(e93.iloc[i,0]),10] <1) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e93.iloc[i,0]),10] < 2 and  df200.iloc[int(e93.iloc[i,0]),10] <2) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e93.iloc[i,0]),10] < 3 and  df200.iloc[int(e93.iloc[i,0]),10] <3) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e93.iloc[i,0]),10] < 4 and  df200.iloc[int(e93.iloc[i,0]),10] <4) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e93.iloc[i,0]),10] < 6 and  df200.iloc[int(e93.iloc[i,0]),10] <6) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e93.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e93.iloc[i,0]),5] < 700000 and  df200.iloc[int(e93.iloc[i,0]),5] < 700000) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e93.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e93.iloc[i,0]),5] < 1400000) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e93.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e93.iloc[i,0]),5] < 2100000) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e93.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e93.iloc[i,0]),5] < 2800000) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e93.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e93.iloc[i,0]),5] < 3500000) :
        e93.iloc[i,2]= e93.iloc[i,2]+(.33*(.2))
    
    
    
    
e94 = pd.DataFrame(np.zeros((773, 3)))
e94.columns = ['a','b','c']




u=94

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e94.iloc[k,0] = dfp.iloc[i,0]
        e94.iloc[k,1] = dfp.iloc[i,1]
        e94.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e94.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e94.iloc[i,0]),4] == df200.iloc[int(e94.iloc[i,0]),4]:
        e94.iloc[i,2]= e94.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e94.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e94.iloc[i,0]),10] < 1 and  df200.iloc[int(e94.iloc[i,0]),10] <1) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e94.iloc[i,0]),10] < 2 and  df200.iloc[int(e94.iloc[i,0]),10] <2) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e94.iloc[i,0]),10] < 3 and  df200.iloc[int(e94.iloc[i,0]),10] <3) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e94.iloc[i,0]),10] < 4 and  df200.iloc[int(e94.iloc[i,0]),10] <4) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e94.iloc[i,0]),10] < 6 and  df200.iloc[int(e94.iloc[i,0]),10] <6) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e94.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e94.iloc[i,0]),5] < 700000 and  df200.iloc[int(e94.iloc[i,0]),5] < 700000) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e94.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e94.iloc[i,0]),5] < 1400000) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e94.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e94.iloc[i,0]),5] < 2100000) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e94.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e94.iloc[i,0]),5] < 2800000) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e94.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e94.iloc[i,0]),5] < 3500000) :
        e94.iloc[i,2]= e94.iloc[i,2]+(.33*(.2))
    
    
    
    
e95 = pd.DataFrame(np.zeros((773, 3)))
e95.columns = ['a','b','c']




u=95

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e95.iloc[k,0] = dfp.iloc[i,0]
        e95.iloc[k,1] = dfp.iloc[i,1]
        e95.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e95.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e95.iloc[i,0]),4] == df200.iloc[int(e95.iloc[i,0]),4]:
        e95.iloc[i,2]= e95.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e95.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e95.iloc[i,0]),10] < 1 and  df200.iloc[int(e95.iloc[i,0]),10] <1) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e95.iloc[i,0]),10] < 2 and  df200.iloc[int(e95.iloc[i,0]),10] <2) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e95.iloc[i,0]),10] < 3 and  df200.iloc[int(e95.iloc[i,0]),10] <3) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e95.iloc[i,0]),10] < 4 and  df200.iloc[int(e95.iloc[i,0]),10] <4) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e95.iloc[i,0]),10] < 6 and  df200.iloc[int(e95.iloc[i,0]),10] <6) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e95.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e95.iloc[i,0]),5] < 700000 and  df200.iloc[int(e95.iloc[i,0]),5] < 700000) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e95.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e95.iloc[i,0]),5] < 1400000) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e95.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e95.iloc[i,0]),5] < 2100000) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e95.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e95.iloc[i,0]),5] < 2800000) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e95.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e95.iloc[i,0]),5] < 3500000) :
        e95.iloc[i,2]= e95.iloc[i,2]+(.33*(.2))
    
    
    
    
e96 = pd.DataFrame(np.zeros((773, 3)))
e96.columns = ['a','b','c']




u=96

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e96.iloc[k,0] = dfp.iloc[i,0]
        e96.iloc[k,1] = dfp.iloc[i,1]
        e96.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e96.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e96.iloc[i,0]),4] == df200.iloc[int(e96.iloc[i,0]),4]:
        e96.iloc[i,2]= e96.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e96.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e96.iloc[i,0]),10] < 1 and  df200.iloc[int(e96.iloc[i,0]),10] <1) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e96.iloc[i,0]),10] < 2 and  df200.iloc[int(e96.iloc[i,0]),10] <2) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e96.iloc[i,0]),10] < 3 and  df200.iloc[int(e96.iloc[i,0]),10] <3) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e96.iloc[i,0]),10] < 4 and  df200.iloc[int(e96.iloc[i,0]),10] <4) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e96.iloc[i,0]),10] < 6 and  df200.iloc[int(e96.iloc[i,0]),10] <6) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e96.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e96.iloc[i,0]),5] < 700000 and  df200.iloc[int(e96.iloc[i,0]),5] < 700000) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e96.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e96.iloc[i,0]),5] < 1400000) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e96.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e96.iloc[i,0]),5] < 2100000) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e96.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e96.iloc[i,0]),5] < 2800000) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e96.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e96.iloc[i,0]),5] < 3500000) :
        e96.iloc[i,2]= e96.iloc[i,2]+(.33*(.2))
    
    
    
    
e97 = pd.DataFrame(np.zeros((773, 3)))
e97.columns = ['a','b','c']




u=97

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e97.iloc[k,0] = dfp.iloc[i,0]
        e97.iloc[k,1] = dfp.iloc[i,1]
        e97.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e97.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e97.iloc[i,0]),4] == df200.iloc[int(e97.iloc[i,0]),4]:
        e97.iloc[i,2]= e97.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e97.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e97.iloc[i,0]),10] < 1 and  df200.iloc[int(e97.iloc[i,0]),10] <1) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e97.iloc[i,0]),10] < 2 and  df200.iloc[int(e97.iloc[i,0]),10] <2) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e97.iloc[i,0]),10] < 3 and  df200.iloc[int(e97.iloc[i,0]),10] <3) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e97.iloc[i,0]),10] < 4 and  df200.iloc[int(e97.iloc[i,0]),10] <4) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e97.iloc[i,0]),10] < 6 and  df200.iloc[int(e97.iloc[i,0]),10] <6) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e97.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e97.iloc[i,0]),5] < 700000 and  df200.iloc[int(e97.iloc[i,0]),5] < 700000) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e97.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e97.iloc[i,0]),5] < 1400000) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e97.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e97.iloc[i,0]),5] < 2100000) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e97.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e97.iloc[i,0]),5] < 2800000) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e97.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e97.iloc[i,0]),5] < 3500000) :
        e97.iloc[i,2]= e97.iloc[i,2]+(.33*(.2))
    
    
    
    
e98 = pd.DataFrame(np.zeros((773, 3)))
e98.columns = ['a','b','c']




u=98

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e98.iloc[k,0] = dfp.iloc[i,0]
        e98.iloc[k,1] = dfp.iloc[i,1]
        e98.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e98.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e98.iloc[i,0]),4] == df200.iloc[int(e98.iloc[i,0]),4]:
        e98.iloc[i,2]= e98.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e98.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e98.iloc[i,0]),10] < 1 and  df200.iloc[int(e98.iloc[i,0]),10] <1) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e98.iloc[i,0]),10] < 2 and  df200.iloc[int(e98.iloc[i,0]),10] <2) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e98.iloc[i,0]),10] < 3 and  df200.iloc[int(e98.iloc[i,0]),10] <3) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e98.iloc[i,0]),10] < 4 and  df200.iloc[int(e98.iloc[i,0]),10] <4) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e98.iloc[i,0]),10] < 6 and  df200.iloc[int(e98.iloc[i,0]),10] <6) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e98.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e98.iloc[i,0]),5] < 700000 and  df200.iloc[int(e98.iloc[i,0]),5] < 700000) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e98.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e98.iloc[i,0]),5] < 1400000) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e98.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e98.iloc[i,0]),5] < 2100000) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e98.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e98.iloc[i,0]),5] < 2800000) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e98.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e98.iloc[i,0]),5] < 3500000) :
        e98.iloc[i,2]= e98.iloc[i,2]+(.33*(.2))
    
    
    
    
e99 = pd.DataFrame(np.zeros((773, 3)))
e99.columns = ['a','b','c']




u=99

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e99.iloc[k,0] = dfp.iloc[i,0]
        e99.iloc[k,1] = dfp.iloc[i,1]
        e99.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e99.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e99.iloc[i,0]),4] == df200.iloc[int(e99.iloc[i,0]),4]:
        e99.iloc[i,2]= e99.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e99.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e99.iloc[i,0]),10] < 1 and  df200.iloc[int(e99.iloc[i,0]),10] <1) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e99.iloc[i,0]),10] < 2 and  df200.iloc[int(e99.iloc[i,0]),10] <2) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e99.iloc[i,0]),10] < 3 and  df200.iloc[int(e99.iloc[i,0]),10] <3) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e99.iloc[i,0]),10] < 4 and  df200.iloc[int(e99.iloc[i,0]),10] <4) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e99.iloc[i,0]),10] < 6 and  df200.iloc[int(e99.iloc[i,0]),10] <6) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e99.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e99.iloc[i,0]),5] < 700000 and  df200.iloc[int(e99.iloc[i,0]),5] < 700000) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e99.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e99.iloc[i,0]),5] < 1400000) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e99.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e99.iloc[i,0]),5] < 2100000) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e99.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e99.iloc[i,0]),5] < 2800000) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e99.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e99.iloc[i,0]),5] < 3500000) :
        e99.iloc[i,2]= e99.iloc[i,2]+(.33*(.2))
    
    
    
    
e100 = pd.DataFrame(np.zeros((773, 3)))
e100.columns = ['a','b','c']




u=100

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e100.iloc[k,0] = dfp.iloc[i,0]
        e100.iloc[k,1] = dfp.iloc[i,1]
        e100.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e100.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e100.iloc[i,0]),4] == df200.iloc[int(e100.iloc[i,0]),4]:
        e100.iloc[i,2]= e100.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e100.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e100.iloc[i,0]),10] < 1 and  df200.iloc[int(e100.iloc[i,0]),10] <1) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e100.iloc[i,0]),10] < 2 and  df200.iloc[int(e100.iloc[i,0]),10] <2) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e100.iloc[i,0]),10] < 3 and  df200.iloc[int(e100.iloc[i,0]),10] <3) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e100.iloc[i,0]),10] < 4 and  df200.iloc[int(e100.iloc[i,0]),10] <4) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e100.iloc[i,0]),10] < 6 and  df200.iloc[int(e100.iloc[i,0]),10] <6) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e100.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e100.iloc[i,0]),5] < 700000 and  df200.iloc[int(e100.iloc[i,0]),5] < 700000) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(1))
    
    elif (df200.iloc[int(e100.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e100.iloc[i,0]),5] < 1400000) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e100.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e100.iloc[i,0]),5] < 2100000) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e100.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e100.iloc[i,0]),5] < 2800000) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e100.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e100.iloc[i,0]),5] < 3500000) :
        e100.iloc[i,2]= e100.iloc[i,2]+(.33*(.2))
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:12:56 2018

@author: trinhhang
"""

import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx

#  os.chdir('/Users/trinhhang/Documents/TDA Project')
# df200 = pd.read_csv('df200.csv')
# dfp = pd.read_csv('dfp.csv')

e0 = pd.DataFrame(np.zeros((773, 3)))
e0.columns = ['a','b','c']




u=0

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e0.iloc[k,0] = dfp.iloc[i,0]
        e0.iloc[k,1] = dfp.iloc[i,1]
        e0.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e0.iloc[i,0]),4] == df200.iloc[int(e0.iloc[i,0]),4]:
        e0.iloc[i,2]= e0.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 1 and  df200.iloc[int(e0.iloc[i,0]),10] <1) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 2 and  df200.iloc[int(e0.iloc[i,0]),10] <2) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 3 and  df200.iloc[int(e0.iloc[i,0]),10] <3) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 4 and  df200.iloc[int(e0.iloc[i,0]),10] <4) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e0.iloc[i,0]),10] < 6 and  df200.iloc[int(e0.iloc[i,0]),10] <6) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e0.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 700000 and  df200.iloc[int(e0.iloc[i,0]),5] <700000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e0.iloc[i,0]),5] <1400000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e0.iloc[i,0]),5] <2100000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e0.iloc[i,0]),5] <2800000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e0.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e0.iloc[i,0]),5] <3500000) :
        e0.iloc[i,2]= e0.iloc[i,2]+(.33*(.2))
        
        
        
############### e  = 101
e101 = pd.DataFrame(np.zeros((773, 3)))
e101.columns = ['a','b','c']




u=101

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e101.iloc[k,0] = dfp.iloc[i,0]
        e101.iloc[k,1] = dfp.iloc[i,1]
        e101.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e101.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e101.iloc[i,0]),4] == df200.iloc[int(e101.iloc[i,0]),4]:
        e101.iloc[i,2]= e101.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e101.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e101.iloc[i,0]),10] < 1 and  df200.iloc[int(e101.iloc[i,0]),10] <1) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e101.iloc[i,0]),10] < 2 and  df200.iloc[int(e101.iloc[i,0]),10] <2) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e101.iloc[i,0]),10] < 3 and  df200.iloc[int(e101.iloc[i,0]),10] <3) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e101.iloc[i,0]),10] < 4 and  df200.iloc[int(e101.iloc[i,0]),10] <4) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e101.iloc[i,0]),10] < 6 and  df200.iloc[int(e101.iloc[i,0]),10] <6) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e101.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e101.iloc[i,0]),5] < 700000 and  df200.iloc[int(e101.iloc[i,0]),5] <700000) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e101.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e101.iloc[i,0]),5] <1400000) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e101.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e101.iloc[i,0]),5] <2100000) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e101.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e101.iloc[i,0]),5] <2800000) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e101.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e101.iloc[i,0]),5] <3500000) :
        e101.iloc[i,2]= e101.iloc[i,2]+(.33*(.2))   
        
####################################    END e = 101   ###################################
        
############### e  = 102
e102 = pd.DataFrame(np.zeros((773, 3)))
e102.columns = ['a','b','c']




u=102

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e102.iloc[k,0] = dfp.iloc[i,0]
        e102.iloc[k,1] = dfp.iloc[i,1]
        e102.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e102.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e102.iloc[i,0]),4] == df200.iloc[int(e102.iloc[i,0]),4]:
        e102.iloc[i,2]= e102.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e102.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e102.iloc[i,0]),10] < 1 and  df200.iloc[int(e102.iloc[i,0]),10] <1) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e102.iloc[i,0]),10] < 2 and  df200.iloc[int(e102.iloc[i,0]),10] <2) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e102.iloc[i,0]),10] < 3 and  df200.iloc[int(e102.iloc[i,0]),10] <3) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e102.iloc[i,0]),10] < 4 and  df200.iloc[int(e102.iloc[i,0]),10] <4) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e102.iloc[i,0]),10] < 6 and  df200.iloc[int(e102.iloc[i,0]),10] <6) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e102.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e102.iloc[i,0]),5] < 700000 and  df200.iloc[int(e102.iloc[i,0]),5] <700000) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e102.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e102.iloc[i,0]),5] <1400000) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e102.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e102.iloc[i,0]),5] <2100000) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e102.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e102.iloc[i,0]),5] <2800000) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e102.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e102.iloc[i,0]),5] <3500000) :
        e102.iloc[i,2]= e102.iloc[i,2]+(.33*(.2))

######## END e = 102 ##############
        
############### e  = 103
e103 = pd.DataFrame(np.zeros((773, 3)))
e103.columns = ['a','b','c']




u=103

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e103.iloc[k,0] = dfp.iloc[i,0]
        e103.iloc[k,1] = dfp.iloc[i,1]
        e103.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e103.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e103.iloc[i,0]),4] == df200.iloc[int(e103.iloc[i,0]),4]:
        e103.iloc[i,2]= e103.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e103.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e103.iloc[i,0]),10] < 1 and  df200.iloc[int(e103.iloc[i,0]),10] <1) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e103.iloc[i,0]),10] < 2 and  df200.iloc[int(e103.iloc[i,0]),10] <2) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e103.iloc[i,0]),10] < 3 and  df200.iloc[int(e103.iloc[i,0]),10] <3) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e103.iloc[i,0]),10] < 4 and  df200.iloc[int(e103.iloc[i,0]),10] <4) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e103.iloc[i,0]),10] < 6 and  df200.iloc[int(e103.iloc[i,0]),10] <6) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e103.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e103.iloc[i,0]),5] < 700000 and  df200.iloc[int(e103.iloc[i,0]),5] <700000) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e103.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e103.iloc[i,0]),5] <1400000) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e103.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e103.iloc[i,0]),5] <2100000) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e103.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e103.iloc[i,0]),5] <2800000) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e103.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e103.iloc[i,0]),5] <3500000) :
        e103.iloc[i,2]= e103.iloc[i,2]+(.33*(.2))

#################################### END e = 103 #############################        

############### e  = 104
e104 = pd.DataFrame(np.zeros((773, 3)))
e104.columns = ['a','b','c']




u=104

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e104.iloc[k,0] = dfp.iloc[i,0]
        e104.iloc[k,1] = dfp.iloc[i,1]
        e104.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e104.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e104.iloc[i,0]),4] == df200.iloc[int(e104.iloc[i,0]),4]:
        e104.iloc[i,2]= e104.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e104.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e104.iloc[i,0]),10] < 1 and  df200.iloc[int(e104.iloc[i,0]),10] <1) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e104.iloc[i,0]),10] < 2 and  df200.iloc[int(e104.iloc[i,0]),10] <2) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e104.iloc[i,0]),10] < 3 and  df200.iloc[int(e104.iloc[i,0]),10] <3) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e104.iloc[i,0]),10] < 4 and  df200.iloc[int(e104.iloc[i,0]),10] <4) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e104.iloc[i,0]),10] < 6 and  df200.iloc[int(e104.iloc[i,0]),10] <6) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e104.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e104.iloc[i,0]),5] < 700000 and  df200.iloc[int(e104.iloc[i,0]),5] <700000) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e104.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e104.iloc[i,0]),5] <1400000) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e104.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e104.iloc[i,0]),5] <2100000) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e104.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e104.iloc[i,0]),5] <2800000) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e104.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e104.iloc[i,0]),5] <3500000) :
        e104.iloc[i,2]= e104.iloc[i,2]+(.33*(.2))

#################################### END e = 104 #############################        
        
############### e  = 105
e105 = pd.DataFrame(np.zeros((773, 3)))
e105.columns = ['a','b','c']




u=105

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e105.iloc[k,0] = dfp.iloc[i,0]
        e105.iloc[k,1] = dfp.iloc[i,1]
        e105.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e105.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e105.iloc[i,0]),4] == df200.iloc[int(e105.iloc[i,0]),4]:
        e105.iloc[i,2]= e105.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e105.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e105.iloc[i,0]),10] < 1 and  df200.iloc[int(e105.iloc[i,0]),10] <1) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e105.iloc[i,0]),10] < 2 and  df200.iloc[int(e105.iloc[i,0]),10] <2) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e105.iloc[i,0]),10] < 3 and  df200.iloc[int(e105.iloc[i,0]),10] <3) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e105.iloc[i,0]),10] < 4 and  df200.iloc[int(e105.iloc[i,0]),10] <4) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e105.iloc[i,0]),10] < 6 and  df200.iloc[int(e105.iloc[i,0]),10] <6) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e105.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e105.iloc[i,0]),5] < 700000 and  df200.iloc[int(e105.iloc[i,0]),5] <700000) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e105.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e105.iloc[i,0]),5] <1400000) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e105.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e105.iloc[i,0]),5] <2100000) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e105.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e105.iloc[i,0]),5] <2800000) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e105.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e105.iloc[i,0]),5] <3500000) :
        e105.iloc[i,2]= e105.iloc[i,2]+(.33*(.2))

#################################### END e = 105 #############################        

############### e  = 106
e106 = pd.DataFrame(np.zeros((773, 3)))
e106.columns = ['a','b','c']




u=106

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e106.iloc[k,0] = dfp.iloc[i,0]
        e106.iloc[k,1] = dfp.iloc[i,1]
        e106.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e106.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e106.iloc[i,0]),4] == df200.iloc[int(e106.iloc[i,0]),4]:
        e106.iloc[i,2]= e106.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e106.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e106.iloc[i,0]),10] < 1 and  df200.iloc[int(e106.iloc[i,0]),10] <1) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e106.iloc[i,0]),10] < 2 and  df200.iloc[int(e106.iloc[i,0]),10] <2) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e106.iloc[i,0]),10] < 3 and  df200.iloc[int(e106.iloc[i,0]),10] <3) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e106.iloc[i,0]),10] < 4 and  df200.iloc[int(e106.iloc[i,0]),10] <4) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e106.iloc[i,0]),10] < 6 and  df200.iloc[int(e106.iloc[i,0]),10] <6) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e106.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e106.iloc[i,0]),5] < 700000 and  df200.iloc[int(e106.iloc[i,0]),5] <700000) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e106.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e106.iloc[i,0]),5] <1400000) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e106.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e106.iloc[i,0]),5] <2100000) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e106.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e106.iloc[i,0]),5] <2800000) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e106.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e106.iloc[i,0]),5] <3500000) :
        e106.iloc[i,2]= e106.iloc[i,2]+(.33*(.2))

#################################### END e = 106 #############################        

############### e  = 107
e107 = pd.DataFrame(np.zeros((773, 3)))
e107.columns = ['a','b','c']




u=107

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e107.iloc[k,0] = dfp.iloc[i,0]
        e107.iloc[k,1] = dfp.iloc[i,1]
        e107.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e107.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e107.iloc[i,0]),4] == df200.iloc[int(e107.iloc[i,0]),4]:
        e107.iloc[i,2]= e107.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e107.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e107.iloc[i,0]),10] < 1 and  df200.iloc[int(e107.iloc[i,0]),10] <1) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e107.iloc[i,0]),10] < 2 and  df200.iloc[int(e107.iloc[i,0]),10] <2) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e107.iloc[i,0]),10] < 3 and  df200.iloc[int(e107.iloc[i,0]),10] <3) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e107.iloc[i,0]),10] < 4 and  df200.iloc[int(e107.iloc[i,0]),10] <4) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e107.iloc[i,0]),10] < 6 and  df200.iloc[int(e107.iloc[i,0]),10] <6) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e107.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e107.iloc[i,0]),5] < 700000 and  df200.iloc[int(e107.iloc[i,0]),5] <700000) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e107.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e107.iloc[i,0]),5] <1400000) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e107.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e107.iloc[i,0]),5] <2100000) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e107.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e107.iloc[i,0]),5] <2800000) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e107.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e107.iloc[i,0]),5] <3500000) :
        e107.iloc[i,2]= e107.iloc[i,2]+(.33*(.2))

#################################### END e = 107 #############################        

############### e  = 108
e108 = pd.DataFrame(np.zeros((773, 3)))
e108.columns = ['a','b','c']




u=108

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e108.iloc[k,0] = dfp.iloc[i,0]
        e108.iloc[k,1] = dfp.iloc[i,1]
        e108.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e108.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e108.iloc[i,0]),4] == df200.iloc[int(e108.iloc[i,0]),4]:
        e108.iloc[i,2]= e108.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e108.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e108.iloc[i,0]),10] < 1 and  df200.iloc[int(e108.iloc[i,0]),10] <1) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e108.iloc[i,0]),10] < 2 and  df200.iloc[int(e108.iloc[i,0]),10] <2) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e108.iloc[i,0]),10] < 3 and  df200.iloc[int(e108.iloc[i,0]),10] <3) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e108.iloc[i,0]),10] < 4 and  df200.iloc[int(e108.iloc[i,0]),10] <4) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e108.iloc[i,0]),10] < 6 and  df200.iloc[int(e108.iloc[i,0]),10] <6) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e108.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e108.iloc[i,0]),5] < 700000 and  df200.iloc[int(e108.iloc[i,0]),5] <700000) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e108.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e108.iloc[i,0]),5] <1400000) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e108.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e108.iloc[i,0]),5] <2100000) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e108.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e108.iloc[i,0]),5] <2800000) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e108.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e108.iloc[i,0]),5] <3500000) :
        e108.iloc[i,2]= e108.iloc[i,2]+(.33*(.2))

#################################### END e = 108 #############################    
        
############### e  = 109
e109 = pd.DataFrame(np.zeros((773, 3)))
e109.columns = ['a','b','c']




u=109

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e109.iloc[k,0] = dfp.iloc[i,0]
        e109.iloc[k,1] = dfp.iloc[i,1]
        e109.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e109.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e109.iloc[i,0]),4] == df200.iloc[int(e109.iloc[i,0]),4]:
        e109.iloc[i,2]= e109.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e109.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e109.iloc[i,0]),10] < 1 and  df200.iloc[int(e109.iloc[i,0]),10] <1) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e109.iloc[i,0]),10] < 2 and  df200.iloc[int(e109.iloc[i,0]),10] <2) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e109.iloc[i,0]),10] < 3 and  df200.iloc[int(e109.iloc[i,0]),10] <3) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e109.iloc[i,0]),10] < 4 and  df200.iloc[int(e109.iloc[i,0]),10] <4) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e109.iloc[i,0]),10] < 6 and  df200.iloc[int(e109.iloc[i,0]),10] <6) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e109.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e109.iloc[i,0]),5] < 700000 and  df200.iloc[int(e109.iloc[i,0]),5] <700000) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e109.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e109.iloc[i,0]),5] <1400000) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e109.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e109.iloc[i,0]),5] <2100000) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e109.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e109.iloc[i,0]),5] <2800000) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e109.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e109.iloc[i,0]),5] <3500000) :
        e109.iloc[i,2]= e109.iloc[i,2]+(.33*(.2))

#################################### END e = 109 #############################        

############### e  = 109
e110 = pd.DataFrame(np.zeros((773, 3)))
e110.columns = ['a','b','c']




u=110

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e110.iloc[k,0] = dfp.iloc[i,0]
        e110.iloc[k,1] = dfp.iloc[i,1]
        e110.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e110.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e110.iloc[i,0]),4] == df200.iloc[int(e110.iloc[i,0]),4]:
        e110.iloc[i,2]= e110.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e110.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e110.iloc[i,0]),10] < 1 and  df200.iloc[int(e110.iloc[i,0]),10] <1) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e110.iloc[i,0]),10] < 2 and  df200.iloc[int(e110.iloc[i,0]),10] <2) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e110.iloc[i,0]),10] < 3 and  df200.iloc[int(e110.iloc[i,0]),10] <3) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e110.iloc[i,0]),10] < 4 and  df200.iloc[int(e110.iloc[i,0]),10] <4) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e110.iloc[i,0]),10] < 6 and  df200.iloc[int(e110.iloc[i,0]),10] <6) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e110.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e110.iloc[i,0]),5] < 700000 and  df200.iloc[int(e110.iloc[i,0]),5] <700000) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e110.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e110.iloc[i,0]),5] <1400000) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e110.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e110.iloc[i,0]),5] <2100000) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e110.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e110.iloc[i,0]),5] <2800000) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e110.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e110.iloc[i,0]),5] <3500000) :
        e110.iloc[i,2]= e110.iloc[i,2]+(.33*(.2))

#################################### END e = 110 #############################   

############### e  = 109
e111 = pd.DataFrame(np.zeros((773, 3)))
e111.columns = ['a','b','c']




u=111

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e111.iloc[k,0] = dfp.iloc[i,0]
        e111.iloc[k,1] = dfp.iloc[i,1]
        e111.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e111.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e111.iloc[i,0]),4] == df200.iloc[int(e111.iloc[i,0]),4]:
        e111.iloc[i,2]= e111.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e111.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e111.iloc[i,0]),10] < 1 and  df200.iloc[int(e111.iloc[i,0]),10] <1) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e111.iloc[i,0]),10] < 2 and  df200.iloc[int(e111.iloc[i,0]),10] <2) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e111.iloc[i,0]),10] < 3 and  df200.iloc[int(e111.iloc[i,0]),10] <3) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e111.iloc[i,0]),10] < 4 and  df200.iloc[int(e111.iloc[i,0]),10] <4) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e111.iloc[i,0]),10] < 6 and  df200.iloc[int(e111.iloc[i,0]),10] <6) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e111.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e111.iloc[i,0]),5] < 700000 and  df200.iloc[int(e111.iloc[i,0]),5] <700000) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e111.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e111.iloc[i,0]),5] <1400000) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e111.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e111.iloc[i,0]),5] <2100000) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e111.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e111.iloc[i,0]),5] <2800000) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e111.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e111.iloc[i,0]),5] <3500000) :
        e111.iloc[i,2]= e111.iloc[i,2]+(.33*(.2))

#################################### END e = 111 #############################        

############### e  = 112
e112 = pd.DataFrame(np.zeros((773, 3)))
e112.columns = ['a','b','c']




u=112

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e112.iloc[k,0] = dfp.iloc[i,0]
        e112.iloc[k,1] = dfp.iloc[i,1]
        e112.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e112.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e112.iloc[i,0]),4] == df200.iloc[int(e112.iloc[i,0]),4]:
        e112.iloc[i,2]= e112.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e112.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e112.iloc[i,0]),10] < 1 and  df200.iloc[int(e112.iloc[i,0]),10] <1) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e112.iloc[i,0]),10] < 2 and  df200.iloc[int(e112.iloc[i,0]),10] <2) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e112.iloc[i,0]),10] < 3 and  df200.iloc[int(e112.iloc[i,0]),10] <3) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e112.iloc[i,0]),10] < 4 and  df200.iloc[int(e112.iloc[i,0]),10] <4) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e112.iloc[i,0]),10] < 6 and  df200.iloc[int(e112.iloc[i,0]),10] <6) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e112.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e112.iloc[i,0]),5] < 700000 and  df200.iloc[int(e112.iloc[i,0]),5] <700000) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e112.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e112.iloc[i,0]),5] <1400000) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e112.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e112.iloc[i,0]),5] <2100000) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e112.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e112.iloc[i,0]),5] <2800000) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e112.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e112.iloc[i,0]),5] <3500000) :
        e112.iloc[i,2]= e112.iloc[i,2]+(.33*(.2))

#################################### END e = 112 #############################        

############### e  = 113
e113 = pd.DataFrame(np.zeros((773, 3)))
e113.columns = ['a','b','c']




u=113

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e113.iloc[k,0] = dfp.iloc[i,0]
        e113.iloc[k,1] = dfp.iloc[i,1]
        e113.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e113.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e113.iloc[i,0]),4] == df200.iloc[int(e113.iloc[i,0]),4]:
        e113.iloc[i,2]= e113.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e113.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e113.iloc[i,0]),10] < 1 and  df200.iloc[int(e113.iloc[i,0]),10] <1) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e113.iloc[i,0]),10] < 2 and  df200.iloc[int(e113.iloc[i,0]),10] <2) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e113.iloc[i,0]),10] < 3 and  df200.iloc[int(e113.iloc[i,0]),10] <3) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e113.iloc[i,0]),10] < 4 and  df200.iloc[int(e113.iloc[i,0]),10] <4) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e113.iloc[i,0]),10] < 6 and  df200.iloc[int(e113.iloc[i,0]),10] <6) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e113.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e113.iloc[i,0]),5] < 700000 and  df200.iloc[int(e113.iloc[i,0]),5] <700000) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e113.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e113.iloc[i,0]),5] <1400000) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e113.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e113.iloc[i,0]),5] <2100000) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e113.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e113.iloc[i,0]),5] <2800000) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e113.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e113.iloc[i,0]),5] <3500000) :
        e113.iloc[i,2]= e113.iloc[i,2]+(.33*(.2))

#################################### END e = 113 #############################        

 ############### e  = 114
e114 = pd.DataFrame(np.zeros((773, 3)))
e114.columns = ['a','b','c']




u=114

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e114.iloc[k,0] = dfp.iloc[i,0]
        e114.iloc[k,1] = dfp.iloc[i,1]
        e114.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e114.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e114.iloc[i,0]),4] == df200.iloc[int(e114.iloc[i,0]),4]:
        e114.iloc[i,2]= e114.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e114.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e114.iloc[i,0]),10] < 1 and  df200.iloc[int(e114.iloc[i,0]),10] <1) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e114.iloc[i,0]),10] < 2 and  df200.iloc[int(e114.iloc[i,0]),10] <2) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e114.iloc[i,0]),10] < 3 and  df200.iloc[int(e114.iloc[i,0]),10] <3) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e114.iloc[i,0]),10] < 4 and  df200.iloc[int(e114.iloc[i,0]),10] <4) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e114.iloc[i,0]),10] < 6 and  df200.iloc[int(e114.iloc[i,0]),10] <6) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e114.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e114.iloc[i,0]),5] < 700000 and  df200.iloc[int(e114.iloc[i,0]),5] <700000) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e114.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e114.iloc[i,0]),5] <1400000) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e114.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e114.iloc[i,0]),5] <2100000) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e114.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e114.iloc[i,0]),5] <2800000) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e114.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e114.iloc[i,0]),5] <3500000) :
        e114.iloc[i,2]= e114.iloc[i,2]+(.33*(.2))

#################################### END e = 114 #############################        

############### e  = 115
e115 = pd.DataFrame(np.zeros((773, 3)))
e115.columns = ['a','b','c']




u=115

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e115.iloc[k,0] = dfp.iloc[i,0]
        e115.iloc[k,1] = dfp.iloc[i,1]
        e115.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e115.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e115.iloc[i,0]),4] == df200.iloc[int(e115.iloc[i,0]),4]:
        e115.iloc[i,2]= e115.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e115.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e115.iloc[i,0]),10] < 1 and  df200.iloc[int(e115.iloc[i,0]),10] <1) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e115.iloc[i,0]),10] < 2 and  df200.iloc[int(e115.iloc[i,0]),10] <2) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e115.iloc[i,0]),10] < 3 and  df200.iloc[int(e115.iloc[i,0]),10] <3) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e115.iloc[i,0]),10] < 4 and  df200.iloc[int(e115.iloc[i,0]),10] <4) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e115.iloc[i,0]),10] < 6 and  df200.iloc[int(e115.iloc[i,0]),10] <6) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e115.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e115.iloc[i,0]),5] < 700000 and  df200.iloc[int(e115.iloc[i,0]),5] <700000) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e115.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e115.iloc[i,0]),5] <1400000) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e115.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e115.iloc[i,0]),5] <2100000) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e115.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e115.iloc[i,0]),5] <2800000) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e115.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e115.iloc[i,0]),5] <3500000) :
        e115.iloc[i,2]= e115.iloc[i,2]+(.33*(.2))

#################################### END e = 115 #############################        

############### e  = 116
e116 = pd.DataFrame(np.zeros((773, 3)))
e116.columns = ['a','b','c']




u=116

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e116.iloc[k,0] = dfp.iloc[i,0]
        e116.iloc[k,1] = dfp.iloc[i,1]
        e116.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e116.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e116.iloc[i,0]),4] == df200.iloc[int(e116.iloc[i,0]),4]:
        e116.iloc[i,2]= e116.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e116.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e116.iloc[i,0]),10] < 1 and  df200.iloc[int(e116.iloc[i,0]),10] <1) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e116.iloc[i,0]),10] < 2 and  df200.iloc[int(e116.iloc[i,0]),10] <2) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e116.iloc[i,0]),10] < 3 and  df200.iloc[int(e116.iloc[i,0]),10] <3) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e116.iloc[i,0]),10] < 4 and  df200.iloc[int(e116.iloc[i,0]),10] <4) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e116.iloc[i,0]),10] < 6 and  df200.iloc[int(e116.iloc[i,0]),10] <6) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e116.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e116.iloc[i,0]),5] < 700000 and  df200.iloc[int(e116.iloc[i,0]),5] <700000) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e116.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e116.iloc[i,0]),5] <1400000) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e116.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e116.iloc[i,0]),5] <2100000) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e116.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e116.iloc[i,0]),5] <2800000) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e116.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e116.iloc[i,0]),5] <3500000) :
        e116.iloc[i,2]= e116.iloc[i,2]+(.33*(.2))

#################################### END e = 116 #############################        

############### e  = 117
e117 = pd.DataFrame(np.zeros((773, 3)))
e117.columns = ['a','b','c']




u=117

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e117.iloc[k,0] = dfp.iloc[i,0]
        e117.iloc[k,1] = dfp.iloc[i,1]
        e117.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e117.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e117.iloc[i,0]),4] == df200.iloc[int(e117.iloc[i,0]),4]:
        e117.iloc[i,2]= e117.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e117.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e117.iloc[i,0]),10] < 1 and  df200.iloc[int(e117.iloc[i,0]),10] <1) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e117.iloc[i,0]),10] < 2 and  df200.iloc[int(e117.iloc[i,0]),10] <2) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e117.iloc[i,0]),10] < 3 and  df200.iloc[int(e117.iloc[i,0]),10] <3) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e117.iloc[i,0]),10] < 4 and  df200.iloc[int(e117.iloc[i,0]),10] <4) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e117.iloc[i,0]),10] < 6 and  df200.iloc[int(e117.iloc[i,0]),10] <6) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e117.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e117.iloc[i,0]),5] < 700000 and  df200.iloc[int(e117.iloc[i,0]),5] <700000) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e117.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e117.iloc[i,0]),5] <1400000) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e117.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e117.iloc[i,0]),5] <2100000) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e117.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e117.iloc[i,0]),5] <2800000) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e117.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e117.iloc[i,0]),5] <3500000) :
        e117.iloc[i,2]= e117.iloc[i,2]+(.33*(.2))

#################################### END e = 117 #############################        

   ############### e  = 118
e118 = pd.DataFrame(np.zeros((773, 3)))
e118.columns = ['a','b','c']




u=118

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e118.iloc[k,0] = dfp.iloc[i,0]
        e118.iloc[k,1] = dfp.iloc[i,1]
        e118.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e118.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e118.iloc[i,0]),4] == df200.iloc[int(e118.iloc[i,0]),4]:
        e118.iloc[i,2]= e118.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e118.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e118.iloc[i,0]),10] < 1 and  df200.iloc[int(e118.iloc[i,0]),10] <1) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e118.iloc[i,0]),10] < 2 and  df200.iloc[int(e118.iloc[i,0]),10] <2) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e118.iloc[i,0]),10] < 3 and  df200.iloc[int(e118.iloc[i,0]),10] <3) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e118.iloc[i,0]),10] < 4 and  df200.iloc[int(e118.iloc[i,0]),10] <4) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e118.iloc[i,0]),10] < 6 and  df200.iloc[int(e118.iloc[i,0]),10] <6) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e118.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e118.iloc[i,0]),5] < 700000 and  df200.iloc[int(e118.iloc[i,0]),5] <700000) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e118.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e118.iloc[i,0]),5] <1400000) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e118.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e118.iloc[i,0]),5] <2100000) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e118.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e118.iloc[i,0]),5] <2800000) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e118.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e118.iloc[i,0]),5] <3500000) :
        e118.iloc[i,2]= e118.iloc[i,2]+(.33*(.2))

#################################### END e = 118 #############################        

############### e  = 119
e119 = pd.DataFrame(np.zeros((773, 3)))
e119.columns = ['a','b','c']




u=119

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e119.iloc[k,0] = dfp.iloc[i,0]
        e119.iloc[k,1] = dfp.iloc[i,1]
        e119.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e119.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e119.iloc[i,0]),4] == df200.iloc[int(e119.iloc[i,0]),4]:
        e119.iloc[i,2]= e119.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e119.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e119.iloc[i,0]),10] < 1 and  df200.iloc[int(e119.iloc[i,0]),10] <1) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e119.iloc[i,0]),10] < 2 and  df200.iloc[int(e119.iloc[i,0]),10] <2) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e119.iloc[i,0]),10] < 3 and  df200.iloc[int(e119.iloc[i,0]),10] <3) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e119.iloc[i,0]),10] < 4 and  df200.iloc[int(e119.iloc[i,0]),10] <4) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e119.iloc[i,0]),10] < 6 and  df200.iloc[int(e119.iloc[i,0]),10] <6) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e119.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e119.iloc[i,0]),5] < 700000 and  df200.iloc[int(e119.iloc[i,0]),5] <700000) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e119.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e119.iloc[i,0]),5] <1400000) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e119.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e119.iloc[i,0]),5] <2100000) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e119.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e119.iloc[i,0]),5] <2800000) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e119.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e119.iloc[i,0]),5] <3500000) :
        e119.iloc[i,2]= e119.iloc[i,2]+(.33*(.2))

#################################### END e = 119 #############################        
        

############### e  = 120
e120 = pd.DataFrame(np.zeros((773, 3)))
e120.columns = ['a','b','c']




u=120

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e120.iloc[k,0] = dfp.iloc[i,0]
        e120.iloc[k,1] = dfp.iloc[i,1]
        e120.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e120.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e120.iloc[i,0]),4] == df200.iloc[int(e120.iloc[i,0]),4]:
        e120.iloc[i,2]= e120.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e120.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e120.iloc[i,0]),10] < 1 and  df200.iloc[int(e120.iloc[i,0]),10] <1) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e120.iloc[i,0]),10] < 2 and  df200.iloc[int(e120.iloc[i,0]),10] <2) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e120.iloc[i,0]),10] < 3 and  df200.iloc[int(e120.iloc[i,0]),10] <3) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e120.iloc[i,0]),10] < 4 and  df200.iloc[int(e120.iloc[i,0]),10] <4) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e120.iloc[i,0]),10] < 6 and  df200.iloc[int(e120.iloc[i,0]),10] <6) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e120.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e120.iloc[i,0]),5] < 700000 and  df200.iloc[int(e120.iloc[i,0]),5] <700000) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e120.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e120.iloc[i,0]),5] <1400000) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e120.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e120.iloc[i,0]),5] <2100000) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e120.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e120.iloc[i,0]),5] <2800000) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e120.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e120.iloc[i,0]),5] <3500000) :
        e120.iloc[i,2]= e120.iloc[i,2]+(.33*(.2))

#################################### END e = 120 #############################        

############### e  = 121
e121 = pd.DataFrame(np.zeros((773, 3)))
e121.columns = ['a','b','c']




u=121

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e121.iloc[k,0] = dfp.iloc[i,0]
        e121.iloc[k,1] = dfp.iloc[i,1]
        e121.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e121.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e121.iloc[i,0]),4] == df200.iloc[int(e121.iloc[i,0]),4]:
        e121.iloc[i,2]= e121.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e121.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e121.iloc[i,0]),10] < 1 and  df200.iloc[int(e121.iloc[i,0]),10] <1) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e121.iloc[i,0]),10] < 2 and  df200.iloc[int(e121.iloc[i,0]),10] <2) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e121.iloc[i,0]),10] < 3 and  df200.iloc[int(e121.iloc[i,0]),10] <3) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e121.iloc[i,0]),10] < 4 and  df200.iloc[int(e121.iloc[i,0]),10] <4) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e121.iloc[i,0]),10] < 6 and  df200.iloc[int(e121.iloc[i,0]),10] <6) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e121.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e121.iloc[i,0]),5] < 700000 and  df200.iloc[int(e121.iloc[i,0]),5] <700000) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e121.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e121.iloc[i,0]),5] <1400000) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e121.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e121.iloc[i,0]),5] <2100000) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e121.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e121.iloc[i,0]),5] <2800000) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e121.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e121.iloc[i,0]),5] <3500000) :
        e121.iloc[i,2]= e121.iloc[i,2]+(.33*(.2))

#################################### END e = 121#############################        

############### e  = 122
e122 = pd.DataFrame(np.zeros((773, 3)))
e122.columns = ['a','b','c']




u=122

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e122.iloc[k,0] = dfp.iloc[i,0]
        e122.iloc[k,1] = dfp.iloc[i,1]
        e122.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e122.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e122.iloc[i,0]),4] == df200.iloc[int(e122.iloc[i,0]),4]:
        e122.iloc[i,2]= e122.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e122.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e122.iloc[i,0]),10] < 1 and  df200.iloc[int(e122.iloc[i,0]),10] <1) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e122.iloc[i,0]),10] < 2 and  df200.iloc[int(e122.iloc[i,0]),10] <2) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e122.iloc[i,0]),10] < 3 and  df200.iloc[int(e122.iloc[i,0]),10] <3) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e122.iloc[i,0]),10] < 4 and  df200.iloc[int(e122.iloc[i,0]),10] <4) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e122.iloc[i,0]),10] < 6 and  df200.iloc[int(e122.iloc[i,0]),10] <6) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e122.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e122.iloc[i,0]),5] < 700000 and  df200.iloc[int(e122.iloc[i,0]),5] <700000) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e122.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e122.iloc[i,0]),5] <1400000) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e122.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e122.iloc[i,0]),5] <2100000) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e122.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e122.iloc[i,0]),5] <2800000) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e122.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e122.iloc[i,0]),5] <3500000) :
        e122.iloc[i,2]= e122.iloc[i,2]+(.33*(.2))

#################################### END e = 122 #############################        

############### e  = 123
e123 = pd.DataFrame(np.zeros((773, 3)))
e123.columns = ['a','b','c']




u=123

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e123.iloc[k,0] = dfp.iloc[i,0]
        e123.iloc[k,1] = dfp.iloc[i,1]
        e123.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e123.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e123.iloc[i,0]),4] == df200.iloc[int(e123.iloc[i,0]),4]:
        e123.iloc[i,2]= e123.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e123.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e123.iloc[i,0]),10] < 1 and  df200.iloc[int(e123.iloc[i,0]),10] <1) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e123.iloc[i,0]),10] < 2 and  df200.iloc[int(e123.iloc[i,0]),10] <2) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e123.iloc[i,0]),10] < 3 and  df200.iloc[int(e123.iloc[i,0]),10] <3) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e123.iloc[i,0]),10] < 4 and  df200.iloc[int(e123.iloc[i,0]),10] <4) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e123.iloc[i,0]),10] < 6 and  df200.iloc[int(e123.iloc[i,0]),10] <6) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e123.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e123.iloc[i,0]),5] < 700000 and  df200.iloc[int(e123.iloc[i,0]),5] <700000) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e123.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e123.iloc[i,0]),5] <1400000) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e123.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e123.iloc[i,0]),5] <2100000) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e123.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e123.iloc[i,0]),5] <2800000) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e123.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e123.iloc[i,0]),5] <3500000) :
        e123.iloc[i,2]= e123.iloc[i,2]+(.33*(.2))

#################################### END e = 123  #############################        

############### e  = 124
e124 = pd.DataFrame(np.zeros((773, 3)))
e124.columns = ['a','b','c']




u=124

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e124.iloc[k,0] = dfp.iloc[i,0]
        e124.iloc[k,1] = dfp.iloc[i,1]
        e124.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e124.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e124.iloc[i,0]),4] == df200.iloc[int(e124.iloc[i,0]),4]:
        e124.iloc[i,2]= e124.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e124.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e124.iloc[i,0]),10] < 1 and  df200.iloc[int(e124.iloc[i,0]),10] <1) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e124.iloc[i,0]),10] < 2 and  df200.iloc[int(e124.iloc[i,0]),10] <2) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e124.iloc[i,0]),10] < 3 and  df200.iloc[int(e124.iloc[i,0]),10] <3) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e124.iloc[i,0]),10] < 4 and  df200.iloc[int(e124.iloc[i,0]),10] <4) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e124.iloc[i,0]),10] < 6 and  df200.iloc[int(e124.iloc[i,0]),10] <6) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e124.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e124.iloc[i,0]),5] < 700000 and  df200.iloc[int(e124.iloc[i,0]),5] <700000) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e124.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e124.iloc[i,0]),5] <1400000) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e124.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e124.iloc[i,0]),5] <2100000) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e124.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e124.iloc[i,0]),5] <2800000) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e124.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e124.iloc[i,0]),5] <3500000) :
        e124.iloc[i,2]= e124.iloc[i,2]+(.33*(.2))

#################################### END e = 124  #############################        

############### e  = 125
e125 = pd.DataFrame(np.zeros((773, 3)))
e125.columns = ['a','b','c']




u=125

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e125.iloc[k,0] = dfp.iloc[i,0]
        e125.iloc[k,1] = dfp.iloc[i,1]
        e125.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e125.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e125.iloc[i,0]),4] == df200.iloc[int(e125.iloc[i,0]),4]:
        e125.iloc[i,2]= e125.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e125.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e125.iloc[i,0]),10] < 1 and  df200.iloc[int(e125.iloc[i,0]),10] <1) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e125.iloc[i,0]),10] < 2 and  df200.iloc[int(e125.iloc[i,0]),10] <2) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e125.iloc[i,0]),10] < 3 and  df200.iloc[int(e125.iloc[i,0]),10] <3) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e125.iloc[i,0]),10] < 4 and  df200.iloc[int(e125.iloc[i,0]),10] <4) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e125.iloc[i,0]),10] < 6 and  df200.iloc[int(e125.iloc[i,0]),10] <6) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e125.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e125.iloc[i,0]),5] < 700000 and  df200.iloc[int(e125.iloc[i,0]),5] <700000) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e125.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e125.iloc[i,0]),5] <1400000) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e125.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e125.iloc[i,0]),5] <2100000) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e125.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e125.iloc[i,0]),5] <2800000) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e125.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e125.iloc[i,0]),5] <3500000) :
        e125.iloc[i,2]= e125.iloc[i,2]+(.33*(.2))

#################################### END e = 125  #############################        

############### e  = 126
e126 = pd.DataFrame(np.zeros((773, 3)))
e126.columns = ['a','b','c']




u=126

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e126.iloc[k,0] = dfp.iloc[i,0]
        e126.iloc[k,1] = dfp.iloc[i,1]
        e126.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e126.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e126.iloc[i,0]),4] == df200.iloc[int(e126.iloc[i,0]),4]:
        e126.iloc[i,2]= e126.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e126.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e126.iloc[i,0]),10] < 1 and  df200.iloc[int(e126.iloc[i,0]),10] <1) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e126.iloc[i,0]),10] < 2 and  df200.iloc[int(e126.iloc[i,0]),10] <2) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e126.iloc[i,0]),10] < 3 and  df200.iloc[int(e126.iloc[i,0]),10] <3) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e126.iloc[i,0]),10] < 4 and  df200.iloc[int(e126.iloc[i,0]),10] <4) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e126.iloc[i,0]),10] < 6 and  df200.iloc[int(e126.iloc[i,0]),10] <6) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e126.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e126.iloc[i,0]),5] < 700000 and  df200.iloc[int(e126.iloc[i,0]),5] <700000) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e126.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e126.iloc[i,0]),5] <1400000) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e126.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e126.iloc[i,0]),5] <2100000) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e126.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e126.iloc[i,0]),5] <2800000) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e126.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e126.iloc[i,0]),5] <3500000) :
        e126.iloc[i,2]= e126.iloc[i,2]+(.33*(.2))

#################################### END e = 126  #############################        

############### e  = 127
e127 = pd.DataFrame(np.zeros((773, 3)))
e127.columns = ['a','b','c']




u=127

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e127.iloc[k,0] = dfp.iloc[i,0]
        e127.iloc[k,1] = dfp.iloc[i,1]
        e127.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e127.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e127.iloc[i,0]),4] == df200.iloc[int(e127.iloc[i,0]),4]:
        e127.iloc[i,2]= e127.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e127.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e127.iloc[i,0]),10] < 1 and  df200.iloc[int(e127.iloc[i,0]),10] <1) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e127.iloc[i,0]),10] < 2 and  df200.iloc[int(e127.iloc[i,0]),10] <2) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e127.iloc[i,0]),10] < 3 and  df200.iloc[int(e127.iloc[i,0]),10] <3) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e127.iloc[i,0]),10] < 4 and  df200.iloc[int(e127.iloc[i,0]),10] <4) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e127.iloc[i,0]),10] < 6 and  df200.iloc[int(e127.iloc[i,0]),10] <6) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e127.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e127.iloc[i,0]),5] < 700000 and  df200.iloc[int(e127.iloc[i,0]),5] <700000) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e127.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e127.iloc[i,0]),5] <1400000) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e127.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e127.iloc[i,0]),5] <2100000) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e127.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e127.iloc[i,0]),5] <2800000) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e127.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e127.iloc[i,0]),5] <3500000) :
        e127.iloc[i,2]= e127.iloc[i,2]+(.33*(.2))

#################################### END e = 127  #############################        

############### e  = 128
e128 = pd.DataFrame(np.zeros((773, 3)))
e128.columns = ['a','b','c']




u=128

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e128.iloc[k,0] = dfp.iloc[i,0]
        e128.iloc[k,1] = dfp.iloc[i,1]
        e128.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e128.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e128.iloc[i,0]),4] == df200.iloc[int(e128.iloc[i,0]),4]:
        e128.iloc[i,2]= e128.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e128.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e128.iloc[i,0]),10] < 1 and  df200.iloc[int(e128.iloc[i,0]),10] <1) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e128.iloc[i,0]),10] < 2 and  df200.iloc[int(e128.iloc[i,0]),10] <2) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e128.iloc[i,0]),10] < 3 and  df200.iloc[int(e128.iloc[i,0]),10] <3) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e128.iloc[i,0]),10] < 4 and  df200.iloc[int(e128.iloc[i,0]),10] <4) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e128.iloc[i,0]),10] < 6 and  df200.iloc[int(e128.iloc[i,0]),10] <6) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e128.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e128.iloc[i,0]),5] < 700000 and  df200.iloc[int(e128.iloc[i,0]),5] <700000) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e128.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e128.iloc[i,0]),5] <1400000) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e128.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e128.iloc[i,0]),5] <2100000) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e128.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e128.iloc[i,0]),5] <2800000) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e128.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e128.iloc[i,0]),5] <3500000) :
        e128.iloc[i,2]= e128.iloc[i,2]+(.33*(.2))

#################################### END e = 128 #############################    

############### e  = 129
e129 = pd.DataFrame(np.zeros((773, 3)))
e129.columns = ['a','b','c']




u=129

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e129.iloc[k,0] = dfp.iloc[i,0]
        e129.iloc[k,1] = dfp.iloc[i,1]
        e129.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e129.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e129.iloc[i,0]),4] == df200.iloc[int(e129.iloc[i,0]),4]:
        e129.iloc[i,2]= e129.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e129.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e129.iloc[i,0]),10] < 1 and  df200.iloc[int(e129.iloc[i,0]),10] <1) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e129.iloc[i,0]),10] < 2 and  df200.iloc[int(e129.iloc[i,0]),10] <2) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e129.iloc[i,0]),10] < 3 and  df200.iloc[int(e129.iloc[i,0]),10] <3) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e129.iloc[i,0]),10] < 4 and  df200.iloc[int(e129.iloc[i,0]),10] <4) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e129.iloc[i,0]),10] < 6 and  df200.iloc[int(e129.iloc[i,0]),10] <6) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e129.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e129.iloc[i,0]),5] < 700000 and  df200.iloc[int(e129.iloc[i,0]),5] <700000) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e129.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e129.iloc[i,0]),5] <1400000) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e129.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e129.iloc[i,0]),5] <2100000) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e129.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e129.iloc[i,0]),5] <2800000) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e129.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e129.iloc[i,0]),5] <3500000) :
        e129.iloc[i,2]= e129.iloc[i,2]+(.33*(.2))

#################################### END e = 129 #############################    
    
############### e  = 130
e130 = pd.DataFrame(np.zeros((773, 3)))
e130.columns = ['a','b','c']




u=130

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e130.iloc[k,0] = dfp.iloc[i,0]
        e130.iloc[k,1] = dfp.iloc[i,1]
        e130.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e130.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e130.iloc[i,0]),4] == df200.iloc[int(e130.iloc[i,0]),4]:
        e130.iloc[i,2]= e130.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e130.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e130.iloc[i,0]),10] < 1 and  df200.iloc[int(e130.iloc[i,0]),10] <1) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e130.iloc[i,0]),10] < 2 and  df200.iloc[int(e130.iloc[i,0]),10] <2) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e130.iloc[i,0]),10] < 3 and  df200.iloc[int(e130.iloc[i,0]),10] <3) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e130.iloc[i,0]),10] < 4 and  df200.iloc[int(e130.iloc[i,0]),10] <4) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e130.iloc[i,0]),10] < 6 and  df200.iloc[int(e130.iloc[i,0]),10] <6) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e130.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e130.iloc[i,0]),5] < 700000 and  df200.iloc[int(e130.iloc[i,0]),5] <700000) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e130.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e130.iloc[i,0]),5] <1400000) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e130.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e130.iloc[i,0]),5] <2100000) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e130.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e130.iloc[i,0]),5] <2800000) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e130.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e130.iloc[i,0]),5] <3500000) :
        e130.iloc[i,2]= e130.iloc[i,2]+(.33*(.2))

#################################### END e = 130 #############################    
  
############### e  = 131
e131 = pd.DataFrame(np.zeros((773, 3)))
e131.columns = ['a','b','c']




u=131

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e131.iloc[k,0] = dfp.iloc[i,0]
        e131.iloc[k,1] = dfp.iloc[i,1]
        e131.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e131.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e131.iloc[i,0]),4] == df200.iloc[int(e131.iloc[i,0]),4]:
        e131.iloc[i,2]= e131.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e131.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e131.iloc[i,0]),10] < 1 and  df200.iloc[int(e131.iloc[i,0]),10] <1) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e131.iloc[i,0]),10] < 2 and  df200.iloc[int(e131.iloc[i,0]),10] <2) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e131.iloc[i,0]),10] < 3 and  df200.iloc[int(e131.iloc[i,0]),10] <3) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e131.iloc[i,0]),10] < 4 and  df200.iloc[int(e131.iloc[i,0]),10] <4) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e131.iloc[i,0]),10] < 6 and  df200.iloc[int(e131.iloc[i,0]),10] <6) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e131.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e131.iloc[i,0]),5] < 700000 and  df200.iloc[int(e131.iloc[i,0]),5] <700000) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e131.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e131.iloc[i,0]),5] <1400000) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e131.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e131.iloc[i,0]),5] <2100000) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e131.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e131.iloc[i,0]),5] <2800000) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e131.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e131.iloc[i,0]),5] <3500000) :
        e131.iloc[i,2]= e131.iloc[i,2]+(.33*(.2))

#################################### END e = 131 #############################    
 
############### e  = 132
e132 = pd.DataFrame(np.zeros((773, 3)))
e132.columns = ['a','b','c']




u=132

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e132.iloc[k,0] = dfp.iloc[i,0]
        e132.iloc[k,1] = dfp.iloc[i,1]
        e132.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e132.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e132.iloc[i,0]),4] == df200.iloc[int(e132.iloc[i,0]),4]:
        e132.iloc[i,2]= e132.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e132.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e132.iloc[i,0]),10] < 1 and  df200.iloc[int(e132.iloc[i,0]),10] <1) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e132.iloc[i,0]),10] < 2 and  df200.iloc[int(e132.iloc[i,0]),10] <2) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e132.iloc[i,0]),10] < 3 and  df200.iloc[int(e132.iloc[i,0]),10] <3) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e132.iloc[i,0]),10] < 4 and  df200.iloc[int(e132.iloc[i,0]),10] <4) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e132.iloc[i,0]),10] < 6 and  df200.iloc[int(e132.iloc[i,0]),10] <6) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e132.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e132.iloc[i,0]),5] < 700000 and  df200.iloc[int(e132.iloc[i,0]),5] <700000) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e132.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e132.iloc[i,0]),5] <1400000) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e132.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e132.iloc[i,0]),5] <2100000) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e132.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e132.iloc[i,0]),5] <2800000) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e132.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e132.iloc[i,0]),5] <3500000) :
        e132.iloc[i,2]= e132.iloc[i,2]+(.33*(.2))

#################################### END e = 132 #############################    
    

############### e  = 133
e133 = pd.DataFrame(np.zeros((773, 3)))
e133.columns = ['a','b','c']




u=133

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e133.iloc[k,0] = dfp.iloc[i,0]
        e133.iloc[k,1] = dfp.iloc[i,1]
        e133.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e133.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e133.iloc[i,0]),4] == df200.iloc[int(e133.iloc[i,0]),4]:
        e133.iloc[i,2]= e133.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e133.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e133.iloc[i,0]),10] < 1 and  df200.iloc[int(e133.iloc[i,0]),10] <1) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e133.iloc[i,0]),10] < 2 and  df200.iloc[int(e133.iloc[i,0]),10] <2) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e133.iloc[i,0]),10] < 3 and  df200.iloc[int(e133.iloc[i,0]),10] <3) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e133.iloc[i,0]),10] < 4 and  df200.iloc[int(e133.iloc[i,0]),10] <4) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e133.iloc[i,0]),10] < 6 and  df200.iloc[int(e133.iloc[i,0]),10] <6) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e133.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e133.iloc[i,0]),5] < 700000 and  df200.iloc[int(e133.iloc[i,0]),5] <700000) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e133.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e133.iloc[i,0]),5] <1400000) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e133.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e133.iloc[i,0]),5] <2100000) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e133.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e133.iloc[i,0]),5] <2800000) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e133.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e133.iloc[i,0]),5] <3500000) :
        e133.iloc[i,2]= e133.iloc[i,2]+(.33*(.2))

#################################### END e = 133 #############################    
    
############### e  = 134
e134 = pd.DataFrame(np.zeros((773, 3)))
e134.columns = ['a','b','c']




u=134

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e134.iloc[k,0] = dfp.iloc[i,0]
        e134.iloc[k,1] = dfp.iloc[i,1]
        e134.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e134.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e134.iloc[i,0]),4] == df200.iloc[int(e134.iloc[i,0]),4]:
        e134.iloc[i,2]= e134.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e134.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e134.iloc[i,0]),10] < 1 and  df200.iloc[int(e134.iloc[i,0]),10] <1) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e134.iloc[i,0]),10] < 2 and  df200.iloc[int(e134.iloc[i,0]),10] <2) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e134.iloc[i,0]),10] < 3 and  df200.iloc[int(e134.iloc[i,0]),10] <3) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e134.iloc[i,0]),10] < 4 and  df200.iloc[int(e134.iloc[i,0]),10] <4) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e134.iloc[i,0]),10] < 6 and  df200.iloc[int(e134.iloc[i,0]),10] <6) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e134.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e134.iloc[i,0]),5] < 700000 and  df200.iloc[int(e134.iloc[i,0]),5] <700000) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e134.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e134.iloc[i,0]),5] <1400000) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e134.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e134.iloc[i,0]),5] <2100000) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e134.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e134.iloc[i,0]),5] <2800000) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e134.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e134.iloc[i,0]),5] <3500000) :
        e134.iloc[i,2]= e134.iloc[i,2]+(.33*(.2))

#################################### END e = 134 #############################    
    
    
############### e  = 135
e135 = pd.DataFrame(np.zeros((773, 3)))
e135.columns = ['a','b','c']




u=135

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e135.iloc[k,0] = dfp.iloc[i,0]
        e135.iloc[k,1] = dfp.iloc[i,1]
        e135.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e135.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e135.iloc[i,0]),4] == df200.iloc[int(e135.iloc[i,0]),4]:
        e135.iloc[i,2]= e135.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e135.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e135.iloc[i,0]),10] < 1 and  df200.iloc[int(e135.iloc[i,0]),10] <1) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e135.iloc[i,0]),10] < 2 and  df200.iloc[int(e135.iloc[i,0]),10] <2) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e135.iloc[i,0]),10] < 3 and  df200.iloc[int(e135.iloc[i,0]),10] <3) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e135.iloc[i,0]),10] < 4 and  df200.iloc[int(e135.iloc[i,0]),10] <4) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e135.iloc[i,0]),10] < 6 and  df200.iloc[int(e135.iloc[i,0]),10] <6) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e135.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e135.iloc[i,0]),5] < 700000 and  df200.iloc[int(e135.iloc[i,0]),5] <700000) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e135.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e135.iloc[i,0]),5] <1400000) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e135.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e135.iloc[i,0]),5] <2100000) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e135.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e135.iloc[i,0]),5] <2800000) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e135.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e135.iloc[i,0]),5] <3500000) :
        e135.iloc[i,2]= e135.iloc[i,2]+(.33*(.2))

#################################### END e = 135 #############################    
    
    
############### e  = 136
e136 = pd.DataFrame(np.zeros((773, 3)))
e136.columns = ['a','b','c']




u=136

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e136.iloc[k,0] = dfp.iloc[i,0]
        e136.iloc[k,1] = dfp.iloc[i,1]
        e136.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e136.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e136.iloc[i,0]),4] == df200.iloc[int(e136.iloc[i,0]),4]:
        e136.iloc[i,2]= e136.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e136.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e136.iloc[i,0]),10] < 1 and  df200.iloc[int(e136.iloc[i,0]),10] <1) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e136.iloc[i,0]),10] < 2 and  df200.iloc[int(e136.iloc[i,0]),10] <2) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e136.iloc[i,0]),10] < 3 and  df200.iloc[int(e136.iloc[i,0]),10] <3) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e136.iloc[i,0]),10] < 4 and  df200.iloc[int(e136.iloc[i,0]),10] <4) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e136.iloc[i,0]),10] < 6 and  df200.iloc[int(e136.iloc[i,0]),10] <6) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e136.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e136.iloc[i,0]),5] < 700000 and  df200.iloc[int(e136.iloc[i,0]),5] <700000) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e136.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e136.iloc[i,0]),5] <1400000) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e136.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e136.iloc[i,0]),5] <2100000) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e136.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e136.iloc[i,0]),5] <2800000) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e136.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e136.iloc[i,0]),5] <3500000) :
        e136.iloc[i,2]= e136.iloc[i,2]+(.33*(.2))

#################################### END e = 136 #############################    
    
    
############### e  = 137
e137 = pd.DataFrame(np.zeros((773, 3)))
e137.columns = ['a','b','c']




u=137

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e137.iloc[k,0] = dfp.iloc[i,0]
        e137.iloc[k,1] = dfp.iloc[i,1]
        e137.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e137.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e137.iloc[i,0]),4] == df200.iloc[int(e137.iloc[i,0]),4]:
        e137.iloc[i,2]= e137.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e137.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e137.iloc[i,0]),10] < 1 and  df200.iloc[int(e137.iloc[i,0]),10] <1) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e137.iloc[i,0]),10] < 2 and  df200.iloc[int(e137.iloc[i,0]),10] <2) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e137.iloc[i,0]),10] < 3 and  df200.iloc[int(e137.iloc[i,0]),10] <3) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e137.iloc[i,0]),10] < 4 and  df200.iloc[int(e137.iloc[i,0]),10] <4) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e137.iloc[i,0]),10] < 6 and  df200.iloc[int(e137.iloc[i,0]),10] <6) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e137.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e137.iloc[i,0]),5] < 700000 and  df200.iloc[int(e137.iloc[i,0]),5] <700000) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e137.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e137.iloc[i,0]),5] <1400000) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e137.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e137.iloc[i,0]),5] <2100000) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e137.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e137.iloc[i,0]),5] <2800000) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e137.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e137.iloc[i,0]),5] <3500000) :
        e137.iloc[i,2]= e137.iloc[i,2]+(.33*(.2))

#################################### END e = 137 #############################    
    
    
############### e  = 138
e138 = pd.DataFrame(np.zeros((773, 3)))
e138.columns = ['a','b','c']




u=138

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e138.iloc[k,0] = dfp.iloc[i,0]
        e138.iloc[k,1] = dfp.iloc[i,1]
        e138.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e138.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e138.iloc[i,0]),4] == df200.iloc[int(e138.iloc[i,0]),4]:
        e138.iloc[i,2]= e138.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e138.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e138.iloc[i,0]),10] < 1 and  df200.iloc[int(e138.iloc[i,0]),10] <1) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e138.iloc[i,0]),10] < 2 and  df200.iloc[int(e138.iloc[i,0]),10] <2) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e138.iloc[i,0]),10] < 3 and  df200.iloc[int(e138.iloc[i,0]),10] <3) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e138.iloc[i,0]),10] < 4 and  df200.iloc[int(e138.iloc[i,0]),10] <4) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e138.iloc[i,0]),10] < 6 and  df200.iloc[int(e138.iloc[i,0]),10] <6) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e138.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e138.iloc[i,0]),5] < 700000 and  df200.iloc[int(e138.iloc[i,0]),5] <700000) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e138.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e138.iloc[i,0]),5] <1400000) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e138.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e138.iloc[i,0]),5] <2100000) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e138.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e138.iloc[i,0]),5] <2800000) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e138.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e138.iloc[i,0]),5] <3500000) :
        e138.iloc[i,2]= e138.iloc[i,2]+(.33*(.2))

#################################### END e = 138 #############################    
    

 ############### e  = 139
e139 = pd.DataFrame(np.zeros((773, 3)))
e139.columns = ['a','b','c']




u=139

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e139.iloc[k,0] = dfp.iloc[i,0]
        e139.iloc[k,1] = dfp.iloc[i,1]
        e139.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e139.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e139.iloc[i,0]),4] == df200.iloc[int(e139.iloc[i,0]),4]:
        e139.iloc[i,2]= e139.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e139.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e139.iloc[i,0]),10] < 1 and  df200.iloc[int(e139.iloc[i,0]),10] <1) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e139.iloc[i,0]),10] < 2 and  df200.iloc[int(e139.iloc[i,0]),10] <2) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e139.iloc[i,0]),10] < 3 and  df200.iloc[int(e139.iloc[i,0]),10] <3) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e139.iloc[i,0]),10] < 4 and  df200.iloc[int(e139.iloc[i,0]),10] <4) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e139.iloc[i,0]),10] < 6 and  df200.iloc[int(e139.iloc[i,0]),10] <6) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e139.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e139.iloc[i,0]),5] < 700000 and  df200.iloc[int(e139.iloc[i,0]),5] <700000) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e139.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e139.iloc[i,0]),5] <1400000) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e139.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e139.iloc[i,0]),5] <2100000) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e139.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e139.iloc[i,0]),5] <2800000) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e139.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e139.iloc[i,0]),5] <3500000) :
        e139.iloc[i,2]= e139.iloc[i,2]+(.33*(.2))

#################################### END e = 139 #############################    
       
############### e  = 140
e140 = pd.DataFrame(np.zeros((773, 3)))
e140.columns = ['a','b','c']




u=140

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e140.iloc[k,0] = dfp.iloc[i,0]
        e140.iloc[k,1] = dfp.iloc[i,1]
        e140.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e140.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e140.iloc[i,0]),4] == df200.iloc[int(e140.iloc[i,0]),4]:
        e140.iloc[i,2]= e140.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e140.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e140.iloc[i,0]),10] < 1 and  df200.iloc[int(e140.iloc[i,0]),10] <1) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e140.iloc[i,0]),10] < 2 and  df200.iloc[int(e140.iloc[i,0]),10] <2) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e140.iloc[i,0]),10] < 3 and  df200.iloc[int(e140.iloc[i,0]),10] <3) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e140.iloc[i,0]),10] < 4 and  df200.iloc[int(e140.iloc[i,0]),10] <4) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e140.iloc[i,0]),10] < 6 and  df200.iloc[int(e140.iloc[i,0]),10] <6) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e140.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e140.iloc[i,0]),5] < 700000 and  df200.iloc[int(e140.iloc[i,0]),5] <700000) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e140.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e140.iloc[i,0]),5] <1400000) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e140.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e140.iloc[i,0]),5] <2100000) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e140.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e140.iloc[i,0]),5] <2800000) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e140.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e140.iloc[i,0]),5] <3500000) :
        e140.iloc[i,2]= e140.iloc[i,2]+(.33*(.2))

#################################### END e = 140 #############################    
 
############### e  = 141
e141 = pd.DataFrame(np.zeros((773, 3)))
e141.columns = ['a','b','c']




u=141

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e141.iloc[k,0] = dfp.iloc[i,0]
        e141.iloc[k,1] = dfp.iloc[i,1]
        e141.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e141.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e141.iloc[i,0]),4] == df200.iloc[int(e141.iloc[i,0]),4]:
        e141.iloc[i,2]= e141.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e141.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e141.iloc[i,0]),10] < 1 and  df200.iloc[int(e141.iloc[i,0]),10] <1) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e141.iloc[i,0]),10] < 2 and  df200.iloc[int(e141.iloc[i,0]),10] <2) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e141.iloc[i,0]),10] < 3 and  df200.iloc[int(e141.iloc[i,0]),10] <3) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e141.iloc[i,0]),10] < 4 and  df200.iloc[int(e141.iloc[i,0]),10] <4) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e141.iloc[i,0]),10] < 6 and  df200.iloc[int(e141.iloc[i,0]),10] <6) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e141.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e141.iloc[i,0]),5] < 700000 and  df200.iloc[int(e141.iloc[i,0]),5] <700000) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e141.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e141.iloc[i,0]),5] <1400000) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e141.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e141.iloc[i,0]),5] <2100000) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e141.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e141.iloc[i,0]),5] <2800000) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e141.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e141.iloc[i,0]),5] <3500000) :
        e141.iloc[i,2]= e141.iloc[i,2]+(.33*(.2))

#################################### END e = 141#############################    
           
  
############### e  = 142
e142 = pd.DataFrame(np.zeros((773, 3)))
e142.columns = ['a','b','c']




u=142

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e142.iloc[k,0] = dfp.iloc[i,0]
        e142.iloc[k,1] = dfp.iloc[i,1]
        e142.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e142.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e142.iloc[i,0]),4] == df200.iloc[int(e142.iloc[i,0]),4]:
        e142.iloc[i,2]= e142.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e142.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e142.iloc[i,0]),10] < 1 and  df200.iloc[int(e142.iloc[i,0]),10] <1) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e142.iloc[i,0]),10] < 2 and  df200.iloc[int(e142.iloc[i,0]),10] <2) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e142.iloc[i,0]),10] < 3 and  df200.iloc[int(e142.iloc[i,0]),10] <3) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e142.iloc[i,0]),10] < 4 and  df200.iloc[int(e142.iloc[i,0]),10] <4) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e142.iloc[i,0]),10] < 6 and  df200.iloc[int(e142.iloc[i,0]),10] <6) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e142.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e142.iloc[i,0]),5] < 700000 and  df200.iloc[int(e142.iloc[i,0]),5] <700000) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e142.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e142.iloc[i,0]),5] <1400000) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e142.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e142.iloc[i,0]),5] <2100000) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e142.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e142.iloc[i,0]),5] <2800000) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e142.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e142.iloc[i,0]),5] <3500000) :
        e142.iloc[i,2]= e142.iloc[i,2]+(.33*(.2))

#################################### END e = 142 #############################    
  
############### e  = 143
e143 = pd.DataFrame(np.zeros((773, 3)))
e143.columns = ['a','b','c']




u=143

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e143.iloc[k,0] = dfp.iloc[i,0]
        e143.iloc[k,1] = dfp.iloc[i,1]
        e143.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e143.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e143.iloc[i,0]),4] == df200.iloc[int(e143.iloc[i,0]),4]:
        e143.iloc[i,2]= e143.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e143.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e143.iloc[i,0]),10] < 1 and  df200.iloc[int(e143.iloc[i,0]),10] <1) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e143.iloc[i,0]),10] < 2 and  df200.iloc[int(e143.iloc[i,0]),10] <2) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e143.iloc[i,0]),10] < 3 and  df200.iloc[int(e143.iloc[i,0]),10] <3) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e143.iloc[i,0]),10] < 4 and  df200.iloc[int(e143.iloc[i,0]),10] <4) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e143.iloc[i,0]),10] < 6 and  df200.iloc[int(e143.iloc[i,0]),10] <6) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e143.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e143.iloc[i,0]),5] < 700000 and  df200.iloc[int(e143.iloc[i,0]),5] <700000) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e143.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e143.iloc[i,0]),5] <1400000) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e143.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e143.iloc[i,0]),5] <2100000) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e143.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e143.iloc[i,0]),5] <2800000) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e143.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e143.iloc[i,0]),5] <3500000) :
        e143.iloc[i,2]= e143.iloc[i,2]+(.33*(.2))

#################################### END e = 143 #############################    
    
############### e  = 144
e144 = pd.DataFrame(np.zeros((773, 3)))
e144.columns = ['a','b','c']




u=144

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e144.iloc[k,0] = dfp.iloc[i,0]
        e144.iloc[k,1] = dfp.iloc[i,1]
        e144.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e144.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e144.iloc[i,0]),4] == df200.iloc[int(e144.iloc[i,0]),4]:
        e144.iloc[i,2]= e144.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e144.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e144.iloc[i,0]),10] < 1 and  df200.iloc[int(e144.iloc[i,0]),10] <1) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e144.iloc[i,0]),10] < 2 and  df200.iloc[int(e144.iloc[i,0]),10] <2) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e144.iloc[i,0]),10] < 3 and  df200.iloc[int(e144.iloc[i,0]),10] <3) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e144.iloc[i,0]),10] < 4 and  df200.iloc[int(e144.iloc[i,0]),10] <4) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e144.iloc[i,0]),10] < 6 and  df200.iloc[int(e144.iloc[i,0]),10] <6) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e144.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e144.iloc[i,0]),5] < 700000 and  df200.iloc[int(e144.iloc[i,0]),5] <700000) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e144.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e144.iloc[i,0]),5] <1400000) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e144.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e144.iloc[i,0]),5] <2100000) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e144.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e144.iloc[i,0]),5] <2800000) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e144.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e144.iloc[i,0]),5] <3500000) :
        e144.iloc[i,2]= e144.iloc[i,2]+(.33*(.2))

#################################### END e = 144 #############################    
    

 ############### e  = 145
e145 = pd.DataFrame(np.zeros((773, 3)))
e145.columns = ['a','b','c']




u=145

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e145.iloc[k,0] = dfp.iloc[i,0]
        e145.iloc[k,1] = dfp.iloc[i,1]
        e145.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e145.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e145.iloc[i,0]),4] == df200.iloc[int(e145.iloc[i,0]),4]:
        e145.iloc[i,2]= e145.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e145.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e145.iloc[i,0]),10] < 1 and  df200.iloc[int(e145.iloc[i,0]),10] <1) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e145.iloc[i,0]),10] < 2 and  df200.iloc[int(e145.iloc[i,0]),10] <2) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e145.iloc[i,0]),10] < 3 and  df200.iloc[int(e145.iloc[i,0]),10] <3) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e145.iloc[i,0]),10] < 4 and  df200.iloc[int(e145.iloc[i,0]),10] <4) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e145.iloc[i,0]),10] < 6 and  df200.iloc[int(e145.iloc[i,0]),10] <6) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e145.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e145.iloc[i,0]),5] < 700000 and  df200.iloc[int(e145.iloc[i,0]),5] <700000) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e145.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e145.iloc[i,0]),5] <1400000) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e145.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e145.iloc[i,0]),5] <2100000) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e145.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e145.iloc[i,0]),5] <2800000) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e145.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e145.iloc[i,0]),5] <3500000) :
        e145.iloc[i,2]= e145.iloc[i,2]+(.33*(.2))

#################################### END e = 145 #############################    
         
############### e  = 146
e146 = pd.DataFrame(np.zeros((773, 3)))
e146.columns = ['a','b','c']




u=146

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e146.iloc[k,0] = dfp.iloc[i,0]
        e146.iloc[k,1] = dfp.iloc[i,1]
        e146.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e146.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e146.iloc[i,0]),4] == df200.iloc[int(e146.iloc[i,0]),4]:
        e146.iloc[i,2]= e146.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e146.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e146.iloc[i,0]),10] < 1 and  df200.iloc[int(e146.iloc[i,0]),10] <1) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e146.iloc[i,0]),10] < 2 and  df200.iloc[int(e146.iloc[i,0]),10] <2) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e146.iloc[i,0]),10] < 3 and  df200.iloc[int(e146.iloc[i,0]),10] <3) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e146.iloc[i,0]),10] < 4 and  df200.iloc[int(e146.iloc[i,0]),10] <4) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e146.iloc[i,0]),10] < 6 and  df200.iloc[int(e146.iloc[i,0]),10] <6) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e146.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e146.iloc[i,0]),5] < 700000 and  df200.iloc[int(e146.iloc[i,0]),5] <700000) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e146.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e146.iloc[i,0]),5] <1400000) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e146.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e146.iloc[i,0]),5] <2100000) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e146.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e146.iloc[i,0]),5] <2800000) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e146.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e146.iloc[i,0]),5] <3500000) :
        e146.iloc[i,2]= e146.iloc[i,2]+(.33*(.2))

#################################### END e = 146 #############################    
    

############### e  = 147
e147 = pd.DataFrame(np.zeros((773, 3)))
e147.columns = ['a','b','c']




u=147

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e147.iloc[k,0] = dfp.iloc[i,0]
        e147.iloc[k,1] = dfp.iloc[i,1]
        e147.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e147.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e147.iloc[i,0]),4] == df200.iloc[int(e147.iloc[i,0]),4]:
        e147.iloc[i,2]= e147.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e147.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e147.iloc[i,0]),10] < 1 and  df200.iloc[int(e147.iloc[i,0]),10] <1) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e147.iloc[i,0]),10] < 2 and  df200.iloc[int(e147.iloc[i,0]),10] <2) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e147.iloc[i,0]),10] < 3 and  df200.iloc[int(e147.iloc[i,0]),10] <3) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e147.iloc[i,0]),10] < 4 and  df200.iloc[int(e147.iloc[i,0]),10] <4) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e147.iloc[i,0]),10] < 6 and  df200.iloc[int(e147.iloc[i,0]),10] <6) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e147.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e147.iloc[i,0]),5] < 700000 and  df200.iloc[int(e147.iloc[i,0]),5] <700000) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e147.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e147.iloc[i,0]),5] <1400000) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e147.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e147.iloc[i,0]),5] <2100000) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e147.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e147.iloc[i,0]),5] <2800000) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e147.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e147.iloc[i,0]),5] <3500000) :
        e147.iloc[i,2]= e147.iloc[i,2]+(.33*(.2))

#################################### END e = 147 #############################    
    

############### e  = 148
e148 = pd.DataFrame(np.zeros((773, 3)))
e148.columns = ['a','b','c']




u=148

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e148.iloc[k,0] = dfp.iloc[i,0]
        e148.iloc[k,1] = dfp.iloc[i,1]
        e148.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e148.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e148.iloc[i,0]),4] == df200.iloc[int(e148.iloc[i,0]),4]:
        e148.iloc[i,2]= e148.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e148.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e148.iloc[i,0]),10] < 1 and  df200.iloc[int(e148.iloc[i,0]),10] <1) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e148.iloc[i,0]),10] < 2 and  df200.iloc[int(e148.iloc[i,0]),10] <2) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e148.iloc[i,0]),10] < 3 and  df200.iloc[int(e148.iloc[i,0]),10] <3) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e148.iloc[i,0]),10] < 4 and  df200.iloc[int(e148.iloc[i,0]),10] <4) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e148.iloc[i,0]),10] < 6 and  df200.iloc[int(e148.iloc[i,0]),10] <6) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e148.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e148.iloc[i,0]),5] < 700000 and  df200.iloc[int(e148.iloc[i,0]),5] <700000) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e148.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e148.iloc[i,0]),5] <1400000) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e148.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e148.iloc[i,0]),5] <2100000) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e148.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e148.iloc[i,0]),5] <2800000) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e148.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e148.iloc[i,0]),5] <3500000) :
        e148.iloc[i,2]= e148.iloc[i,2]+(.33*(.2))

#################################### END e = 148 #############################    
    

############### e  = 149
e149 = pd.DataFrame(np.zeros((773, 3)))
e149.columns = ['a','b','c']




u=149

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e149.iloc[k,0] = dfp.iloc[i,0]
        e149.iloc[k,1] = dfp.iloc[i,1]
        e149.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e149.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e149.iloc[i,0]),4] == df200.iloc[int(e149.iloc[i,0]),4]:
        e149.iloc[i,2]= e149.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e149.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e149.iloc[i,0]),10] < 1 and  df200.iloc[int(e149.iloc[i,0]),10] <1) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e149.iloc[i,0]),10] < 2 and  df200.iloc[int(e149.iloc[i,0]),10] <2) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e149.iloc[i,0]),10] < 3 and  df200.iloc[int(e149.iloc[i,0]),10] <3) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e149.iloc[i,0]),10] < 4 and  df200.iloc[int(e149.iloc[i,0]),10] <4) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e149.iloc[i,0]),10] < 6 and  df200.iloc[int(e149.iloc[i,0]),10] <6) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e149.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e149.iloc[i,0]),5] < 700000 and  df200.iloc[int(e149.iloc[i,0]),5] <700000) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e149.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e149.iloc[i,0]),5] <1400000) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e149.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e149.iloc[i,0]),5] <2100000) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e149.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e149.iloc[i,0]),5] <2800000) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e149.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e149.iloc[i,0]),5] <3500000) :
        e149.iloc[i,2]= e149.iloc[i,2]+(.33*(.2))

#################################### END e = 149 #############################    
    
############### e  = 150
e150 = pd.DataFrame(np.zeros((773, 3)))
e150.columns = ['a','b','c']




u=150

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e150.iloc[k,0] = dfp.iloc[i,0]
        e150.iloc[k,1] = dfp.iloc[i,1]
        e150.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e150.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e150.iloc[i,0]),4] == df200.iloc[int(e150.iloc[i,0]),4]:
        e150.iloc[i,2]= e150.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e150.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e150.iloc[i,0]),10] < 1 and  df200.iloc[int(e150.iloc[i,0]),10] <1) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e150.iloc[i,0]),10] < 2 and  df200.iloc[int(e150.iloc[i,0]),10] <2) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e150.iloc[i,0]),10] < 3 and  df200.iloc[int(e150.iloc[i,0]),10] <3) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e150.iloc[i,0]),10] < 4 and  df200.iloc[int(e150.iloc[i,0]),10] <4) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e150.iloc[i,0]),10] < 6 and  df200.iloc[int(e150.iloc[i,0]),10] <6) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e150.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e150.iloc[i,0]),5] < 700000 and  df200.iloc[int(e150.iloc[i,0]),5] <700000) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e150.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e150.iloc[i,0]),5] <1400000) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e150.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e150.iloc[i,0]),5] <2100000) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e150.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e150.iloc[i,0]),5] <2800000) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e150.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e150.iloc[i,0]),5] <3500000) :
        e150.iloc[i,2]= e150.iloc[i,2]+(.33*(.2))

#################################### END e = 150 #############################    
    

############### e  = 151
e151 = pd.DataFrame(np.zeros((773, 3)))
e151.columns = ['a','b','c']




u=151

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e151.iloc[k,0] = dfp.iloc[i,0]
        e151.iloc[k,1] = dfp.iloc[i,1]
        e151.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e151.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e151.iloc[i,0]),4] == df200.iloc[int(e151.iloc[i,0]),4]:
        e151.iloc[i,2]= e151.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e151.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e151.iloc[i,0]),10] < 1 and  df200.iloc[int(e151.iloc[i,0]),10] <1) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e151.iloc[i,0]),10] < 2 and  df200.iloc[int(e151.iloc[i,0]),10] <2) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e151.iloc[i,0]),10] < 3 and  df200.iloc[int(e151.iloc[i,0]),10] <3) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e151.iloc[i,0]),10] < 4 and  df200.iloc[int(e151.iloc[i,0]),10] <4) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e151.iloc[i,0]),10] < 6 and  df200.iloc[int(e151.iloc[i,0]),10] <6) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e151.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e151.iloc[i,0]),5] < 700000 and  df200.iloc[int(e151.iloc[i,0]),5] <700000) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e151.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e151.iloc[i,0]),5] <1400000) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e151.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e151.iloc[i,0]),5] <2100000) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e151.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e151.iloc[i,0]),5] <2800000) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e151.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e151.iloc[i,0]),5] <3500000) :
        e151.iloc[i,2]= e151.iloc[i,2]+(.33*(.2))

#################################### END e = 151 #############################    
 
############### e  = 152
e152 = pd.DataFrame(np.zeros((773, 3)))
e152.columns = ['a','b','c']




u=152

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e152.iloc[k,0] = dfp.iloc[i,0]
        e152.iloc[k,1] = dfp.iloc[i,1]
        e152.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e152.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e152.iloc[i,0]),4] == df200.iloc[int(e152.iloc[i,0]),4]:
        e152.iloc[i,2]= e152.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e152.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e152.iloc[i,0]),10] < 1 and  df200.iloc[int(e152.iloc[i,0]),10] <1) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e152.iloc[i,0]),10] < 2 and  df200.iloc[int(e152.iloc[i,0]),10] <2) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e152.iloc[i,0]),10] < 3 and  df200.iloc[int(e152.iloc[i,0]),10] <3) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e152.iloc[i,0]),10] < 4 and  df200.iloc[int(e152.iloc[i,0]),10] <4) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e152.iloc[i,0]),10] < 6 and  df200.iloc[int(e152.iloc[i,0]),10] <6) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e152.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e152.iloc[i,0]),5] < 700000 and  df200.iloc[int(e152.iloc[i,0]),5] <700000) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e152.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e152.iloc[i,0]),5] <1400000) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e152.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e152.iloc[i,0]),5] <2100000) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e152.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e152.iloc[i,0]),5] <2800000) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e152.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e152.iloc[i,0]),5] <3500000) :
        e152.iloc[i,2]= e152.iloc[i,2]+(.33*(.2))

#################################### END e = 152 #############################    
   
############### e  = 153
e153 = pd.DataFrame(np.zeros((773, 3)))
e153.columns = ['a','b','c']




u=153

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e153.iloc[k,0] = dfp.iloc[i,0]
        e153.iloc[k,1] = dfp.iloc[i,1]
        e153.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e153.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e153.iloc[i,0]),4] == df200.iloc[int(e153.iloc[i,0]),4]:
        e153.iloc[i,2]= e153.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e153.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e153.iloc[i,0]),10] < 1 and  df200.iloc[int(e153.iloc[i,0]),10] <1) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e153.iloc[i,0]),10] < 2 and  df200.iloc[int(e153.iloc[i,0]),10] <2) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e153.iloc[i,0]),10] < 3 and  df200.iloc[int(e153.iloc[i,0]),10] <3) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e153.iloc[i,0]),10] < 4 and  df200.iloc[int(e153.iloc[i,0]),10] <4) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e153.iloc[i,0]),10] < 6 and  df200.iloc[int(e153.iloc[i,0]),10] <6) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e153.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e153.iloc[i,0]),5] < 700000 and  df200.iloc[int(e153.iloc[i,0]),5] <700000) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e153.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e153.iloc[i,0]),5] <1400000) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e153.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e153.iloc[i,0]),5] <2100000) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e153.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e153.iloc[i,0]),5] <2800000) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e153.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e153.iloc[i,0]),5] <3500000) :
        e153.iloc[i,2]= e153.iloc[i,2]+(.33*(.2))

#################################### END e = 153 #############################    
            
############### e  = 154
e154 = pd.DataFrame(np.zeros((773, 3)))
e154.columns = ['a','b','c']




u=154

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e154.iloc[k,0] = dfp.iloc[i,0]
        e154.iloc[k,1] = dfp.iloc[i,1]
        e154.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e154.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e154.iloc[i,0]),4] == df200.iloc[int(e154.iloc[i,0]),4]:
        e154.iloc[i,2]= e154.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e154.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e154.iloc[i,0]),10] < 1 and  df200.iloc[int(e154.iloc[i,0]),10] <1) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e154.iloc[i,0]),10] < 2 and  df200.iloc[int(e154.iloc[i,0]),10] <2) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e154.iloc[i,0]),10] < 3 and  df200.iloc[int(e154.iloc[i,0]),10] <3) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e154.iloc[i,0]),10] < 4 and  df200.iloc[int(e154.iloc[i,0]),10] <4) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e154.iloc[i,0]),10] < 6 and  df200.iloc[int(e154.iloc[i,0]),10] <6) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e154.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e154.iloc[i,0]),5] < 700000 and  df200.iloc[int(e154.iloc[i,0]),5] <700000) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e154.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e154.iloc[i,0]),5] <1400000) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e154.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e154.iloc[i,0]),5] <2100000) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e154.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e154.iloc[i,0]),5] <2800000) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e154.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e154.iloc[i,0]),5] <3500000) :
        e154.iloc[i,2]= e154.iloc[i,2]+(.33*(.2))

#################################### END e = 154 #############################    
    

############### e  = 155
e155 = pd.DataFrame(np.zeros((773, 3)))
e155.columns = ['a','b','c']




u=155

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e155.iloc[k,0] = dfp.iloc[i,0]
        e155.iloc[k,1] = dfp.iloc[i,1]
        e155.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e155.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e155.iloc[i,0]),4] == df200.iloc[int(e155.iloc[i,0]),4]:
        e155.iloc[i,2]= e155.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e155.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e155.iloc[i,0]),10] < 1 and  df200.iloc[int(e155.iloc[i,0]),10] <1) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e155.iloc[i,0]),10] < 2 and  df200.iloc[int(e155.iloc[i,0]),10] <2) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e155.iloc[i,0]),10] < 3 and  df200.iloc[int(e155.iloc[i,0]),10] <3) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e155.iloc[i,0]),10] < 4 and  df200.iloc[int(e155.iloc[i,0]),10] <4) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e155.iloc[i,0]),10] < 6 and  df200.iloc[int(e155.iloc[i,0]),10] <6) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e155.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e155.iloc[i,0]),5] < 700000 and  df200.iloc[int(e155.iloc[i,0]),5] <700000) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e155.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e155.iloc[i,0]),5] <1400000) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e155.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e155.iloc[i,0]),5] <2100000) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e155.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e155.iloc[i,0]),5] <2800000) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e155.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e155.iloc[i,0]),5] <3500000) :
        e155.iloc[i,2]= e155.iloc[i,2]+(.33*(.2))

#################################### END e = 155 #############################    
    

############### e  = 156
e156 = pd.DataFrame(np.zeros((773, 3)))
e156.columns = ['a','b','c']




u=156

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e156.iloc[k,0] = dfp.iloc[i,0]
        e156.iloc[k,1] = dfp.iloc[i,1]
        e156.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e156.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e156.iloc[i,0]),4] == df200.iloc[int(e156.iloc[i,0]),4]:
        e156.iloc[i,2]= e156.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e156.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e156.iloc[i,0]),10] < 1 and  df200.iloc[int(e156.iloc[i,0]),10] <1) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e156.iloc[i,0]),10] < 2 and  df200.iloc[int(e156.iloc[i,0]),10] <2) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e156.iloc[i,0]),10] < 3 and  df200.iloc[int(e156.iloc[i,0]),10] <3) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e156.iloc[i,0]),10] < 4 and  df200.iloc[int(e156.iloc[i,0]),10] <4) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e156.iloc[i,0]),10] < 6 and  df200.iloc[int(e156.iloc[i,0]),10] <6) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e156.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e156.iloc[i,0]),5] < 700000 and  df200.iloc[int(e156.iloc[i,0]),5] <700000) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e156.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e156.iloc[i,0]),5] <1400000) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e156.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e156.iloc[i,0]),5] <2100000) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e156.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e156.iloc[i,0]),5] <2800000) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e156.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e156.iloc[i,0]),5] <3500000) :
        e156.iloc[i,2]= e156.iloc[i,2]+(.33*(.2))

#################################### END e = 156 #############################    
    

############### e  = 157
e157 = pd.DataFrame(np.zeros((773, 3)))
e157.columns = ['a','b','c']




u=157

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e157.iloc[k,0] = dfp.iloc[i,0]
        e157.iloc[k,1] = dfp.iloc[i,1]
        e157.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e157.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e157.iloc[i,0]),4] == df200.iloc[int(e157.iloc[i,0]),4]:
        e157.iloc[i,2]= e157.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e157.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e157.iloc[i,0]),10] < 1 and  df200.iloc[int(e157.iloc[i,0]),10] <1) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e157.iloc[i,0]),10] < 2 and  df200.iloc[int(e157.iloc[i,0]),10] <2) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e157.iloc[i,0]),10] < 3 and  df200.iloc[int(e157.iloc[i,0]),10] <3) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e157.iloc[i,0]),10] < 4 and  df200.iloc[int(e157.iloc[i,0]),10] <4) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e157.iloc[i,0]),10] < 6 and  df200.iloc[int(e157.iloc[i,0]),10] <6) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e157.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e157.iloc[i,0]),5] < 700000 and  df200.iloc[int(e157.iloc[i,0]),5] <700000) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e157.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e157.iloc[i,0]),5] <1400000) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e157.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e157.iloc[i,0]),5] <2100000) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e157.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e157.iloc[i,0]),5] <2800000) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e157.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e157.iloc[i,0]),5] <3500000) :
        e157.iloc[i,2]= e157.iloc[i,2]+(.33*(.2))

#################################### END e = 157#############################    
    

############### e  = 158
e158 = pd.DataFrame(np.zeros((773, 3)))
e158.columns = ['a','b','c']




u=158

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e158.iloc[k,0] = dfp.iloc[i,0]
        e158.iloc[k,1] = dfp.iloc[i,1]
        e158.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e158.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e158.iloc[i,0]),4] == df200.iloc[int(e158.iloc[i,0]),4]:
        e158.iloc[i,2]= e158.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e158.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e158.iloc[i,0]),10] < 1 and  df200.iloc[int(e158.iloc[i,0]),10] <1) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e158.iloc[i,0]),10] < 2 and  df200.iloc[int(e158.iloc[i,0]),10] <2) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e158.iloc[i,0]),10] < 3 and  df200.iloc[int(e158.iloc[i,0]),10] <3) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e158.iloc[i,0]),10] < 4 and  df200.iloc[int(e158.iloc[i,0]),10] <4) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e158.iloc[i,0]),10] < 6 and  df200.iloc[int(e158.iloc[i,0]),10] <6) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e158.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e158.iloc[i,0]),5] < 700000 and  df200.iloc[int(e158.iloc[i,0]),5] <700000) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e158.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e158.iloc[i,0]),5] <1400000) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e158.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e158.iloc[i,0]),5] <2100000) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e158.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e158.iloc[i,0]),5] <2800000) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e158.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e158.iloc[i,0]),5] <3500000) :
        e158.iloc[i,2]= e158.iloc[i,2]+(.33*(.2))

#################################### END e = 158 ############################  
        
############### e  = 159
e159 = pd.DataFrame(np.zeros((773, 3)))
e159.columns = ['a','b','c']




u=159

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e159.iloc[k,0] = dfp.iloc[i,0]
        e159.iloc[k,1] = dfp.iloc[i,1]
        e159.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e159.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e159.iloc[i,0]),4] == df200.iloc[int(e159.iloc[i,0]),4]:
        e159.iloc[i,2]= e159.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e159.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e159.iloc[i,0]),10] < 1 and  df200.iloc[int(e159.iloc[i,0]),10] <1) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e159.iloc[i,0]),10] < 2 and  df200.iloc[int(e159.iloc[i,0]),10] <2) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e159.iloc[i,0]),10] < 3 and  df200.iloc[int(e159.iloc[i,0]),10] <3) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e159.iloc[i,0]),10] < 4 and  df200.iloc[int(e159.iloc[i,0]),10] <4) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e159.iloc[i,0]),10] < 6 and  df200.iloc[int(e159.iloc[i,0]),10] <6) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e159.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e159.iloc[i,0]),5] < 700000 and  df200.iloc[int(e159.iloc[i,0]),5] <700000) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e159.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e159.iloc[i,0]),5] <1400000) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e159.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e159.iloc[i,0]),5] <2100000) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e159.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e159.iloc[i,0]),5] <2800000) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e159.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e159.iloc[i,0]),5] <3500000) :
        e159.iloc[i,2]= e159.iloc[i,2]+(.33*(.2))

#################################### END e = 159 ############################  
        
############### e  = 160
e160 = pd.DataFrame(np.zeros((773, 3)))
e160.columns = ['a','b','c']




u=160

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e160.iloc[k,0] = dfp.iloc[i,0]
        e160.iloc[k,1] = dfp.iloc[i,1]
        e160.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e160.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e160.iloc[i,0]),4] == df200.iloc[int(e160.iloc[i,0]),4]:
        e160.iloc[i,2]= e160.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e160.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e160.iloc[i,0]),10] < 1 and  df200.iloc[int(e160.iloc[i,0]),10] <1) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e160.iloc[i,0]),10] < 2 and  df200.iloc[int(e160.iloc[i,0]),10] <2) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e160.iloc[i,0]),10] < 3 and  df200.iloc[int(e160.iloc[i,0]),10] <3) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e160.iloc[i,0]),10] < 4 and  df200.iloc[int(e160.iloc[i,0]),10] <4) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e160.iloc[i,0]),10] < 6 and  df200.iloc[int(e160.iloc[i,0]),10] <6) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e160.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e160.iloc[i,0]),5] < 700000 and  df200.iloc[int(e160.iloc[i,0]),5] <700000) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e160.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e160.iloc[i,0]),5] <1400000) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e160.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e160.iloc[i,0]),5] <2100000) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e160.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e160.iloc[i,0]),5] <2800000) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e160.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e160.iloc[i,0]),5] <3500000) :
        e160.iloc[i,2]= e160.iloc[i,2]+(.33*(.2))

#################################### END e = 160 ############################  

 ############### e  = 161
e161 = pd.DataFrame(np.zeros((773, 3)))
e161.columns = ['a','b','c']




u=161

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e161.iloc[k,0] = dfp.iloc[i,0]
        e161.iloc[k,1] = dfp.iloc[i,1]
        e161.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e161.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e161.iloc[i,0]),4] == df200.iloc[int(e161.iloc[i,0]),4]:
        e161.iloc[i,2]= e161.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e161.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e161.iloc[i,0]),10] < 1 and  df200.iloc[int(e161.iloc[i,0]),10] <1) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e161.iloc[i,0]),10] < 2 and  df200.iloc[int(e161.iloc[i,0]),10] <2) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e161.iloc[i,0]),10] < 3 and  df200.iloc[int(e161.iloc[i,0]),10] <3) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e161.iloc[i,0]),10] < 4 and  df200.iloc[int(e161.iloc[i,0]),10] <4) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e161.iloc[i,0]),10] < 6 and  df200.iloc[int(e161.iloc[i,0]),10] <6) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e161.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e161.iloc[i,0]),5] < 700000 and  df200.iloc[int(e161.iloc[i,0]),5] <700000) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e161.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e161.iloc[i,0]),5] <1400000) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e161.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e161.iloc[i,0]),5] <2100000) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e161.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e161.iloc[i,0]),5] <2800000) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e161.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e161.iloc[i,0]),5] <3500000) :
        e161.iloc[i,2]= e161.iloc[i,2]+(.33*(.2))

#################################### END e = 161 ############################  
        
        
############### e  = 162
e162 = pd.DataFrame(np.zeros((773, 3)))
e162.columns = ['a','b','c']




u=162

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e162.iloc[k,0] = dfp.iloc[i,0]
        e162.iloc[k,1] = dfp.iloc[i,1]
        e162.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e162.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e162.iloc[i,0]),4] == df200.iloc[int(e162.iloc[i,0]),4]:
        e162.iloc[i,2]= e162.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e162.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e162.iloc[i,0]),10] < 1 and  df200.iloc[int(e162.iloc[i,0]),10] <1) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e162.iloc[i,0]),10] < 2 and  df200.iloc[int(e162.iloc[i,0]),10] <2) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e162.iloc[i,0]),10] < 3 and  df200.iloc[int(e162.iloc[i,0]),10] <3) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e162.iloc[i,0]),10] < 4 and  df200.iloc[int(e162.iloc[i,0]),10] <4) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e162.iloc[i,0]),10] < 6 and  df200.iloc[int(e162.iloc[i,0]),10] <6) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e162.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e162.iloc[i,0]),5] < 700000 and  df200.iloc[int(e162.iloc[i,0]),5] <700000) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e162.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e162.iloc[i,0]),5] <1400000) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e162.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e162.iloc[i,0]),5] <2100000) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e162.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e162.iloc[i,0]),5] <2800000) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e162.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e162.iloc[i,0]),5] <3500000) :
        e162.iloc[i,2]= e162.iloc[i,2]+(.33*(.2))

#################################### END e = 162 ############################  
        
############### e  = 163
e163 = pd.DataFrame(np.zeros((773, 3)))
e163.columns = ['a','b','c']




u=163

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e163.iloc[k,0] = dfp.iloc[i,0]
        e163.iloc[k,1] = dfp.iloc[i,1]
        e163.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e163.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e163.iloc[i,0]),4] == df200.iloc[int(e163.iloc[i,0]),4]:
        e163.iloc[i,2]= e163.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e163.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e163.iloc[i,0]),10] < 1 and  df200.iloc[int(e163.iloc[i,0]),10] <1) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e163.iloc[i,0]),10] < 2 and  df200.iloc[int(e163.iloc[i,0]),10] <2) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e163.iloc[i,0]),10] < 3 and  df200.iloc[int(e163.iloc[i,0]),10] <3) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e163.iloc[i,0]),10] < 4 and  df200.iloc[int(e163.iloc[i,0]),10] <4) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e163.iloc[i,0]),10] < 6 and  df200.iloc[int(e163.iloc[i,0]),10] <6) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e163.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e163.iloc[i,0]),5] < 700000 and  df200.iloc[int(e163.iloc[i,0]),5] <700000) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e163.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e163.iloc[i,0]),5] <1400000) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e163.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e163.iloc[i,0]),5] <2100000) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e163.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e163.iloc[i,0]),5] <2800000) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e163.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e163.iloc[i,0]),5] <3500000) :
        e163.iloc[i,2]= e163.iloc[i,2]+(.33*(.2))

#################################### END e = 163 ############################  
        
############### e  = 164
e164 = pd.DataFrame(np.zeros((773, 3)))
e164.columns = ['a','b','c']




u=164

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e164.iloc[k,0] = dfp.iloc[i,0]
        e164.iloc[k,1] = dfp.iloc[i,1]
        e164.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e164.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e164.iloc[i,0]),4] == df200.iloc[int(e164.iloc[i,0]),4]:
        e164.iloc[i,2]= e164.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e164.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e164.iloc[i,0]),10] < 1 and  df200.iloc[int(e164.iloc[i,0]),10] <1) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e164.iloc[i,0]),10] < 2 and  df200.iloc[int(e164.iloc[i,0]),10] <2) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e164.iloc[i,0]),10] < 3 and  df200.iloc[int(e164.iloc[i,0]),10] <3) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e164.iloc[i,0]),10] < 4 and  df200.iloc[int(e164.iloc[i,0]),10] <4) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e164.iloc[i,0]),10] < 6 and  df200.iloc[int(e164.iloc[i,0]),10] <6) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e164.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e164.iloc[i,0]),5] < 700000 and  df200.iloc[int(e164.iloc[i,0]),5] <700000) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e164.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e164.iloc[i,0]),5] <1400000) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e164.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e164.iloc[i,0]),5] <2100000) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e164.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e164.iloc[i,0]),5] <2800000) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e164.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e164.iloc[i,0]),5] <3500000) :
        e164.iloc[i,2]= e164.iloc[i,2]+(.33*(.2))

#################################### END e = 164 ############################  

############### e  = 165
e165 = pd.DataFrame(np.zeros((773, 3)))
e165.columns = ['a','b','c']




u=165

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e165.iloc[k,0] = dfp.iloc[i,0]
        e165.iloc[k,1] = dfp.iloc[i,1]
        e165.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e165.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e165.iloc[i,0]),4] == df200.iloc[int(e165.iloc[i,0]),4]:
        e165.iloc[i,2]= e165.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e165.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e165.iloc[i,0]),10] < 1 and  df200.iloc[int(e165.iloc[i,0]),10] <1) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e165.iloc[i,0]),10] < 2 and  df200.iloc[int(e165.iloc[i,0]),10] <2) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e165.iloc[i,0]),10] < 3 and  df200.iloc[int(e165.iloc[i,0]),10] <3) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e165.iloc[i,0]),10] < 4 and  df200.iloc[int(e165.iloc[i,0]),10] <4) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e165.iloc[i,0]),10] < 6 and  df200.iloc[int(e165.iloc[i,0]),10] <6) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e165.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e165.iloc[i,0]),5] < 700000 and  df200.iloc[int(e165.iloc[i,0]),5] <700000) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e165.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e165.iloc[i,0]),5] <1400000) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e165.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e165.iloc[i,0]),5] <2100000) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e165.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e165.iloc[i,0]),5] <2800000) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e165.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e165.iloc[i,0]),5] <3500000) :
        e165.iloc[i,2]= e165.iloc[i,2]+(.33*(.2))

#################################### END e = 165 ############################  

############### e  = 166
e166 = pd.DataFrame(np.zeros((773, 3)))
e166.columns = ['a','b','c']




u=166

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e166.iloc[k,0] = dfp.iloc[i,0]
        e166.iloc[k,1] = dfp.iloc[i,1]
        e166.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e166.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e166.iloc[i,0]),4] == df200.iloc[int(e166.iloc[i,0]),4]:
        e166.iloc[i,2]= e166.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e166.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e166.iloc[i,0]),10] < 1 and  df200.iloc[int(e166.iloc[i,0]),10] <1) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e166.iloc[i,0]),10] < 2 and  df200.iloc[int(e166.iloc[i,0]),10] <2) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e166.iloc[i,0]),10] < 3 and  df200.iloc[int(e166.iloc[i,0]),10] <3) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e166.iloc[i,0]),10] < 4 and  df200.iloc[int(e166.iloc[i,0]),10] <4) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e166.iloc[i,0]),10] < 6 and  df200.iloc[int(e166.iloc[i,0]),10] <6) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e166.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e166.iloc[i,0]),5] < 700000 and  df200.iloc[int(e166.iloc[i,0]),5] <700000) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e166.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e166.iloc[i,0]),5] <1400000) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e166.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e166.iloc[i,0]),5] <2100000) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e166.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e166.iloc[i,0]),5] <2800000) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e166.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e166.iloc[i,0]),5] <3500000) :
        e166.iloc[i,2]= e166.iloc[i,2]+(.33*(.2))

#################################### END e = 166 ############################  

############### e  = 167
e167 = pd.DataFrame(np.zeros((773, 3)))
e167.columns = ['a','b','c']




u=167

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e167.iloc[k,0] = dfp.iloc[i,0]
        e167.iloc[k,1] = dfp.iloc[i,1]
        e167.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e167.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e167.iloc[i,0]),4] == df200.iloc[int(e167.iloc[i,0]),4]:
        e167.iloc[i,2]= e167.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e167.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e167.iloc[i,0]),10] < 1 and  df200.iloc[int(e167.iloc[i,0]),10] <1) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e167.iloc[i,0]),10] < 2 and  df200.iloc[int(e167.iloc[i,0]),10] <2) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e167.iloc[i,0]),10] < 3 and  df200.iloc[int(e167.iloc[i,0]),10] <3) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e167.iloc[i,0]),10] < 4 and  df200.iloc[int(e167.iloc[i,0]),10] <4) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e167.iloc[i,0]),10] < 6 and  df200.iloc[int(e167.iloc[i,0]),10] <6) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e167.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e167.iloc[i,0]),5] < 700000 and  df200.iloc[int(e167.iloc[i,0]),5] <700000) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e167.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e167.iloc[i,0]),5] <1400000) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e167.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e167.iloc[i,0]),5] <2100000) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e167.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e167.iloc[i,0]),5] <2800000) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e167.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e167.iloc[i,0]),5] <3500000) :
        e167.iloc[i,2]= e167.iloc[i,2]+(.33*(.2))

#################################### END e = 167 ############################  

############### e  = 168
e168 = pd.DataFrame(np.zeros((773, 3)))
e168.columns = ['a','b','c']




u=168

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e168.iloc[k,0] = dfp.iloc[i,0]
        e168.iloc[k,1] = dfp.iloc[i,1]
        e168.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e168.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e168.iloc[i,0]),4] == df200.iloc[int(e168.iloc[i,0]),4]:
        e168.iloc[i,2]= e168.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e168.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e168.iloc[i,0]),10] < 1 and  df200.iloc[int(e168.iloc[i,0]),10] <1) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e168.iloc[i,0]),10] < 2 and  df200.iloc[int(e168.iloc[i,0]),10] <2) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e168.iloc[i,0]),10] < 3 and  df200.iloc[int(e168.iloc[i,0]),10] <3) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e168.iloc[i,0]),10] < 4 and  df200.iloc[int(e168.iloc[i,0]),10] <4) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e168.iloc[i,0]),10] < 6 and  df200.iloc[int(e168.iloc[i,0]),10] <6) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e168.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e168.iloc[i,0]),5] < 700000 and  df200.iloc[int(e168.iloc[i,0]),5] <700000) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e168.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e168.iloc[i,0]),5] <1400000) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e168.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e168.iloc[i,0]),5] <2100000) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e168.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e168.iloc[i,0]),5] <2800000) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e168.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e168.iloc[i,0]),5] <3500000) :
        e168.iloc[i,2]= e168.iloc[i,2]+(.33*(.2))

#################################### END e = 168 ############################  

############### e  = 169
e169 = pd.DataFrame(np.zeros((773, 3)))
e169.columns = ['a','b','c']




u=169

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e169.iloc[k,0] = dfp.iloc[i,0]
        e169.iloc[k,1] = dfp.iloc[i,1]
        e169.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e169.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e169.iloc[i,0]),4] == df200.iloc[int(e169.iloc[i,0]),4]:
        e169.iloc[i,2]= e169.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e169.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e169.iloc[i,0]),10] < 1 and  df200.iloc[int(e169.iloc[i,0]),10] <1) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e169.iloc[i,0]),10] < 2 and  df200.iloc[int(e169.iloc[i,0]),10] <2) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e169.iloc[i,0]),10] < 3 and  df200.iloc[int(e169.iloc[i,0]),10] <3) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e169.iloc[i,0]),10] < 4 and  df200.iloc[int(e169.iloc[i,0]),10] <4) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e169.iloc[i,0]),10] < 6 and  df200.iloc[int(e169.iloc[i,0]),10] <6) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e169.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e169.iloc[i,0]),5] < 700000 and  df200.iloc[int(e169.iloc[i,0]),5] <700000) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e169.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e169.iloc[i,0]),5] <1400000) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e169.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e169.iloc[i,0]),5] <2100000) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e169.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e169.iloc[i,0]),5] <2800000) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e169.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e169.iloc[i,0]),5] <3500000) :
        e169.iloc[i,2]= e169.iloc[i,2]+(.33*(.2))

#################################### END e = 169 ############################  

############### e  = 170
e170 = pd.DataFrame(np.zeros((773, 3)))
e170.columns = ['a','b','c']




u=170

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e170.iloc[k,0] = dfp.iloc[i,0]
        e170.iloc[k,1] = dfp.iloc[i,1]
        e170.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e170.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e170.iloc[i,0]),4] == df200.iloc[int(e170.iloc[i,0]),4]:
        e170.iloc[i,2]= e170.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e170.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e170.iloc[i,0]),10] < 1 and  df200.iloc[int(e170.iloc[i,0]),10] <1) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e170.iloc[i,0]),10] < 2 and  df200.iloc[int(e170.iloc[i,0]),10] <2) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e170.iloc[i,0]),10] < 3 and  df200.iloc[int(e170.iloc[i,0]),10] <3) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e170.iloc[i,0]),10] < 4 and  df200.iloc[int(e170.iloc[i,0]),10] <4) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e170.iloc[i,0]),10] < 6 and  df200.iloc[int(e170.iloc[i,0]),10] <6) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e170.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e170.iloc[i,0]),5] < 700000 and  df200.iloc[int(e170.iloc[i,0]),5] <700000) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e170.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e170.iloc[i,0]),5] <1400000) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e170.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e170.iloc[i,0]),5] <2100000) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e170.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e170.iloc[i,0]),5] <2800000) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e170.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e170.iloc[i,0]),5] <3500000) :
        e170.iloc[i,2]= e170.iloc[i,2]+(.33*(.2))

#################################### END e = 170 ############################  

############### e  = 171
e171 = pd.DataFrame(np.zeros((773, 3)))
e171.columns = ['a','b','c']




u=171

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e171.iloc[k,0] = dfp.iloc[i,0]
        e171.iloc[k,1] = dfp.iloc[i,1]
        e171.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e171.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e171.iloc[i,0]),4] == df200.iloc[int(e171.iloc[i,0]),4]:
        e171.iloc[i,2]= e171.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e171.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e171.iloc[i,0]),10] < 1 and  df200.iloc[int(e171.iloc[i,0]),10] <1) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e171.iloc[i,0]),10] < 2 and  df200.iloc[int(e171.iloc[i,0]),10] <2) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e171.iloc[i,0]),10] < 3 and  df200.iloc[int(e171.iloc[i,0]),10] <3) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e171.iloc[i,0]),10] < 4 and  df200.iloc[int(e171.iloc[i,0]),10] <4) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e171.iloc[i,0]),10] < 6 and  df200.iloc[int(e171.iloc[i,0]),10] <6) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e171.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e171.iloc[i,0]),5] < 700000 and  df200.iloc[int(e171.iloc[i,0]),5] <700000) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e171.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e171.iloc[i,0]),5] <1400000) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e171.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e171.iloc[i,0]),5] <2100000) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e171.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e171.iloc[i,0]),5] <2800000) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e171.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e171.iloc[i,0]),5] <3500000) :
        e171.iloc[i,2]= e171.iloc[i,2]+(.33*(.2))

#################################### END e = 171 ############################  

############### e  = 172
e172 = pd.DataFrame(np.zeros((773, 3)))
e172.columns = ['a','b','c']




u=172

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e172.iloc[k,0] = dfp.iloc[i,0]
        e172.iloc[k,1] = dfp.iloc[i,1]
        e172.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e172.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e172.iloc[i,0]),4] == df200.iloc[int(e172.iloc[i,0]),4]:
        e172.iloc[i,2]= e172.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e172.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e172.iloc[i,0]),10] < 1 and  df200.iloc[int(e172.iloc[i,0]),10] <1) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e172.iloc[i,0]),10] < 2 and  df200.iloc[int(e172.iloc[i,0]),10] <2) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e172.iloc[i,0]),10] < 3 and  df200.iloc[int(e172.iloc[i,0]),10] <3) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e172.iloc[i,0]),10] < 4 and  df200.iloc[int(e172.iloc[i,0]),10] <4) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e172.iloc[i,0]),10] < 6 and  df200.iloc[int(e172.iloc[i,0]),10] <6) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e172.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e172.iloc[i,0]),5] < 700000 and  df200.iloc[int(e172.iloc[i,0]),5] <700000) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e172.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e172.iloc[i,0]),5] <1400000) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e172.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e172.iloc[i,0]),5] <2100000) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e172.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e172.iloc[i,0]),5] <2800000) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e172.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e172.iloc[i,0]),5] <3500000) :
        e172.iloc[i,2]= e172.iloc[i,2]+(.33*(.2))

#################################### END e = 172 ############################  

############### e  = 173
e173 = pd.DataFrame(np.zeros((773, 3)))
e173.columns = ['a','b','c']




u=173

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e173.iloc[k,0] = dfp.iloc[i,0]
        e173.iloc[k,1] = dfp.iloc[i,1]
        e173.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e173.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e173.iloc[i,0]),4] == df200.iloc[int(e173.iloc[i,0]),4]:
        e173.iloc[i,2]= e173.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e173.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e173.iloc[i,0]),10] < 1 and  df200.iloc[int(e173.iloc[i,0]),10] <1) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e173.iloc[i,0]),10] < 2 and  df200.iloc[int(e173.iloc[i,0]),10] <2) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e173.iloc[i,0]),10] < 3 and  df200.iloc[int(e173.iloc[i,0]),10] <3) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e173.iloc[i,0]),10] < 4 and  df200.iloc[int(e173.iloc[i,0]),10] <4) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e173.iloc[i,0]),10] < 6 and  df200.iloc[int(e173.iloc[i,0]),10] <6) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e173.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e173.iloc[i,0]),5] < 700000 and  df200.iloc[int(e173.iloc[i,0]),5] <700000) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e173.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e173.iloc[i,0]),5] <1400000) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e173.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e173.iloc[i,0]),5] <2100000) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e173.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e173.iloc[i,0]),5] <2800000) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e173.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e173.iloc[i,0]),5] <3500000) :
        e173.iloc[i,2]= e173.iloc[i,2]+(.33*(.2))

#################################### END e = 173 ############################  

        
############### e  = 174
e174 = pd.DataFrame(np.zeros((773, 3)))
e174.columns = ['a','b','c']




u=174

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e174.iloc[k,0] = dfp.iloc[i,0]
        e174.iloc[k,1] = dfp.iloc[i,1]
        e174.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e174.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e174.iloc[i,0]),4] == df200.iloc[int(e174.iloc[i,0]),4]:
        e174.iloc[i,2]= e174.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e174.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e174.iloc[i,0]),10] < 1 and  df200.iloc[int(e174.iloc[i,0]),10] <1) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e174.iloc[i,0]),10] < 2 and  df200.iloc[int(e174.iloc[i,0]),10] <2) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e174.iloc[i,0]),10] < 3 and  df200.iloc[int(e174.iloc[i,0]),10] <3) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e174.iloc[i,0]),10] < 4 and  df200.iloc[int(e174.iloc[i,0]),10] <4) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e174.iloc[i,0]),10] < 6 and  df200.iloc[int(e174.iloc[i,0]),10] <6) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e174.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e174.iloc[i,0]),5] < 700000 and  df200.iloc[int(e174.iloc[i,0]),5] <700000) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e174.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e174.iloc[i,0]),5] <1400000) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e174.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e174.iloc[i,0]),5] <2100000) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e174.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e174.iloc[i,0]),5] <2800000) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e174.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e174.iloc[i,0]),5] <3500000) :
        e174.iloc[i,2]= e174.iloc[i,2]+(.33*(.2))

#################################### END e = 174 ############################  

############### e  = 175
e175 = pd.DataFrame(np.zeros((773, 3)))
e175.columns = ['a','b','c']




u=175

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e175.iloc[k,0] = dfp.iloc[i,0]
        e175.iloc[k,1] = dfp.iloc[i,1]
        e175.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e175.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e175.iloc[i,0]),4] == df200.iloc[int(e175.iloc[i,0]),4]:
        e175.iloc[i,2]= e175.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e175.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e175.iloc[i,0]),10] < 1 and  df200.iloc[int(e175.iloc[i,0]),10] <1) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e175.iloc[i,0]),10] < 2 and  df200.iloc[int(e175.iloc[i,0]),10] <2) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e175.iloc[i,0]),10] < 3 and  df200.iloc[int(e175.iloc[i,0]),10] <3) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e175.iloc[i,0]),10] < 4 and  df200.iloc[int(e175.iloc[i,0]),10] <4) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e175.iloc[i,0]),10] < 6 and  df200.iloc[int(e175.iloc[i,0]),10] <6) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e175.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e175.iloc[i,0]),5] < 700000 and  df200.iloc[int(e175.iloc[i,0]),5] <700000) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e175.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e175.iloc[i,0]),5] <1400000) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e175.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e175.iloc[i,0]),5] <2100000) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e175.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e175.iloc[i,0]),5] <2800000) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e175.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e175.iloc[i,0]),5] <3500000) :
        e175.iloc[i,2]= e175.iloc[i,2]+(.33*(.2))

#################################### END e = 175 ############################  

############### e  = 176
e176 = pd.DataFrame(np.zeros((773, 3)))
e176.columns = ['a','b','c']




u=176

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e176.iloc[k,0] = dfp.iloc[i,0]
        e176.iloc[k,1] = dfp.iloc[i,1]
        e176.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e176.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e176.iloc[i,0]),4] == df200.iloc[int(e176.iloc[i,0]),4]:
        e176.iloc[i,2]= e176.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e176.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e176.iloc[i,0]),10] < 1 and  df200.iloc[int(e176.iloc[i,0]),10] <1) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e176.iloc[i,0]),10] < 2 and  df200.iloc[int(e176.iloc[i,0]),10] <2) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e176.iloc[i,0]),10] < 3 and  df200.iloc[int(e176.iloc[i,0]),10] <3) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e176.iloc[i,0]),10] < 4 and  df200.iloc[int(e176.iloc[i,0]),10] <4) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e176.iloc[i,0]),10] < 6 and  df200.iloc[int(e176.iloc[i,0]),10] <6) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e176.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e176.iloc[i,0]),5] < 700000 and  df200.iloc[int(e176.iloc[i,0]),5] <700000) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e176.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e176.iloc[i,0]),5] <1400000) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e176.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e176.iloc[i,0]),5] <2100000) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e176.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e176.iloc[i,0]),5] <2800000) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e176.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e176.iloc[i,0]),5] <3500000) :
        e176.iloc[i,2]= e176.iloc[i,2]+(.33*(.2))

#################################### END e = 176 ############################  

############### e  = 177
e177 = pd.DataFrame(np.zeros((773, 3)))
e177.columns = ['a','b','c']




u=177

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e177.iloc[k,0] = dfp.iloc[i,0]
        e177.iloc[k,1] = dfp.iloc[i,1]
        e177.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e177.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e177.iloc[i,0]),4] == df200.iloc[int(e177.iloc[i,0]),4]:
        e177.iloc[i,2]= e177.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e177.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e177.iloc[i,0]),10] < 1 and  df200.iloc[int(e177.iloc[i,0]),10] <1) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e177.iloc[i,0]),10] < 2 and  df200.iloc[int(e177.iloc[i,0]),10] <2) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e177.iloc[i,0]),10] < 3 and  df200.iloc[int(e177.iloc[i,0]),10] <3) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e177.iloc[i,0]),10] < 4 and  df200.iloc[int(e177.iloc[i,0]),10] <4) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e177.iloc[i,0]),10] < 6 and  df200.iloc[int(e177.iloc[i,0]),10] <6) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e177.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e177.iloc[i,0]),5] < 700000 and  df200.iloc[int(e177.iloc[i,0]),5] <700000) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e177.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e177.iloc[i,0]),5] <1400000) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e177.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e177.iloc[i,0]),5] <2100000) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e177.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e177.iloc[i,0]),5] <2800000) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e177.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e177.iloc[i,0]),5] <3500000) :
        e177.iloc[i,2]= e177.iloc[i,2]+(.33*(.2))

#################################### END e = 177 ############################  

############### e  = 178
e178 = pd.DataFrame(np.zeros((773, 3)))
e178.columns = ['a','b','c']




u=178

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e178.iloc[k,0] = dfp.iloc[i,0]
        e178.iloc[k,1] = dfp.iloc[i,1]
        e178.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e178.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e178.iloc[i,0]),4] == df200.iloc[int(e178.iloc[i,0]),4]:
        e178.iloc[i,2]= e178.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e178.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e178.iloc[i,0]),10] < 1 and  df200.iloc[int(e178.iloc[i,0]),10] <1) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e178.iloc[i,0]),10] < 2 and  df200.iloc[int(e178.iloc[i,0]),10] <2) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e178.iloc[i,0]),10] < 3 and  df200.iloc[int(e178.iloc[i,0]),10] <3) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e178.iloc[i,0]),10] < 4 and  df200.iloc[int(e178.iloc[i,0]),10] <4) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e178.iloc[i,0]),10] < 6 and  df200.iloc[int(e178.iloc[i,0]),10] <6) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e178.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e178.iloc[i,0]),5] < 700000 and  df200.iloc[int(e178.iloc[i,0]),5] <700000) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e178.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e178.iloc[i,0]),5] <1400000) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e178.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e178.iloc[i,0]),5] <2100000) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e178.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e178.iloc[i,0]),5] <2800000) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e178.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e178.iloc[i,0]),5] <3500000) :
        e178.iloc[i,2]= e178.iloc[i,2]+(.33*(.2))

#################################### END e = 178 ############################  

############### e  = 179
e179 = pd.DataFrame(np.zeros((773, 3)))
e179.columns = ['a','b','c']




u=179

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e179.iloc[k,0] = dfp.iloc[i,0]
        e179.iloc[k,1] = dfp.iloc[i,1]
        e179.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e179.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e179.iloc[i,0]),4] == df200.iloc[int(e179.iloc[i,0]),4]:
        e179.iloc[i,2]= e179.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e179.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e179.iloc[i,0]),10] < 1 and  df200.iloc[int(e179.iloc[i,0]),10] <1) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e179.iloc[i,0]),10] < 2 and  df200.iloc[int(e179.iloc[i,0]),10] <2) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e179.iloc[i,0]),10] < 3 and  df200.iloc[int(e179.iloc[i,0]),10] <3) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e179.iloc[i,0]),10] < 4 and  df200.iloc[int(e179.iloc[i,0]),10] <4) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e179.iloc[i,0]),10] < 6 and  df200.iloc[int(e179.iloc[i,0]),10] <6) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e179.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e179.iloc[i,0]),5] < 700000 and  df200.iloc[int(e179.iloc[i,0]),5] <700000) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e179.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e179.iloc[i,0]),5] <1400000) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e179.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e179.iloc[i,0]),5] <2100000) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e179.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e179.iloc[i,0]),5] <2800000) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e179.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e179.iloc[i,0]),5] <3500000) :
        e179.iloc[i,2]= e179.iloc[i,2]+(.33*(.2))

#################################### END e = 179 ############################  

############### e  = 180
e180 = pd.DataFrame(np.zeros((773, 3)))
e180.columns = ['a','b','c']




u=180

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e180.iloc[k,0] = dfp.iloc[i,0]
        e180.iloc[k,1] = dfp.iloc[i,1]
        e180.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e180.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e180.iloc[i,0]),4] == df200.iloc[int(e180.iloc[i,0]),4]:
        e180.iloc[i,2]= e180.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e180.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e180.iloc[i,0]),10] < 1 and  df200.iloc[int(e180.iloc[i,0]),10] <1) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e180.iloc[i,0]),10] < 2 and  df200.iloc[int(e180.iloc[i,0]),10] <2) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e180.iloc[i,0]),10] < 3 and  df200.iloc[int(e180.iloc[i,0]),10] <3) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e180.iloc[i,0]),10] < 4 and  df200.iloc[int(e180.iloc[i,0]),10] <4) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e180.iloc[i,0]),10] < 6 and  df200.iloc[int(e180.iloc[i,0]),10] <6) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e180.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e180.iloc[i,0]),5] < 700000 and  df200.iloc[int(e180.iloc[i,0]),5] <700000) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e180.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e180.iloc[i,0]),5] <1400000) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e180.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e180.iloc[i,0]),5] <2100000) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e180.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e180.iloc[i,0]),5] <2800000) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e180.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e180.iloc[i,0]),5] <3500000) :
        e180.iloc[i,2]= e180.iloc[i,2]+(.33*(.2))

#################################### END e = 180 ############################  

############### e  = 181
e181 = pd.DataFrame(np.zeros((773, 3)))
e181.columns = ['a','b','c']




u=181

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e181.iloc[k,0] = dfp.iloc[i,0]
        e181.iloc[k,1] = dfp.iloc[i,1]
        e181.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e181.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e181.iloc[i,0]),4] == df200.iloc[int(e181.iloc[i,0]),4]:
        e181.iloc[i,2]= e181.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e181.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e181.iloc[i,0]),10] < 1 and  df200.iloc[int(e181.iloc[i,0]),10] <1) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e181.iloc[i,0]),10] < 2 and  df200.iloc[int(e181.iloc[i,0]),10] <2) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e181.iloc[i,0]),10] < 3 and  df200.iloc[int(e181.iloc[i,0]),10] <3) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e181.iloc[i,0]),10] < 4 and  df200.iloc[int(e181.iloc[i,0]),10] <4) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e181.iloc[i,0]),10] < 6 and  df200.iloc[int(e181.iloc[i,0]),10] <6) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e181.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e181.iloc[i,0]),5] < 700000 and  df200.iloc[int(e181.iloc[i,0]),5] <700000) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e181.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e181.iloc[i,0]),5] <1400000) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e181.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e181.iloc[i,0]),5] <2100000) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e181.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e181.iloc[i,0]),5] <2800000) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e181.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e181.iloc[i,0]),5] <3500000) :
        e181.iloc[i,2]= e181.iloc[i,2]+(.33*(.2))

#################################### END e = 181 ############################  

############### e  = 182
e182 = pd.DataFrame(np.zeros((773, 3)))
e182.columns = ['a','b','c']




u=182

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e182.iloc[k,0] = dfp.iloc[i,0]
        e182.iloc[k,1] = dfp.iloc[i,1]
        e182.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e182.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e182.iloc[i,0]),4] == df200.iloc[int(e182.iloc[i,0]),4]:
        e182.iloc[i,2]= e182.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e182.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e182.iloc[i,0]),10] < 1 and  df200.iloc[int(e182.iloc[i,0]),10] <1) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e182.iloc[i,0]),10] < 2 and  df200.iloc[int(e182.iloc[i,0]),10] <2) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e182.iloc[i,0]),10] < 3 and  df200.iloc[int(e182.iloc[i,0]),10] <3) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e182.iloc[i,0]),10] < 4 and  df200.iloc[int(e182.iloc[i,0]),10] <4) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e182.iloc[i,0]),10] < 6 and  df200.iloc[int(e182.iloc[i,0]),10] <6) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e182.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e182.iloc[i,0]),5] < 700000 and  df200.iloc[int(e182.iloc[i,0]),5] <700000) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e182.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e182.iloc[i,0]),5] <1400000) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e182.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e182.iloc[i,0]),5] <2100000) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e182.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e182.iloc[i,0]),5] <2800000) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e182.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e182.iloc[i,0]),5] <3500000) :
        e182.iloc[i,2]= e182.iloc[i,2]+(.33*(.2))

#################################### END e = 182 ############################  

 ############### e  = 183
e183 = pd.DataFrame(np.zeros((773, 3)))
e183.columns = ['a','b','c']




u=183

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e183.iloc[k,0] = dfp.iloc[i,0]
        e183.iloc[k,1] = dfp.iloc[i,1]
        e183.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e183.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e183.iloc[i,0]),4] == df200.iloc[int(e183.iloc[i,0]),4]:
        e183.iloc[i,2]= e183.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e183.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e183.iloc[i,0]),10] < 1 and  df200.iloc[int(e183.iloc[i,0]),10] <1) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e183.iloc[i,0]),10] < 2 and  df200.iloc[int(e183.iloc[i,0]),10] <2) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e183.iloc[i,0]),10] < 3 and  df200.iloc[int(e183.iloc[i,0]),10] <3) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e183.iloc[i,0]),10] < 4 and  df200.iloc[int(e183.iloc[i,0]),10] <4) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e183.iloc[i,0]),10] < 6 and  df200.iloc[int(e183.iloc[i,0]),10] <6) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e183.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e183.iloc[i,0]),5] < 700000 and  df200.iloc[int(e183.iloc[i,0]),5] <700000) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e183.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e183.iloc[i,0]),5] <1400000) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e183.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e183.iloc[i,0]),5] <2100000) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e183.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e183.iloc[i,0]),5] <2800000) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e183.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e183.iloc[i,0]),5] <3500000) :
        e183.iloc[i,2]= e183.iloc[i,2]+(.33*(.2))

#################################### END e = 183 ############################  

############### e  = 184
e184 = pd.DataFrame(np.zeros((773, 3)))
e184.columns = ['a','b','c']




u=184

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e184.iloc[k,0] = dfp.iloc[i,0]
        e184.iloc[k,1] = dfp.iloc[i,1]
        e184.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e184.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e184.iloc[i,0]),4] == df200.iloc[int(e184.iloc[i,0]),4]:
        e184.iloc[i,2]= e184.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e184.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e184.iloc[i,0]),10] < 1 and  df200.iloc[int(e184.iloc[i,0]),10] <1) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e184.iloc[i,0]),10] < 2 and  df200.iloc[int(e184.iloc[i,0]),10] <2) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e184.iloc[i,0]),10] < 3 and  df200.iloc[int(e184.iloc[i,0]),10] <3) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e184.iloc[i,0]),10] < 4 and  df200.iloc[int(e184.iloc[i,0]),10] <4) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e184.iloc[i,0]),10] < 6 and  df200.iloc[int(e184.iloc[i,0]),10] <6) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e184.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e184.iloc[i,0]),5] < 700000 and  df200.iloc[int(e184.iloc[i,0]),5] <700000) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e184.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e184.iloc[i,0]),5] <1400000) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e184.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e184.iloc[i,0]),5] <2100000) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e184.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e184.iloc[i,0]),5] <2800000) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e184.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e184.iloc[i,0]),5] <3500000) :
        e184.iloc[i,2]= e184.iloc[i,2]+(.33*(.2))

#################################### END e = 184 ############################  

############### e  = 185
e185 = pd.DataFrame(np.zeros((773, 3)))
e185.columns = ['a','b','c']




u=185

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e185.iloc[k,0] = dfp.iloc[i,0]
        e185.iloc[k,1] = dfp.iloc[i,1]
        e185.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e185.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e185.iloc[i,0]),4] == df200.iloc[int(e185.iloc[i,0]),4]:
        e185.iloc[i,2]= e185.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e185.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e185.iloc[i,0]),10] < 1 and  df200.iloc[int(e185.iloc[i,0]),10] <1) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e185.iloc[i,0]),10] < 2 and  df200.iloc[int(e185.iloc[i,0]),10] <2) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e185.iloc[i,0]),10] < 3 and  df200.iloc[int(e185.iloc[i,0]),10] <3) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e185.iloc[i,0]),10] < 4 and  df200.iloc[int(e185.iloc[i,0]),10] <4) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e185.iloc[i,0]),10] < 6 and  df200.iloc[int(e185.iloc[i,0]),10] <6) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e185.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e185.iloc[i,0]),5] < 700000 and  df200.iloc[int(e185.iloc[i,0]),5] <700000) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e185.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e185.iloc[i,0]),5] <1400000) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e185.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e185.iloc[i,0]),5] <2100000) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e185.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e185.iloc[i,0]),5] <2800000) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e185.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e185.iloc[i,0]),5] <3500000) :
        e185.iloc[i,2]= e185.iloc[i,2]+(.33*(.2))

#################################### END e = 185 ############################  

############### e  = 186
e186 = pd.DataFrame(np.zeros((773, 3)))
e186.columns = ['a','b','c']




u=186

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e186.iloc[k,0] = dfp.iloc[i,0]
        e186.iloc[k,1] = dfp.iloc[i,1]
        e186.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e186.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e186.iloc[i,0]),4] == df200.iloc[int(e186.iloc[i,0]),4]:
        e186.iloc[i,2]= e186.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e186.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e186.iloc[i,0]),10] < 1 and  df200.iloc[int(e186.iloc[i,0]),10] <1) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e186.iloc[i,0]),10] < 2 and  df200.iloc[int(e186.iloc[i,0]),10] <2) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e186.iloc[i,0]),10] < 3 and  df200.iloc[int(e186.iloc[i,0]),10] <3) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e186.iloc[i,0]),10] < 4 and  df200.iloc[int(e186.iloc[i,0]),10] <4) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e186.iloc[i,0]),10] < 6 and  df200.iloc[int(e186.iloc[i,0]),10] <6) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e186.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e186.iloc[i,0]),5] < 700000 and  df200.iloc[int(e186.iloc[i,0]),5] <700000) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e186.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e186.iloc[i,0]),5] <1400000) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e186.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e186.iloc[i,0]),5] <2100000) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e186.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e186.iloc[i,0]),5] <2800000) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e186.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e186.iloc[i,0]),5] <3500000) :
        e186.iloc[i,2]= e186.iloc[i,2]+(.33*(.2))

#################################### END e = 186 ############################  

############### e  = 187
e187 = pd.DataFrame(np.zeros((773, 3)))
e187.columns = ['a','b','c']




u=187

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e187.iloc[k,0] = dfp.iloc[i,0]
        e187.iloc[k,1] = dfp.iloc[i,1]
        e187.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e187.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e187.iloc[i,0]),4] == df200.iloc[int(e187.iloc[i,0]),4]:
        e187.iloc[i,2]= e187.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e187.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e187.iloc[i,0]),10] < 1 and  df200.iloc[int(e187.iloc[i,0]),10] <1) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e187.iloc[i,0]),10] < 2 and  df200.iloc[int(e187.iloc[i,0]),10] <2) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e187.iloc[i,0]),10] < 3 and  df200.iloc[int(e187.iloc[i,0]),10] <3) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e187.iloc[i,0]),10] < 4 and  df200.iloc[int(e187.iloc[i,0]),10] <4) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e187.iloc[i,0]),10] < 6 and  df200.iloc[int(e187.iloc[i,0]),10] <6) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e187.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e187.iloc[i,0]),5] < 700000 and  df200.iloc[int(e187.iloc[i,0]),5] <700000) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e187.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e187.iloc[i,0]),5] <1400000) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e187.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e187.iloc[i,0]),5] <2100000) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e187.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e187.iloc[i,0]),5] <2800000) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e187.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e187.iloc[i,0]),5] <3500000) :
        e187.iloc[i,2]= e187.iloc[i,2]+(.33*(.2))

#################################### END e = 187 ############################  

############### e  = 188
e188 = pd.DataFrame(np.zeros((773, 3)))
e188.columns = ['a','b','c']




u=188

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e188.iloc[k,0] = dfp.iloc[i,0]
        e188.iloc[k,1] = dfp.iloc[i,1]
        e188.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e188.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e188.iloc[i,0]),4] == df200.iloc[int(e188.iloc[i,0]),4]:
        e188.iloc[i,2]= e188.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e188.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e188.iloc[i,0]),10] < 1 and  df200.iloc[int(e188.iloc[i,0]),10] <1) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e188.iloc[i,0]),10] < 2 and  df200.iloc[int(e188.iloc[i,0]),10] <2) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e188.iloc[i,0]),10] < 3 and  df200.iloc[int(e188.iloc[i,0]),10] <3) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e188.iloc[i,0]),10] < 4 and  df200.iloc[int(e188.iloc[i,0]),10] <4) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e188.iloc[i,0]),10] < 6 and  df200.iloc[int(e188.iloc[i,0]),10] <6) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e188.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e188.iloc[i,0]),5] < 700000 and  df200.iloc[int(e188.iloc[i,0]),5] <700000) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e188.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e188.iloc[i,0]),5] <1400000) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e188.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e188.iloc[i,0]),5] <2100000) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e188.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e188.iloc[i,0]),5] <2800000) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e188.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e188.iloc[i,0]),5] <3500000) :
        e188.iloc[i,2]= e188.iloc[i,2]+(.33*(.2))

#################################### END e = 188 ############################  
        
############### e  = 189
e189 = pd.DataFrame(np.zeros((773, 3)))
e189.columns = ['a','b','c']




u=189

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e189.iloc[k,0] = dfp.iloc[i,0]
        e189.iloc[k,1] = dfp.iloc[i,1]
        e189.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e189.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e189.iloc[i,0]),4] == df200.iloc[int(e189.iloc[i,0]),4]:
        e189.iloc[i,2]= e189.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e189.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e189.iloc[i,0]),10] < 1 and  df200.iloc[int(e189.iloc[i,0]),10] <1) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e189.iloc[i,0]),10] < 2 and  df200.iloc[int(e189.iloc[i,0]),10] <2) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e189.iloc[i,0]),10] < 3 and  df200.iloc[int(e189.iloc[i,0]),10] <3) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e189.iloc[i,0]),10] < 4 and  df200.iloc[int(e189.iloc[i,0]),10] <4) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e189.iloc[i,0]),10] < 6 and  df200.iloc[int(e189.iloc[i,0]),10] <6) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e189.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e189.iloc[i,0]),5] < 700000 and  df200.iloc[int(e189.iloc[i,0]),5] <700000) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e189.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e189.iloc[i,0]),5] <1400000) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e189.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e189.iloc[i,0]),5] <2100000) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e189.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e189.iloc[i,0]),5] <2800000) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e189.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e189.iloc[i,0]),5] <3500000) :
        e189.iloc[i,2]= e189.iloc[i,2]+(.33*(.2))

#################################### END e = 189 ############################  

############### e  = 190
e190 = pd.DataFrame(np.zeros((773, 3)))
e190.columns = ['a','b','c']




u=190

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e190.iloc[k,0] = dfp.iloc[i,0]
        e190.iloc[k,1] = dfp.iloc[i,1]
        e190.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e190.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e190.iloc[i,0]),4] == df200.iloc[int(e190.iloc[i,0]),4]:
        e190.iloc[i,2]= e190.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e190.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e190.iloc[i,0]),10] < 1 and  df200.iloc[int(e190.iloc[i,0]),10] <1) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e190.iloc[i,0]),10] < 2 and  df200.iloc[int(e190.iloc[i,0]),10] <2) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e190.iloc[i,0]),10] < 3 and  df200.iloc[int(e190.iloc[i,0]),10] <3) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e190.iloc[i,0]),10] < 4 and  df200.iloc[int(e190.iloc[i,0]),10] <4) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e190.iloc[i,0]),10] < 6 and  df200.iloc[int(e190.iloc[i,0]),10] <6) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e190.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e190.iloc[i,0]),5] < 700000 and  df200.iloc[int(e190.iloc[i,0]),5] <700000) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e190.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e190.iloc[i,0]),5] <1400000) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e190.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e190.iloc[i,0]),5] <2100000) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e190.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e190.iloc[i,0]),5] <2800000) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e190.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e190.iloc[i,0]),5] <3500000) :
        e190.iloc[i,2]= e190.iloc[i,2]+(.33*(.2))

#################################### END e = 190 ############################  
        
############### e  = 191
e191 = pd.DataFrame(np.zeros((773, 3)))
e191.columns = ['a','b','c']




u=191

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e191.iloc[k,0] = dfp.iloc[i,0]
        e191.iloc[k,1] = dfp.iloc[i,1]
        e191.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e191.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e191.iloc[i,0]),4] == df200.iloc[int(e191.iloc[i,0]),4]:
        e191.iloc[i,2]= e191.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e191.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e191.iloc[i,0]),10] < 1 and  df200.iloc[int(e191.iloc[i,0]),10] <1) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e191.iloc[i,0]),10] < 2 and  df200.iloc[int(e191.iloc[i,0]),10] <2) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e191.iloc[i,0]),10] < 3 and  df200.iloc[int(e191.iloc[i,0]),10] <3) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e191.iloc[i,0]),10] < 4 and  df200.iloc[int(e191.iloc[i,0]),10] <4) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e191.iloc[i,0]),10] < 6 and  df200.iloc[int(e191.iloc[i,0]),10] <6) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e191.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e191.iloc[i,0]),5] < 700000 and  df200.iloc[int(e191.iloc[i,0]),5] <700000) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e191.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e191.iloc[i,0]),5] <1400000) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e191.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e191.iloc[i,0]),5] <2100000) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e191.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e191.iloc[i,0]),5] <2800000) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e191.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e191.iloc[i,0]),5] <3500000) :
        e191.iloc[i,2]= e191.iloc[i,2]+(.33*(.2))

#################################### END e = 191 ############################  

############### e  = 192
e192 = pd.DataFrame(np.zeros((773, 3)))
e192.columns = ['a','b','c']




u=192

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e192.iloc[k,0] = dfp.iloc[i,0]
        e192.iloc[k,1] = dfp.iloc[i,1]
        e192.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e192.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e192.iloc[i,0]),4] == df200.iloc[int(e192.iloc[i,0]),4]:
        e192.iloc[i,2]= e192.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e192.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e192.iloc[i,0]),10] < 1 and  df200.iloc[int(e192.iloc[i,0]),10] <1) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e192.iloc[i,0]),10] < 2 and  df200.iloc[int(e192.iloc[i,0]),10] <2) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e192.iloc[i,0]),10] < 3 and  df200.iloc[int(e192.iloc[i,0]),10] <3) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e192.iloc[i,0]),10] < 4 and  df200.iloc[int(e192.iloc[i,0]),10] <4) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e192.iloc[i,0]),10] < 6 and  df200.iloc[int(e192.iloc[i,0]),10] <6) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e192.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e192.iloc[i,0]),5] < 700000 and  df200.iloc[int(e192.iloc[i,0]),5] <700000) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e192.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e192.iloc[i,0]),5] <1400000) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e192.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e192.iloc[i,0]),5] <2100000) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e192.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e192.iloc[i,0]),5] <2800000) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e192.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e192.iloc[i,0]),5] <3500000) :
        e192.iloc[i,2]= e192.iloc[i,2]+(.33*(.2))

#################################### END e = 192 ############################  

############### e  = 193
e193 = pd.DataFrame(np.zeros((773, 3)))
e193.columns = ['a','b','c']




u=193

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e193.iloc[k,0] = dfp.iloc[i,0]
        e193.iloc[k,1] = dfp.iloc[i,1]
        e193.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e193.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e193.iloc[i,0]),4] == df200.iloc[int(e193.iloc[i,0]),4]:
        e193.iloc[i,2]= e193.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e193.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e193.iloc[i,0]),10] < 1 and  df200.iloc[int(e193.iloc[i,0]),10] <1) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e193.iloc[i,0]),10] < 2 and  df200.iloc[int(e193.iloc[i,0]),10] <2) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e193.iloc[i,0]),10] < 3 and  df200.iloc[int(e193.iloc[i,0]),10] <3) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e193.iloc[i,0]),10] < 4 and  df200.iloc[int(e193.iloc[i,0]),10] <4) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e193.iloc[i,0]),10] < 6 and  df200.iloc[int(e193.iloc[i,0]),10] <6) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e193.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e193.iloc[i,0]),5] < 700000 and  df200.iloc[int(e193.iloc[i,0]),5] <700000) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e193.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e193.iloc[i,0]),5] <1400000) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e193.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e193.iloc[i,0]),5] <2100000) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e193.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e193.iloc[i,0]),5] <2800000) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e193.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e193.iloc[i,0]),5] <3500000) :
        e193.iloc[i,2]= e193.iloc[i,2]+(.33*(.2))

#################################### END e = 193 ############################  
############### e  = 194
e194 = pd.DataFrame(np.zeros((773, 3)))
e194.columns = ['a','b','c']




u=194

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e194.iloc[k,0] = dfp.iloc[i,0]
        e194.iloc[k,1] = dfp.iloc[i,1]
        e194.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e194.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e194.iloc[i,0]),4] == df200.iloc[int(e194.iloc[i,0]),4]:
        e194.iloc[i,2]= e194.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e194.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e194.iloc[i,0]),10] < 1 and  df200.iloc[int(e194.iloc[i,0]),10] <1) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e194.iloc[i,0]),10] < 2 and  df200.iloc[int(e194.iloc[i,0]),10] <2) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e194.iloc[i,0]),10] < 3 and  df200.iloc[int(e194.iloc[i,0]),10] <3) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e194.iloc[i,0]),10] < 4 and  df200.iloc[int(e194.iloc[i,0]),10] <4) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e194.iloc[i,0]),10] < 6 and  df200.iloc[int(e194.iloc[i,0]),10] <6) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e194.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e194.iloc[i,0]),5] < 700000 and  df200.iloc[int(e194.iloc[i,0]),5] <700000) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e194.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e194.iloc[i,0]),5] <1400000) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e194.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e194.iloc[i,0]),5] <2100000) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e194.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e194.iloc[i,0]),5] <2800000) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e194.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e194.iloc[i,0]),5] <3500000) :
        e194.iloc[i,2]= e194.iloc[i,2]+(.33*(.2))

#################################### END e = 194 ############################  
        
############### e  = 195
e195 = pd.DataFrame(np.zeros((773, 3)))
e195.columns = ['a','b','c']




u=195

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e195.iloc[k,0] = dfp.iloc[i,0]
        e195.iloc[k,1] = dfp.iloc[i,1]
        e195.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e195.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e195.iloc[i,0]),4] == df200.iloc[int(e195.iloc[i,0]),4]:
        e195.iloc[i,2]= e195.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e195.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e195.iloc[i,0]),10] < 1 and  df200.iloc[int(e195.iloc[i,0]),10] <1) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e195.iloc[i,0]),10] < 2 and  df200.iloc[int(e195.iloc[i,0]),10] <2) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e195.iloc[i,0]),10] < 3 and  df200.iloc[int(e195.iloc[i,0]),10] <3) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e195.iloc[i,0]),10] < 4 and  df200.iloc[int(e195.iloc[i,0]),10] <4) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e195.iloc[i,0]),10] < 6 and  df200.iloc[int(e195.iloc[i,0]),10] <6) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e195.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e195.iloc[i,0]),5] < 700000 and  df200.iloc[int(e195.iloc[i,0]),5] <700000) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e195.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e195.iloc[i,0]),5] <1400000) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e195.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e195.iloc[i,0]),5] <2100000) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e195.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e195.iloc[i,0]),5] <2800000) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e195.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e195.iloc[i,0]),5] <3500000) :
        e195.iloc[i,2]= e195.iloc[i,2]+(.33*(.2))

#################################### END e = 195 ############################  

############### e  = 196
e196 = pd.DataFrame(np.zeros((773, 3)))
e196.columns = ['a','b','c']




u=196

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e196.iloc[k,0] = dfp.iloc[i,0]
        e196.iloc[k,1] = dfp.iloc[i,1]
        e196.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e196.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e196.iloc[i,0]),4] == df200.iloc[int(e196.iloc[i,0]),4]:
        e196.iloc[i,2]= e196.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e196.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e196.iloc[i,0]),10] < 1 and  df200.iloc[int(e196.iloc[i,0]),10] <1) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e196.iloc[i,0]),10] < 2 and  df200.iloc[int(e196.iloc[i,0]),10] <2) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e196.iloc[i,0]),10] < 3 and  df200.iloc[int(e196.iloc[i,0]),10] <3) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e196.iloc[i,0]),10] < 4 and  df200.iloc[int(e196.iloc[i,0]),10] <4) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e196.iloc[i,0]),10] < 6 and  df200.iloc[int(e196.iloc[i,0]),10] <6) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e196.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e196.iloc[i,0]),5] < 700000 and  df200.iloc[int(e196.iloc[i,0]),5] <700000) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e196.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e196.iloc[i,0]),5] <1400000) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e196.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e196.iloc[i,0]),5] <2100000) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e196.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e196.iloc[i,0]),5] <2800000) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e196.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e196.iloc[i,0]),5] <3500000) :
        e196.iloc[i,2]= e196.iloc[i,2]+(.33*(.2))

#################################### END e = 196 ############################  
        
############### e  = 197
e197 = pd.DataFrame(np.zeros((773, 3)))
e197.columns = ['a','b','c']




u=197

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e197.iloc[k,0] = dfp.iloc[i,0]
        e197.iloc[k,1] = dfp.iloc[i,1]
        e197.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e197.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e197.iloc[i,0]),4] == df200.iloc[int(e197.iloc[i,0]),4]:
        e197.iloc[i,2]= e197.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e197.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e197.iloc[i,0]),10] < 1 and  df200.iloc[int(e197.iloc[i,0]),10] <1) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e197.iloc[i,0]),10] < 2 and  df200.iloc[int(e197.iloc[i,0]),10] <2) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e197.iloc[i,0]),10] < 3 and  df200.iloc[int(e197.iloc[i,0]),10] <3) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e197.iloc[i,0]),10] < 4 and  df200.iloc[int(e197.iloc[i,0]),10] <4) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e197.iloc[i,0]),10] < 6 and  df200.iloc[int(e197.iloc[i,0]),10] <6) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e197.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e197.iloc[i,0]),5] < 700000 and  df200.iloc[int(e197.iloc[i,0]),5] <700000) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e197.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e197.iloc[i,0]),5] <1400000) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e197.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e197.iloc[i,0]),5] <2100000) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e197.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e197.iloc[i,0]),5] <2800000) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e197.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e197.iloc[i,0]),5] <3500000) :
        e197.iloc[i,2]= e197.iloc[i,2]+(.33*(.2))

#################################### END e = 197 ############################  
        
############### e  = 198
e198 = pd.DataFrame(np.zeros((773, 3)))
e198.columns = ['a','b','c']




u=198

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e198.iloc[k,0] = dfp.iloc[i,0]
        e198.iloc[k,1] = dfp.iloc[i,1]
        e198.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e198.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e198.iloc[i,0]),4] == df200.iloc[int(e198.iloc[i,0]),4]:
        e198.iloc[i,2]= e198.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e198.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e198.iloc[i,0]),10] < 1 and  df200.iloc[int(e198.iloc[i,0]),10] <1) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e198.iloc[i,0]),10] < 2 and  df200.iloc[int(e198.iloc[i,0]),10] <2) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e198.iloc[i,0]),10] < 3 and  df200.iloc[int(e198.iloc[i,0]),10] <3) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e198.iloc[i,0]),10] < 4 and  df200.iloc[int(e198.iloc[i,0]),10] <4) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e198.iloc[i,0]),10] < 6 and  df200.iloc[int(e198.iloc[i,0]),10] <6) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e198.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e198.iloc[i,0]),5] < 700000 and  df200.iloc[int(e198.iloc[i,0]),5] <700000) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e198.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e198.iloc[i,0]),5] <1400000) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e198.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e198.iloc[i,0]),5] <2100000) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e198.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e198.iloc[i,0]),5] <2800000) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e198.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e198.iloc[i,0]),5] <3500000) :
        e198.iloc[i,2]= e198.iloc[i,2]+(.33*(.2))

#################################### END e = 198 ############################  
 
############### e  = 199
e199 = pd.DataFrame(np.zeros((773, 3)))
e199.columns = ['a','b','c']




u=199

nodes = list([ ])
for i in range(0, np.shape(dfp)[0]):

    if dfp.iloc[i,0] == u:
        nodes.append(int(dfp.iloc[i,1]))
    
    
    if dfp.iloc[i,1] == u:
        nodes.append(int(dfp.iloc[i,0]))
  
    
    
  
nodes = list(set(nodes))   #  remove duplicates

nodes.sort()

nodes = np.array(nodes)


k=0
for i in range(0, np.shape(dfp)[0]):
    
    if (dfp.iloc[i,0] in nodes) and (dfp.iloc[i,1] in nodes):
        e199.iloc[k,0] = dfp.iloc[i,0]
        e199.iloc[k,1] = dfp.iloc[i,1]
        e199.iloc[k,2] = .001
        k=k+1



#####################   NODES complete
        
        
        
        
###########################   Loop for groups        
for i in range(0,773):
    
    if e199.iloc[i,2]== 0:
        continue
        
    elif df200.iloc[int(e199.iloc[i,0]),4] == df200.iloc[int(e199.iloc[i,0]),4]:
        e199.iloc[i,2]= e199.iloc[i,2]+.33
        


###########################   Loop for review score        
for i in range(0,773):
    
    if e199.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e199.iloc[i,0]),10] < 1 and  df200.iloc[int(e199.iloc[i,0]),10] <1) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.2))
    
    elif (df200.iloc[int(e199.iloc[i,0]),10] < 2 and  df200.iloc[int(e199.iloc[i,0]),10] <2) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.4))
        
    elif (df200.iloc[int(e199.iloc[i,0]),10] < 3 and  df200.iloc[int(e199.iloc[i,0]),10] <3) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.6))
        
    elif (df200.iloc[int(e199.iloc[i,0]),10] < 4 and  df200.iloc[int(e199.iloc[i,0]),10] <4) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.8))
        
    elif (df200.iloc[int(e199.iloc[i,0]),10] < 6 and  df200.iloc[int(e199.iloc[i,0]),10] <6) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(1))
     
    
    
    
###########################   Loop for salerank        
for i in range(0,773):
    
    if e199.iloc[i,2]== 0:
        continue  
    elif (df200.iloc[int(e199.iloc[i,0]),5] < 700000 and  df200.iloc[int(e199.iloc[i,0]),5] <700000) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(1))

    
    elif (df200.iloc[int(e199.iloc[i,0]),5] < 1400000 and  df200.iloc[int(e199.iloc[i,0]),5] <1400000) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.8))
    
    elif (df200.iloc[int(e199.iloc[i,0]),5] < 2100000 and  df200.iloc[int(e199.iloc[i,0]),5] <2100000) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.6))
    
    elif (df200.iloc[int(e199.iloc[i,0]),5] < 2800000 and  df200.iloc[int(e199.iloc[i,0]),5] <2800000) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.4))
    
    elif (df200.iloc[int(e199.iloc[i,0]),5] < 3500000 and  df200.iloc[int(e199.iloc[i,0]),5] <3500000) :
        e199.iloc[i,2]= e199.iloc[i,2]+(.33*(.2))

#################################### END e = 199 ############################         
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:42:14 2018

@author: warrenkeil
"""

 
import random
import sys
import os  
import math
import numpy as np     
import matplotlib.pyplot as plt    
import pandas as pd
import IPython      
import scipy as sci 
import pylab 
import sympy as sym 
import dionysus as d 
import csv
import networkx as nx




'''
e111


def adj(e):
    a = pd.DataFrame(np.zeros((200, 200)))
    
    for i in range(0,773):
        
        if e.iloc[i,0] !=0 and e.iloc[i,1] !=0 :
            a.iloc[int(e.iloc[i,0]), int(e.iloc[i,1])] = e.iloc[i,2]
        
    return a

        

a0 = adj(e0)



dlm = discrete_laplacian_eigen(a40)


ddm = diff_dist_mat(dlm[0],dlm[1],.1)   #  t is last arg of function. we can change it


flow = d.fill_freudenthal(ddm)
#rip = d.fill_rips(ddm,2,.3)

p = d.homology_persistence(flow)
dgms = d.init_diagrams(p, flow)


d.plot.plot_bars(dgms[0])
d.plot.plot_bars(dgms[1])
      

'''
#####################3


def adj(e):
    a = pd.DataFrame(np.zeros((200, 200)))
    
    for i in range(0,773):
        
        if e.iloc[i,0] !=0 and e.iloc[i,1] !=0 :
            a.iloc[int(e.iloc[i,0]), int(e.iloc[i,1])] = e.iloc[i,2]
     
        
    dlm = discrete_laplacian_eigen(a)   
    ddm = diff_dist_mat(dlm[0],dlm[1],.1) 
    flow = d.fill_freudenthal(ddm)
    p = d.homology_persistence(flow)
    dgms = d.init_diagrams(p, flow)
    return a,dgms

        

a0,d0 = adj(e0)
a1,d1 = adj(e1)     
a2,d2 = adj(e2)        
a3,d3 = adj(e3)
a4,d4 = adj(e4)
a5,d5 = adj(e5)
a6,d6 = adj(e6)
a7,d7 = adj(e7)
a8,d8 = adj(e8)
a9,d9 = adj(e9)

print('nine done')
a10,d10 = adj(e10)
a11,d11 = adj(e11)
a12,d12 = adj(e12)
a13,d13 = adj(e13)
a14,d14 = adj(e14)
a15,d15 = adj(e15)
a16,d16 = adj(e16)
a17,d17 = adj(e17)
a18,d18 = adj(e18)
a19,d19 = adj(e19)
a20,d20 = adj(e20)
a21,d21 = adj(e21)
a22,d22 = adj(e22)
a23,d23 = adj(e23)
a24,d24 = adj(e24)
a25,d25 = adj(e25)
a26,d26 = adj(e26)
a27,d27 = adj(e27)
a28,d28 = adj(e28)
a29,d29 = adj(e29)
a30,d30 = adj(e30)
a31,d31 = adj(e31)
a32,d32 = adj(e32)        
a33,d33 = adj(e33)
a34,d34 = adj(e34)
print('34')
a35,d35 = adj(e35)
a36,d36 = adj(e36)
a37,d37 = adj(e37)
a38,d38 = adj(e38)
a39,d39 = adj(e39)
a40,d40 = adj(e40)
a41,d41 = adj(e41)
a42,d42 = adj(e42)
a43,d43 = adj(e43)
a44,d44 = adj(e44)
a45,d45 = adj(e45)
a46,d46 = adj(e46)
a47,d47 = adj(e47)
a48,d48 = adj(e48)
a49,d49 = adj(e49)
a50,d50 = adj(e50)
a51,d51 = adj(e51)
a52,d52=adj(e52)
a53,d53=adj(e53)
a54,d54=adj(e54)
a55,d55=adj(e55)

print('55 done')
a56,d56=adj(e56)
a57,d57=adj(e57)
a58,d58=adj(e58)
a59,d59=adj(e59)
a60,d60=adj(e60)
a61,d61=adj(e61)
a62,d62=adj(e62)
a63,d63=adj(e63)
a64,d64=adj(e64)
a65,d65=adj(e65)
a66,d66=adj(e66)
a67,d67=adj(e67)
a68,d68=adj(e68)
a69,d69=adj(e69)
a70,d70=adj(e70)
a71,d71=adj(e71)
a72,d72=adj(e72)
a73,d73=adj(e73)
a74,d74=adj(e74)
a75,d75=adj(e75)
a76,d76=adj(e76)
a77,d77=adj(e77)
a78,d78=adj(e78)
a79,d79=adj(e79)
a80,d80=adj(e80)
a81,d81=adj(e81)
a82,d82=adj(e82)
a83,d83=adj(e83)
a84,d84=adj(e84)
a85,d85=adj(e85)
print('85')
a86,d86=adj(e86)
a87,d87=adj(e87)
a88,d88=adj(e88)
a89,d89=adj(e89)
a90,d90=adj(e90)
a91,d91=adj(e91)
a92,d92=adj(e92)
a93,d93=adj(e93)
a94,d94=adj(e94)
a95,d95=adj(e95)
a96,d96=adj(e96)
a97,d97=adj(e97)
a98,d98=adj(e98)
a99,d99=adj(e99)
a100,d100=adj(e100)
a101,d101=adj(e101)
a102,d102=adj(e102)
a103,d103=adj(e103)
a104,d104=adj(e104)
a105,d105=adj(e105)
a106,d106=adj(e106)
a107,d107=adj(e107)
a108,d108=adj(e108)
a109,d109=adj(e109)
a110,d110=adj(e110)
a111,d111=adj(e111)
a112,d112=adj(e112)
a113,d113=adj(e113)
a114,d114=adj(e114)
a115,d115=adj(e115)
a116,d116=adj(e116)
a117,d117=adj(e117)
a118,d118=adj(e118)
a119,d119=adj(e119)
a120,d120=adj(e120)
a121,d121=adj(e121)
a122,d122=adj(e122)
a123,d123=adj(e123)
a124,d124=adj(e124)
a125,d125=adj(e125)
a126,d126=adj(e126)
print('126')
a127,d127=adj(e127)
a128,d128=adj(e128)
a129,d129=adj(e129)
a130,d130=adj(e130)
a131,d131=adj(e131)
a132,d132=adj(e132)
a133,d133=adj(e133)
a134,d134=adj(e134)
a135,d135=adj(e135)
a136,d136=adj(e136)
a137,d137=adj(e137)
a138,d138=adj(e138)
a139,d139=adj(e139)
a140,d140=adj(e140)
a141,d141=adj(e141)
a142,d142=adj(e142)
a143,d143=adj(e143)
print('143 done')
a144,d144=adj(e144)
a145,d145=adj(e145)
a146,d146=adj(e146)
a147,d147=adj(e147)
a148,d148=adj(e148)
a149,d149=adj(e149)
a150,d150=adj(e150)
a151,d151=adj(e151)
a152,d152=adj(e152)
a153,d153=adj(e153)
a154,d154=adj(e154)
a155,d155=adj(e155)
a156,d156=adj(e156)
a157,d157=adj(e157)
a158,d158=adj(e158)
a159,d159=adj(e159)
a160,d160=adj(e160)
a161,d161=adj(e161)
a162,d162=adj(e162)
a163,d163=adj(e163)
a164,d164=adj(e164)
a165,d165=adj(e165)
a166,d166=adj(e166)
a167,d167=adj(e167)
a168,d168=adj(e168)
a169,d169=adj(e169)
a170,d170=adj(e170)
a171,d171=adj(e171)
a172,d172=adj(e172)
a173,d173=adj(e173)
a174,d174=adj(e174)
a175,d175=adj(e175)
a176,d176=adj(e176)
a177,d177=adj(e177)
print('177')
a178,d178=adj(e178)
a179,d179=adj(e179)
a180,d180=adj(e180)
a181,d181=adj(e181)
a182,d182=adj(e182)
a183,d183=adj(e183)
a184,d184=adj(e184)
a185,d185=adj(e185)
a186,d186=adj(e186)
a187,d187=adj(e187)
a188,d188=adj(e188)
a189,d189=adj(e189)
a190,d190=adj(e190)
a191,d191=adj(e191)
a192,d192=adj(e192)
a193,d193=adj(e193)
a194,d194=adj(e194)
a195,d195=adj(e195)
a196,d196=adj(e196)
a197,d197=adj(e197)
a198,d198=adj(e198)
a199,d199=adj(e199)

print('done') 



    
     
    
    
       
    
       
    
      
    
    
        
    
    
    






