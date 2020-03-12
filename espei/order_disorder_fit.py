# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:57:22 2019

@author: suh960
"""
from itertools import *
import json 
from operator import mul
from functools import reduce
from collections import defaultdict
import numpy as np
import copy 
import sys
import math
import string
import os

def decide_symmetry(sublattice_model,ratios_in):
    symmetry=True
    ratios_in2=copy.deepcopy(ratios_in)
    for i in range(len(sublattice_model)):

        if sublattice_model[i]==['VA']:
            del ratios_in2[i]
            continue;

    if ratios_in2.count(ratios_in2[0])==len(ratios_in2):
        
        pass;
    else:
        symmetry=False
    return symmetry

            
def find_mole_data(site_data,ratios,symmetry):
    mole_data=copy.deepcopy(site_data)
    for i in mole_data:
        e=copy.deepcopy(i)
        array=[]
        ratios_al=0
        for j in range(len(i)):
            if type(i[j])==list and len(i[j])>1:
                i[j]=np.array(i[j])*ratios[j]
                if len(array)==0:
                    array=i[j]
                    ratios_al=ratios[j]
                else:
                    array1=i[j]
                    ratios_al=ratios_al+ratios[j]
                    array=np.vstack((array,array1))
        ave=np.sum(array,axis=0)/(ratios_al)
        for h in range(len(e)):
            if type(e[h])==list and len(e[h])>1:
                i[h]=list(ave)
    return mole_data
def SM_modify(originals, site,mole,ratios):
    R=8.31451
    H=originals.tolist()
                           
    for i in range(len(H)):
        H[i][0]=float("{:.2g}".format(H[i][0]))
    originals=np.mat(H)

    SM_fitting_result=[]
    for i in range(len(site)):
        I=copy.deepcopy(site[i])
        M=copy.deepcopy(mole[i])
        diff=[]

        for j in range(len(I)):
            J=I[j]
            K=M[j]
            

            if type(J) !=int:
                for k in range(len(J)):
                    if J[k]==0:
                        J[k]=0.0000000001;
                    if J[k]==1:
                        J[k]=0.9999999999;
                    if K[k]==0:
                        K[k]=0.0000000001;
                    if K[k]==1:
                        K[k]=0.9999999999;
                    J[k]=J[k]*math.log(J[k])
                    K[k]=K[k]*math.log(K[k])
            
            if type(J) !=int:
                
                diff.append(sum(J)*ratios[j]-sum(K)*ratios[j])
            else:
                diff.append((J-K))

        SM_fitting_result.append(sum(diff)*R)
    SM_fitting_result=np.mat(SM_fitting_result).transpose()
    
 
    SM_fitting_change=SM_fitting_result.tolist()
    for i in range(len(SM_fitting_change)):
        SM_fitting_change[i][0]=float("{:.2g}".format(SM_fitting_change[i][0]))
    SM_fitting_result=np.mat(SM_fitting_change)
#
#    b=originals+SM_fitting_result
#    for i in range(len(b)):
#        if b[i][0]>0:


    return -(originals+SM_fitting_result)

            

                

def find_endmember(sublatti_model,symmetry):
    sublattice_model=copy.deepcopy(sublatti_model)
    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])
    length=len(sublattice_model)
    x=product(sublattice_model[0], repeat=length)
    x=list(x)

    b=copy.deepcopy(x)
    for n in b:
        n1=list(n)
        if n.count(n1[0])==length:
            x.remove(n);
    uns=[]
    for i in range(len(x)):
        x[i]=list(x[i])
        uns.append(list(x[i]))
        x[i].sort()
    
    end=[]
    [end.append(i) for i in x if not i in end]

    if symmetry==True:
        return end
    else:
        return uns
def find_inter2(sublatti_model):
    sublattice_model=copy.deepcopy(sublatti_model)
    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])


    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])

    length=len(sublattice_model)
    x=product(sublattice_model[0], repeat=length-1)
    y=combinations(sublattice_model[0],2)
    x=list(x)
    y=list(y)
    for i in range(len(x)):
        x[i]=list(x[i])
    for j in range(len(y)):
        y[j]=list(y[j])
    z=product(x,y)

    return list(z)


def find_crossinter2(sublatti_model):
    sublattice_model=copy.deepcopy(sublatti_model)
    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])

    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])

    length=len(sublattice_model)
    x=product(sublattice_model[0], repeat=length-2)
    y=combinations(sublattice_model[0],2)
    x=list(x)
    y=list(y)

    for i in range(len(x)):
        x[i]=list(x[i])
    for j in range(len(y)):
        y[j]=list(y[j])

    z=product(y,y)
    z=list(z)
    z1=copy.deepcopy(z)

    for i in z1:
        i=list(i)
        if i[0]==i[1]:
            pass;
        else:
            g=copy.deepcopy(i)
            g[0]=i[1]
            g[1]=i[0]
            if tuple(g) in z and tuple(i) in z:
                z.remove(tuple(i))

    if length-2>0:
        z=product(x,z)  

    return list(z)


def find_inter3(sublatti_model):
    sublattice_model=copy.deepcopy(sublatti_model)
    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])

    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])
    
    length=len(sublattice_model)
    x=product(sublattice_model[0], repeat=length-1)
    y=combinations(sublattice_model[0],3)
    x=list(x)
    y=list(y)
    for i in range(len(x)):
        x[i]=list(x[i])
    for j in range(len(y)):
        y[j]=list(y[j])
    z=product(x,y)
    q=product(y,x)

    return list(z)



def paramemter(site_data,endmember,sublattice_model,inter2,cross_inter2,symmetry,inter3=None):
    exa=sublattice_model[0]
    if symmetry==True:

        end=defaultdict(list)
    
        length=len(sublattice_model)
        x=product(sublattice_model[0], repeat=length)
        x=list(x)
        b=copy.deepcopy(x)
        for n in b:
            n1=list(n)
            if n.count(n1[0])==length:
                x.remove(n);
    
        undetele_end=defaultdict(list)

        for i in x:
            
            j=sorted(list(i))
    
            undetele_end[tuple(j)].append(i)
    

        for i in site_data:
            for j in endmember:
                
                j=tuple(j)
                a=undetele_end[j]
              
                value1=[]
              
                len_end=len(j)-1
                e2=0

                for n in a:
                    value1=[]

                    for k in range(len(n)):

                        n1=exa.index(n[k])
                        
                        value1_te=i[k]
                        
                        value1.append(value1_te[n1])
                
                        if k==len_end:
                            e1=reduce(mul,value1)
                     
                    e2=e2+e1

                end[j].append(e2)


        inter2_reL0=defaultdict(list)
        inter2_reL1=defaultdict(list)
        inter2_reL2=defaultdict(list)
        for i in site_data:
            
            for j in inter2:
                
                j=list(j)
                h=copy.deepcopy(j)
                for e in range(len(h)):
                    h[e]=tuple(h[e])

                inter2_ele=defaultdict(list)
                L0_va=0
                L1_va=0
                L2_va=0
                for k in range(len(j[0])+1):
                    c=copy.deepcopy(j[0])
                    c.insert(k,j[-1])
                    
                    inter2_ele=defaultdict(list)
                    value=[]
        
               
                    L0=[]
                    L1=[]
                    L2=[]
                    for l in range(len(c)):
                        a=l
                        y=i[a]
                        
                     
                        if type(c[l])==list:
                            
                            
                            for m in c[l]:
                                b=exa.index(m)
                                inter2_ele[tuple(c[l])].append(y[b])
                                
                        else:
                            d=exa.index(c[l])
                            inter2_ele[c[l]].append(y[d])
     
                    for keys,values in inter2_ele.items():
                        
                        
                        if type(keys)==tuple:
                            
                            L0.append(reduce(mul,values))
                            L1.append(reduce(mul,values)*(values[0]-values[1]))
                            L2.append(reduce(mul,values)*(values[0]-values[1])**2)
                        else:
                            
                            L0.append(reduce(mul,values))
                            L1.append(reduce(mul,values))
                            L2.append(reduce(mul,values))
    
                    L0_va=L0_va+reduce(mul,L0)
                    L1_va=L1_va+reduce(mul,L1)
                    L2_va=L2_va+reduce(mul,L2)
                    
                    if k==len(j[0]):
                        inter2_reL0[tuple(h)].append(L0_va)
                        inter2_reL1[tuple(h)].append(L1_va)
                        inter2_reL2[tuple(h)].append(L2_va)
         
        inter2_new_reL0=copy.deepcopy(inter2_reL0)
        inter2_new_reL1=copy.deepcopy(inter2_reL1)
        inter2_new_reL2=copy.deepcopy(inter2_reL2)
        for key,item in inter2_reL0.items():
            keys=copy.deepcopy(key)
            keys=list(keys)
            if len(key[0])>1:
                a=list(key[0])
                a.sort()
                keys[0]=tuple(a)
                if key==tuple(keys):
                    continue;
                else:
                    for nu in range(len(item)):
                        inter2_new_reL0[tuple(keys)][nu]=inter2_new_reL0[tuple(keys)][nu]+inter2_new_reL0[key][nu]
                        inter2_new_reL1[tuple(keys)][nu]=inter2_new_reL1[tuple(keys)][nu]+inter2_new_reL1[key][nu]
                        inter2_new_reL2[tuple(keys)][nu]=inter2_new_reL2[tuple(keys)][nu]+inter2_new_reL2[key][nu]
                    del inter2_new_reL0[key]
                    del inter2_new_reL1[key]
                    del inter2_new_reL2[key]                     
    
    
    
        cross_inter2_reL0=defaultdict(list)
        cross_inter2_reL1=defaultdict(list)
        cross_inter2_reL2=defaultdict(list)
        for i in site_data:
            
                
            for j in cross_inter2:
                    
                j=list(j)
                h=copy.deepcopy(j)

                for e in range(len(h)):
                    h[e]=list(h[e])
                    for x in range(len(h[e])):
                        if type(h[e][x])==list:
                            h[e][x]=tuple(h[e][x])
                    h[e]=tuple(h[e])
                    j[e]=list(j[e])

                cross_inter_ele=[]
               
                L0=[]
                L1=[]
                L2=[]

                if type(h[1][1])==tuple:

                    for k in range(len(j[0])+1):

                        c=copy.deepcopy(j[0])
                        c.insert(k,j[-1][0])
                   
                        for l in range(len(j[0])+2):
                            d=copy.deepcopy(c)
                            d.insert(l,j[-1][1])
                            if d not in cross_inter_ele:
                                cross_inter_ele.append(d)
                            else:
                                continue;
                else: 
                    cross_inter_ele.append(j)
                    o=[]
                    o.append(j[1])
                    o.append(j[0])

                    if o not in cross_inter_ele:
                        cross_inter_ele.append(o)
                        repeat=1
    
                    
    

                for n in range(len(cross_inter_ele)):
                    cross_inter2_ele=defaultdict(list)
                    L0_va=0
                    L1_va=0
                    L2_va=0
    
                    
                    
                    nume=1

                    for k in range(len(cross_inter_ele[n])):

                       
                        
                        
                        a=k
                        y=i[k]

                     

                        if type(cross_inter_ele[n][k])==list:
                            
                            
                            
                            for m in cross_inter_ele[n][k]:

                                b=exa.index(m)
                                cross_inter2_ele[nume].append(y[b])
                            nume=nume+1
                                
                        else:

                            d=exa.index(cross_inter_ele[n][k])
                            cross_inter2_ele[cross_inter_ele[n][k]].append(y[d])

                    L0_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])
                    L1_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])*(cross_inter2_ele[1][0]-cross_inter2_ele[2][0])
                    L2_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])*(cross_inter2_ele[1][1]-cross_inter2_ele[2][1])

                    L01=[]
                    L11=[]
                    L21=[]

                    for keys in cross_inter2_ele.keys():

                        if type(h[1][1])==tuple:
                            pass;
                        else:
                            L01=[1]
                            L11=[1]
                            L21=[1]


                        if keys ==1 or keys==2:
                            pass;     
                        else:

                            L01.append(reduce(mul,cross_inter2_ele[keys]))
                            L11.append(reduce(mul,cross_inter2_ele[keys]))
                            L21.append(reduce(mul,cross_inter2_ele[keys]))
                    
                    L0.append(reduce(mul,L01)*L0_va)
                    L1.append(reduce(mul,L11)*L1_va)
                    L2.append(reduce(mul,L21)*L2_va)

                
                    
                    

                cross_inter2_reL0[tuple(h)].append(sum(L0))
                cross_inter2_reL1[tuple(h)].append(sum(L1))
                cross_inter2_reL2[tuple(h)].append(sum(L2))

        cross_inter2_new_reL0=copy.deepcopy(cross_inter2_reL0)
        cross_inter2_new_reL1=copy.deepcopy(cross_inter2_reL1)
        cross_inter2_new_reL2=copy.deepcopy(cross_inter2_reL2)
        for key,item in cross_inter2_reL0.items():
            keys=copy.deepcopy(key)
            keys=list(keys)
            if len(key[0])>1:
                a=list(key[0])
                a.sort()
                keys[0]=tuple(a)
                if key==tuple(keys):
                    continue;
                else:
                    for nu in range(len(item)):
                        cross_inter2_new_reL0[tuple(keys)][nu]=cross_inter2_new_reL0[tuple(keys)][nu]+cross_inter2_new_reL0[key][nu]
                        cross_inter2_new_reL1[tuple(keys)][nu]=cross_inter2_new_reL1[tuple(keys)][nu]+cross_inter2_new_reL1[key][nu]
                        cross_inter2_new_reL2[tuple(keys)][nu]=cross_inter2_new_reL2[tuple(keys)][nu]+cross_inter2_new_reL2[key][nu]
                    del cross_inter2_new_reL0[key]
                    del cross_inter2_new_reL1[key]
                    del cross_inter2_new_reL2[key]    
                
        if inter3:
            inter3_reL0=defaultdict(list)
            inter3_reL1=defaultdict(list)
            inter3_reL2=defaultdict(list)
            for i in site_data:
                
                for j in inter3:
                    
                    j=list(j)
                    
                    h=copy.deepcopy(j)
                    for e in range(len(h)):
                        h[e]=tuple(h[e])
                            
                    inter3_ele=defaultdict(list)
                    L0_va=0
                    L1_va=0
                    L2_va=0
                    for k in range(len(j[0])+1):
                   
                        c=copy.deepcopy(j[0])
                        c.insert(k,j[-1])
                  
                        inter3_ele=defaultdict(list)
                        value=[]
        
               
                        L0=[]
                        L1=[]
                        L2=[]
                        for l in range(len(c)):
                            a=l
                            y=i[a]
                             
                            if type(c[l])==list:
                            
                            
                                for m in c[l]:
                                    b=exa.index(m)
                                    inter3_ele[tuple(c[l])].append(y[b])
                                
                            else:
                                d=exa.index(c[l])
                                inter3_ele[c[l]].append(y[d])

                        for key,value in inter3_ele.items():
                    
                            if len(key)>2:
                                L0.append(reduce(mul,value)*(value[0]+(1-value[0]-value[1]-value[2])/3))
                                L1.append(reduce(mul,value)*(value[1]+(1-value[0]-value[1]-value[2])/3))
                                L2.append(reduce(mul,value)*(value[2]+(1-value[0]-value[1]-value[2])/3))
                            else:
                                L0.append(value[0])
                                L1.append(value[0])
                                L2.append(value[0])
                
                        L0_va=L0_va+reduce(mul,L0)
                        L1_va=L1_va+reduce(mul,L1)
                        L2_va=L2_va+reduce(mul,L2)
                        if k==len(j[0]):
                            inter3_reL0[tuple(h)].append(L0_va)
                            inter3_reL1[tuple(h)].append(L1_va)
                            inter3_reL2[tuple(h)].append(L2_va)



    
            
            return end,inter2_reL0,inter2_reL1,inter2_reL2,cross_inter2_reL0,cross_inter2_reL1,cross_inter2_reL2,inter3_reL0,inter3_reL1,inter3_reL2
        else:
            return end,inter2_reL0,inter2_reL1,inter2_reL2,cross_inter2_reL0,cross_inter2_reL1,cross_inter2_reL2
    else:
        end=defaultdict(list)
        inter2_reL0=defaultdict(list)
        inter2_reL1=defaultdict(list)
        inter2_reL2=defaultdict(list)
        cross_inter2_reL0=defaultdict(list)
        cross_inter2_reL1=defaultdict(list)
        cross_inter2_reL2=defaultdict(list)
        inter3_reL0=defaultdict(list)
        inter3_reL1=defaultdict(list)
        inter3_reL2=defaultdict(list)

        for i in site_data:
            for j in endmember:
                value=[]
                for k in range(len(j)):
                    p=exa.index(j[k])
                    y=i[k]
                    value.append(y[p])
                end[tuple(j)].append(reduce(mul,value))

            for j in inter2:
                
                j=list(j)
                h=copy.deepcopy(j)
                for e in range(len(h)):
                    if type(h[e])==list:
                        h[e].sort()
                    h[e]=tuple(h[e])

                inter2_ele=defaultdict(list)
                L0_va=0
                L1_va=0
                L2_va=0
                for k in range(len(j[0])+1):
                    c=copy.deepcopy(j[0])
                    c.insert(k,j[-1])
                    
                    inter2_ele=defaultdict(list)
                    value=[]
        
               
                    L0=[]
                    L1=[]
                    L2=[]
                    for l in range(len(c)):
                        a=l
                        y=i[a]
                        
                     
                        if type(c[l])==list:
                            
                            
                            for m in c[l]:
                                b=exa.index(m)
                                inter2_ele[tuple(c[l])].append(y[b])
                                
                        else:
                            d=exa.index(c[l])
                            inter2_ele[c[l]].append(y[d])
     
                    for keys,values in inter2_ele.items():
                        
                        
                        if type(keys)==tuple:
                            
                            L0.append(reduce(mul,values))
                            L1.append(reduce(mul,values)*(values[0]-values[1]))
                            L2.append(reduce(mul,values)*(values[0]-values[1])**2)
                        else:
                            
                            L0.append(reduce(mul,values))
                            L1.append(reduce(mul,values))
                            L2.append(reduce(mul,values))
                    for l in range(len(c)):
                        if type(c[l])==list:
                            c[l]=tuple(c[l])
                            
                    inter2_reL0[tuple(c)].append(reduce(mul,L0))
                    inter2_reL1[tuple(c)].append(reduce(mul,L1))
                    inter2_reL2[tuple(c)].append(reduce(mul,L2))
      
            for j in cross_inter2:
                    
                j=list(j)
                h=copy.deepcopy(j)

                for e in range(len(h)):
                    h[e]=list(h[e])
                    for x in range(len(h[e])):
                        if type(h[e][x])==list:
                            h[e][x]=tuple(h[e][x])
                    h[e]=tuple(h[e])
                    j[e]=list(j[e])

                cross_inter_ele=[]
               
                L0=[]
                L1=[]
                L2=[]

                if type(h[1][1])==tuple:


                    for k in range(len(j[0])+1):

                        c=copy.deepcopy(j[0])
                        c.insert(k,j[-1][0])
                   
                        for l in range(len(j[0])+2):
                            d=copy.deepcopy(c)
                            d.insert(l,j[-1][1])
                            if d not in cross_inter_ele:
                                cross_inter_ele.append(d)
                            else:
                                continue;
                else: 
                    cross_inter_ele.append(j)
                    o=[]
                    o.append(j[1])
                    o.append(j[0])

                    if o not in cross_inter_ele:
                        cross_inter_ele.append(o)
                        repeat=1

                for n in range(len(cross_inter_ele)):
                    cross_inter2_ele=defaultdict(list)
                    L0_va=0
                    L1_va=0
                    L2_va=0
                    nam=copy.deepcopy(cross_inter_ele[n])
                    for f in range(len(nam)):
                        if type(nam[f])==list:
                            nam[f]=tuple(nam[f])

                    nume=1

                    for k in range(len(cross_inter_ele[n])):

                        a=k
                        y=i[k]

                        if type(cross_inter_ele[n][k])==list:
                            
                            
                            
                            for m in cross_inter_ele[n][k]:

                                b=exa.index(m)
                                cross_inter2_ele[nume].append(y[b])
                            nume=nume+1
                                
                        else:

                            d=exa.index(cross_inter_ele[n][k])
                            cross_inter2_ele[cross_inter_ele[n][k]].append(y[d])

                    L0_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])
                    L1_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])*(cross_inter2_ele[1][0]-cross_inter2_ele[2][0])
                    L2_va=reduce(mul,cross_inter2_ele[1])*reduce(mul,cross_inter2_ele[2])*(cross_inter2_ele[1][1]-cross_inter2_ele[2][1])

                    L01=[]
                    L11=[]
                    L21=[]

                    for keys in cross_inter2_ele.keys():

                        if type(h[1][1])==tuple:
                            pass;
                        else:
                            L01=[1]
                            L11=[1]
                            L21=[1]


                        if keys ==1 or keys==2:
                            pass;     
                        else:

                            L01.append(reduce(mul,cross_inter2_ele[keys]))
                            L11.append(reduce(mul,cross_inter2_ele[keys]))
                            L21.append(reduce(mul,cross_inter2_ele[keys]))
                    cross_inter2_reL0[tuple(nam)].append(reduce(mul,L01)*L0_va)
                    cross_inter2_reL1[tuple(nam)].append(reduce(mul,L11)*L1_va)
                    cross_inter2_reL2[tuple(nam)].append(reduce(mul,L21)*L2_va)


        if inter3:
            inter3_reL0=defaultdict(list)
            inter3_reL1=defaultdict(list)
            inter3_reL2=defaultdict(list)
            for i in site_data:
                
                for j in inter3:
                    
                    j=list(j)
                    
                    h=copy.deepcopy(j)
                    for e in range(len(h)):
                        h[e]=tuple(h[e])
                            
                    inter3_ele=defaultdict(list)
                    L0_va=0
                    L1_va=0
                    L2_va=0
                    for k in range(len(j[0])+1):
                   
                        c=copy.deepcopy(j[0])
                        c.insert(k,j[-1])
                        nam=copy.deepcopy(c)
                        for o in range(len(nam)):
                            if type(nam[o])==list:
                                nam[o]=tuple(nam[o])

                  
                        inter3_ele=defaultdict(list)
                        value=[]
        
               
                        L0=[]
                        L1=[]
                        L2=[]
                        for l in range(len(c)):
                            a=l
                            y=i[a]
                             
                            if type(c[l])==list:
                            
                            
                                for m in c[l]:
                                    b=exa.index(m)
                                    inter3_ele[tuple(c[l])].append(y[b])
                                
                            else:
                                d=exa.index(c[l])
                                inter3_ele[c[l]].append(y[d])

                        for key,value in inter3_ele.items():
                    
                            if len(key)>2:
                                L0.append(reduce(mul,value)*(value[0]+(1-value[0]-value[1]-value[2])/3))
                                L1.append(reduce(mul,value)*(value[1]+(1-value[0]-value[1]-value[2])/3))
                                L2.append(reduce(mul,value)*(value[2]+(1-value[0]-value[1]-value[2])/3))
                            else:
                                L0.append(value[0])
                                L1.append(value[0])
                                L2.append(value[0])
                        inter3_reL0[tuple(nam)].append(reduce(mul,L0))
                        inter3_reL1[tuple(nam)].append(reduce(mul,L1))
                        inter3_reL2[tuple(nam)].append(reduce(mul,L2))

        if inter3:
            return end,inter2_reL0,inter2_reL1,inter2_reL2,cross_inter2_reL0,cross_inter2_reL1,cross_inter2_reL2,inter3_reL0,inter3_reL1,inter3_reL2
        else:  
            return end,inter2_reL0,inter2_reL1,inter2_reL2,cross_inter2_reL0,cross_inter2_reL1,cross_inter2_reL2


                    
                
        

        



def getting_output(result_min,para_fin,para_name,x_pa,x_pa1,x_pa2,symmetry,name_inter3=None,bonus=None):
    output=defaultdict(defaultdict)
    result_list=result_min.tolist()


                                
    l,para_fin_c=np.shape(para_fin[0]) 
                          
    for i in range(para_fin_c):
        
        output["endmember"][str(para_name[0][i])]=result_list[i]
        if symmetry:
            a=permutations(para_name[0][i],len(para_name[0][i]))
            a=list(a)
            a=list(set(a))

            for j in a:
                j=list(j)

                output["endmember"][str(j)]=result_list[i]
            
    else:  
        if len(x_pa)>0:
            n=0
            for i in x_pa:
                i=int(i)
                
                
#                output["inter2_L0"][str(para_name[1][i])]=result_list[n+para_fin_c]
                if symmetry:
                    
                    for k in range(len(para_name[1][i][0])+1):
                        li_para=list(para_name[1][i][0])
                        c=copy.deepcopy(li_para)
                        c.insert(k,list(para_name[1][i][-1]))
                        
                        for j in range(len(c)):
                            if type(c[j])==list:
                                pass;
                            else:
                                c[j]='['+c[j]+']'
                        
                        output["inter2_L0"][str(c)]=result_list[n+para_fin_c]
                else:
                    output["inter2_L0"][str(para_name[1][i])]=result_list[n+para_fin_c]

                n=n+1

        
     
        if len(x_pa1)>0:
            n=0
            l,para_crossinter2_c=np.shape(para_fin[4])
            for i in x_pa1:
                i=int(i)
            


                if symmetry:
    
                    for k in range(len(para_name[2][i][0])+1):
                        li_para=list(para_name[2][i][0])
                        c=copy.deepcopy(li_para)
                        c.insert(k,list(para_name[2][i][-1]))
                        for j in range(len(c)):
                            if type(c[j])==list:
                                pass;
                            else:
                                c[j]='['+c[j]+']'

                        output["inter2_L1"][str(c)]=result_list[n+para_fin_c+len(x_pa)]
                else:
                    output["inter2_L1"][str(para_name[2][i])]=result_list[n+para_fin_c+len(x_pa)]
                n=n+1
                
            
        if len(x_pa2)>0:
            n=0
            for i in x_pa2:
                i=int(i)

                if symmetry:
                   
                    for k in range(len(para_name[3][i][0])+1):
                        li_para=list(para_name[3][i][0])
                        c=copy.deepcopy(li_para)
                        c.insert(k,list(para_name[3][i][-1]))
                        for j in range(len(c)):
                            if type(c[j])==list:
                                pass;
                            else:
                                c[j]='['+c[j]+']'

                        output["inter2_L2"][str(c)]=result_list[n+para_fin_c+len(x_pa)+len(x_pa1)]
                else:
                    output["inter2_L2"][str(para_name[3][i])]=result_list[n+para_fin_c+len(x_pa)+len(x_pa1)]
                n=n+1
                
        if len(x_pa2)>0:
            l,para_crossinter2_c=np.shape(para_fin[4])
            for i in range(para_crossinter2_c):
#                output["cross_inter_L0"][str(para_name[4][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
#                output["cross_inter_L1"][str(para_name[5][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+i]
#                output["cross_inter_L2"][str(para_name[6][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+para_crossinter2_c+i]
                cross_inter_ele=[]
                if symmetry:

                    if type(para_name[4][i][-1][-1])==tuple:
                        for k in range(len(para_name[4][i][0])+1):
                            li_para=list(para_name[4][i][0])
                            c=copy.deepcopy(li_para)
                            c.insert(k,list(para_name[4][i][-1][0]))
                            for l in range(len(para_name[4][i][0])+2):
                                d=copy.deepcopy(c)
                                d.insert(l,list(para_name[4][i][-1][1]))
                                if d not in cross_inter_ele:
                                    cross_inter_ele.append(d)
                                else:
                                    continue;
                      
                        for a in cross_inter_ele:
                            for j in range(len(a)):
                                if type(a[j])==list:
                                    pass;
                                else:
                                    a[j]='['+a[j]+']'
                            output["cross_inter_L0"][str(a)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
                            output["cross_inter_L1"][str(a)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+i]
                            output["cross_inter_L2"][str(a)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+para_crossinter2_c+i]
                    else:
                        para_namecross1=[0,0]
                        para_name[4][i][1]=para_namecross1[0]=list(para_name[4][i][1])
                        para_name[4][i][0]=para_namecross1[1]=list(para_name[4][i][0])
                        para_namecross2=[0,0]
                        para_name[5][i][1]=para_namecross2[0]=list(para_name[5][i][1])
                        para_name[5][i][0]=para_namecross2[1]=list(para_name[5][i][0])
                        para_namecross3=[0,0]
                        para_name[6][i][1]=para_namecross3[0]=list(para_name[6][i][1])
                        para_name[6][i][0]=para_namecross3[1]=list(para_name[6][i][0])
                        output["cross_inter_L0"][str(para_name[4][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
                        output["cross_inter_L1"][str(para_name[5][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+i]
                        output["cross_inter_L2"][str(para_name[6][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+para_crossinter2_c+i]
                        output["cross_inter_L0"][str(para_namecross1)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
                        output["cross_inter_L1"][str(para_namecross2)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+i]
                        output["cross_inter_L2"][str(para_namecross3)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+para_crossinter2_c+i]        
                else:
                    output["cross_inter_L0"][str(para_name[4][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
                    output["cross_inter_L1"][str(para_name[5][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+i]
                    output["cross_inter_L2"][str(para_name[6][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_crossinter2_c+para_crossinter2_c+i]
    if name_inter3:
        l,para_inter3_c=np.shape(para_fin[7])
        for i in range(para_inter3_c):
            
#            output["inter3_L0"][str(para_name[7][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
            if symmetry:
                    
                for k in range(len(para_name[7][i][0])+1):
                    li_para=list(para_name[7][i][0])
                    c=copy.deepcopy(li_para)
                    c.insert(k,list(para_name[7][i][-1]))
                    for j in range(len(c)):
                        if type(c[j])==list:
                            pass;
                        else:
                            c[j]='['+c[j]+']'
                    output["inter3_L0"][str(c)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
            else:
                output["inter3_L0"][str(para_name[7][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+i]
        for i in range(para_inter3_c):
#            output["inter3_L1"][str(para_name[8][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+i]
            if symmetry:
                for k in range(len(para_name[8][i][0])+1):
                    li_para=list(para_name[8][i][0])
                    c=copy.deepcopy(li_para)
                    c.insert(k,list(para_name[8][i][-1]))
                    for j in range(len(c)):
                        if type(c[j])==list:
                            pass;
                        else:
                            c[j]='['+c[j]+']'
                    output["inter3_L1"][str(c)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+i]
            else:
                output["inter3_L1"][str(para_name[8][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+i]
        for i in range(para_inter3_c):
#            output["inter3_L2"][str(para_name[9][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+para_inter3_c+i]
            if symmetry:
                for k in range(len(para_name[9][i][0])+1):
                    li_para=list(para_name[9][i][0])
                    c=copy.deepcopy(li_para)
                    c.insert(k,list(para_name[9][i][-1]))
                    for j in range(len(c)):
                        if type(c[j])==list:
                            pass;
                        else:
                            c[j]='['+c[j]+']'
                    output["inter3_L2"][str(c)]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+para_inter3_c+i]
            else:
                output["inter3_L2"][str(para_name[9][i])]=result_list[para_fin_c+len(x_pa)+len(x_pa1)+len(x_pa2)+para_inter3_c+para_inter3_c+i]
    return output


def fitting(site,mole,results,key,sublattice,symmetry,cal_type):
    B=results
    para_site={}
    para_mole={}
    para_fin={}
    para_name={}
    
    for i in range(len(site)):
        n=site[i]
        para_mat_site=[]
        para_name_pre=[]
        for keys,values in n.items():
         
            
            if len(para_mat_site)==0:
                para_mat_site=np.mat(values).transpose()
                para_name_pre.append(list(keys))
            
            else:
                para_mat1=np.mat(values).transpose()

                para_mat_site=np.hstack((para_mat_site,para_mat1))
                para_name_pre.append(list(keys))
                
        para_site[i]=para_mat_site
        para_name[i]=para_name_pre


        

    for i in range(len(mole)):
        n=mole[i]
        para_mat_mole=[]
        for keys,values in n.items():
            
            if len(para_mat_mole)==0:
                 para_mat_mole=np.mat(values).transpose()
            
            else:
                para_mat1=np.mat(values).transpose()
                para_mat_mole=np.hstack((para_mat_mole,para_mat1))
        para_mole[i]=para_mat_mole

    for i in para_site.keys():

        para_fin[i]=para_site[i]-para_mole[i]

    z,len_L=np.shape(para_fin[1])

    a0=list(np.linspace(0,len_L-1,len_L))


    select=[]

    result={}
    length,R=np.shape(B)
    m1=0
    if length>100:
        a=range(len_L,-1,-1)
    else:
        a=range(len_L+1)
    b=range(len_L+1)
    
    for n2 in b:


        x=combinations(a0,n2)
        x_re2=list(x)

        for m in range(len(x_re2)):
            x_pa2=x_re2[m]
            x_pa2=list(x_pa2)
            
            #apa_tot=0
            if n2==0:
                apa_totl2=[];
            else:
                for o in range(len(x_pa2)):
                    a1=x_pa2[o]
                    a1=int(a1)
                    L2=para_fin[3]
                    apa=L2[:,a1]
                        
                    if o == 0:
                        apa_tot=apa
                    else:
                        apa_tot=np.hstack((apa_tot,apa))
                 
                apa_totl2=apa_tot

            for n1 in a:
                

                

                x=combinations(a0,n1)
                x_re1=list(x)
  
           

                for m in range(len(x_re1)):
                    
                    x_pa1=x_re1[m]
                    x_pa1=list(x_pa1)
                    
                    
                    #apa_tot=0
                    if n1==0:
                        apa_totl1=[];
                    else:
                        for o in range(len(x_pa1)):
                            a1=x_pa1[o]
                            a1=int(a1)
                            L1=para_fin[2]
                            
                            apa=L1[:,a1]
                
                            if o == 0:
                                apa_tot=apa
                            else: 
                                apa_tot=np.hstack((apa_tot,apa))
                        apa_totl1=apa_tot
                    


                    
                            
                             

  
                    for n in a:
                        

                        x=combinations(a0,n)
                        
                        x_re=list(x)
                        
                        if n==0:
                            continue;
    

                        for m in range(len(x_re)):
                            x_pa=x_re[m]
                            x_pa=list(x_pa)
                            
                            


                            apa_tot=0
                            for o in range(len(x_pa)):
                                a1=x_pa[o]
                                a1=int(a1)
                                L0=para_fin[1]
                                
                                apa=L0[:,a1]
                               
                                if o == 0:
                                    apa_tot=apa
                                else:
                              
                                    apa_tot=np.hstack((apa_tot,apa))
                            ele=para_fin[0]
                  
                            apa_totl0=np.hstack((ele,apa_tot))
                            
                            if len(apa_totl2)==0:
                                if len(apa_totl1)==0:
                                    apa_all=apa_totl0
                                else:
                                    apa_all=np.hstack((apa_totl0,apa_totl1))
                            elif len(apa_totl1)==0 and len(apa_totl2)>0:
                                apa_all=np.hstack((apa_totl0,apa_totl2))
                            else:
                                apa_all=np.hstack((apa_totl0,apa_totl1,apa_totl2))
                            name_inter3=False

                                    
                            if n1>-1:
                                if 7 in para_fin.keys():
                                    apa_all=np.hstack((apa_all,para_fin[7],para_fin[8],para_fin[9]))
                                    name_inter3=True
                            if n2>0:
                                
                                apa_all=np.hstack((apa_all,para_fin[4],para_fin[5],para_fin[6]))
                            
                            
                            X1=np.linalg.pinv(apa_all)

                            X=np.matmul(X1,B)
                    
                            result[m1]=X
                            
                            C=np.matmul(apa_all,X)

                            check_point=0
                            H_1=B.tolist()
                            H=copy.deepcopy(H_1)
                            F=C.tolist()
                            for i in range(len(H)):
                                H[i][0]=float("{:.5g}".format(H[i][0]))
                            for i in range(len(H)):
                                number=H[i][0]
                                
                                G=F[i][0]
                                number_dec=number-int(number)
                                number_dec=abs(float("{:.5f}".format(number_dec)))
                                
                                digit=len(str(number_dec).split('.')[1])
                                format_para="{:."+str(digit)+"f}"
                                F[i]=float(format_para.format(G))
                                if digit>check_point:
                                    check_point=digit
                            
                            
                            check_point=check_point-1
                            F=np.mat(F).transpose()
                            B=np.mat(H)

                            C=F
                            
                            D=B-C
                            D=np.abs(D)
                            
                            factor=np.mean(D)
                            select.append(factor)

#                            if n==8 and n1==8:
#                                test=np.array([[-43582.437],[-93413.353],[-65861.2],[3819.8849],[1068.6684],[0],[-32899.859],[0],[0]])
#                                E=np.dot(apa_all,test)
                            
                            
                                
                            factor1=math.floor(factor*10**(check_point-1))/(10**(check_point-1))
                            m1=m1+1

                            if factor1 ==0:
                                min_select=select.index(factor)



                                
                                
                                output=getting_output(result[min_select],para_fin,para_name,x_pa,x_pa1,x_pa2,symmetry,name_inter3)

                                file_name='./order_disorder_result_'+str(cal_type)+'_'+str(key)+'.json'
                                
                                with open(file_name,'w') as f:
                                    json.dump(output,f,indent=4)
                                return()
    



                              
def change_tdb(order_phase,disorder_phase,number,pre_fitting_phase):
    name=os.listdir()
    input_name=''
    name_num=0
    for i in name:
        if '.tdb' in i and pre_fitting_phase in i:
            input_name=i;
            break;
        elif '.tdb' in i or '.TDB' in i:
            input_name=i

    with open(input_name,'r') as r:
        lines=r.readlines()
        fix_lines=copy.deepcopy(lines)
    used_sign=[]
    length=0
    for l in range(len(lines)):
        if 'PHASE' in lines[l] and '%' in lines[l]:
            m=lines[l].split("%")
    
            re=m[1].split(' ',1)
            used_sign.append(re[0])
            
        if 'FUNCTION' in lines[l] and 'N !' not in lines[l]:
            n=1
    
            while 'N !' not in lines[l+n]:
                lines[l]=str(lines[l])+str(lines[l+n])
                lines[l+n]=' '
                n=n+1
                    
            lines[l]=str(lines[l])+str(lines[l+n])
            lines[l+n]=' '
            if n>length:
                length=n
    for l in range(len(lines)):
    
        if 'FUNCTION' in lines[l] and 'N !' in lines[l]:
            n=0
            for x in range(1,length+2):
                    
                if 'FUNCTION' in lines[l+x]:
                    n=n+1
    
            if n==0:
                end_mark=l
    
    with open('order_disorder_result_SM_MIX_'+order_phase+'.json','r') as re1:
        result_1=json.loads(re1.read())
        
    with open('order_disorder_result_HM_MIX_'+order_phase+'.json','r') as re:         
        result=json.loads(re.read())
        result_co=copy.deepcopy(result)
        for i,j in result_co.items():
            for k,l in j.items():
                if l[0]<10**(-4):
                    del result[i][k];
        result1_co=copy.deepcopy(result_1)
        for i,j in result1_co.items():
            for k,l in j.items():
                if l[0]<10**(-4):
                    del result_1[i][k];            
    
    
        n=number
        insert_line=1
        w_l_to=[]
        for key,value in result.items():
    
            for ele,num in value.items():

                if np.abs(num[0])<0.0001:
                    num[0]=0

                or_ele=copy.deepcopy(ele)
                w_l='PARAMETER G('+order_phase+','
    
    
                ele=ele.replace('\'','')
                if key =='endmember':
                    ele=ele.replace(',',':')
                else:
                    ele=ele.replace(', [',':')
                    ele=ele.replace(', (',':')
                    ele=ele.replace('],',':')
                    ele=ele.replace('),',':')
                ele=ele.replace('[','')
                ele=ele.replace('(','')
                ele=ele.replace(']','')
                ele=ele.replace(')','')
                ele=ele.replace(' ','')
                w_l=w_l+ele+':VA;'
                f_l='FUNCTION '
                if key=='inter2_L0' or key=='inter3_L0' or key=='cross_inter_L0' or key =='endmember':
                    w_l=w_l+'0'
                elif key=='inter2_L1' or key=='inter3_L1' or key=='cross_inter_L1':
    
                    w_l=w_l+'1'
                elif key=='inter2_L2' or key=='inter3_L2' or key=='cross_inter_L2':
                    w_l=w_l+'2'
                w_l=w_l+') 1 '
                if n<10:
                    num_mark='RR000'
                elif n>99 and n<1000:
                    num_mark='RR0'
                elif n>9 and n<100:
                    num_mark='RR00'
                elif n>999:
                    num_mark='RR'
    
                if key in result_1.keys() and or_ele in result_1[key].keys():
                    if np.abs(result_1[key][or_ele][0])<0.0001:
                        result_1[key][or_ele][0]=0
                    n_1=n+1
                    if n_1<10:
                        num_mark1='RR000'
                    elif n_1>99 and n_1<1000:
                        num_mark1='RR0'
                    elif n_1>9 and n_1<100:
                        num_mark1='RR00'
                    elif n_1>999:
                        num_mark1='RR'
                    w_l=w_l+num_mark+str(n)+'# '+'+ T*'+num_mark1+str(n_1)+'#'+'; 10000 N !\n'
                    f_l=f_l+num_mark+str(n)+' 1 '+str(num[0])+'; 10000 N !\n'+f_l+num_mark1+str(n_1)+' 1 '+str(result_1[key][or_ele][0])+'; 10000 N !\n'
                    n=n+1
                else:
                    w_l=w_l+num_mark+str(n)+'#'+'; 10000 N !\n'
                    f_l=f_l+num_mark+str(n)+' 1 '+str(num[0])+'; 10000 N !\n'
                
                n=n+1
                w_l_to.append(w_l)

    
    
                fix_lines.insert(end_mark+insert_line,f_l)
                insert_line=insert_line+1
    
    
    
        for key,value in result_1.items():
    
    
            for ele,num in value.items():
                if np.abs(num[0])<0.0001:
                    num[0]=0
                or_ele=copy.deepcopy(ele)
                w_l='PARAMETER G('+order_phase+','
    
    
                ele=ele.replace('\'','')
                if key =='endmember':
                    ele=ele.replace(',',':')
                else:
                    ele=ele.replace(', [',':')
                    ele=ele.replace(', (',':')
                    ele=ele.replace('],',':')
                    ele=ele.replace('),',':')
                ele=ele.replace('[','')
                ele=ele.replace('(','')
                ele=ele.replace(']','')
                ele=ele.replace(')','')
                ele=ele.replace(' ','')
                w_l=w_l+ele+':VA;'
                f_l='FUNCTION '
                if key=='inter2_L0' or key=='inter3_L0' or key=='cross_inter_L0' or key =='endmember':
                    w_l=w_l+'0'
                elif key=='inter2_L1' or key=='inter3_L1' or key=='cross_inter_L1':
    
                    w_l=w_l+'1'
                elif key=='inter2_L2' or key=='inter3_L2' or key=='cross_inter_L2':
                    w_l=w_l+'2'
                w_l=w_l+') 1 '
                if n<10:
                    num_mark='RR000'
                elif n>99 and n<1000:
                    num_mark='RR0'
                elif n>9 and n<100:
                    num_mark='RR00'
                elif n>999:
                    num_mark='RR'
             
                if key in result.keys() and or_ele in result[key].keys():
                    continue;
                else:
                    w_l=w_l+'T*'+num_mark+str(n)+'#'+'; 10000 N !\n'
                    f_l=f_l+num_mark+str(n)+' 1 '+str(num[0])+'; 10000 N !\n'
                
                n=n+1
                w_l_to.append(w_l)
    
    
                fix_lines.insert(end_mark+insert_line,f_l)
                insert_line=insert_line+1            
                    
                    
    
    
        not_used=[]
        for i in string.punctuation:
            if i =='%' or i=='!' or i in used_sign:
                continue;
            not_used.append(i)
        type_lines='\n\nTYPE_DEFINITION '+not_used[0]+' GES AMEND_PHASE_DESCRIPTION '+order_phase+' DIS_PART '+disorder_phase+',,,!'
        fix_lines.insert(end_mark+insert_line,type_lines)
     
                
    with open(order_phase+'_changed.tdb','w') as w:
        for l in range(len(fix_lines)):
            if 'PHASE' in fix_lines[l] and order_phase in fix_lines[l]:
                replace_item='%'+not_used[0]
                fix_lines[l]=fix_lines[l].replace('%',replace_item)
            if '('+order_phase+',' not in fix_lines[l]:
                w.write(fix_lines[l])
    
    
                
    
    
    with open(order_phase+'_changed.tdb','a') as a:
        a.write('\n'*3)
        a.write('$'*78)
        a.write('\n')
        a.write('$')
        a.write(' '*31)
        a.write('order-disorder')
        a.write(' '*31)
        a.write('$\n')
        a.write('$'*78)
        a.write('\n')
        a.write('\n'*3)
        for i in w_l_to:
            a.write(i)
    for i in name:
        if '_changed.tdb' in i and pre_fitting_phase in i and len(pre_fitting_phase)>0:
            os.remove(input_name)
            break
    return n,order_phase
    
def produce_json(ordered_phase,data,type_M): 
    name_path=[]
    key_words=type_M+'_MIX'
    sub_model=data['phases'][ordered_phase]['sublattice_model']
    sub_model1=copy.deepcopy(data['phases'][ordered_phase]['sublattice_model'])


    for root, dirs, files in os.walk('input-data'):
        for i in files:
            if key_words in i and ordered_phase in i:
                name_path.append(os.path.join(root, i));
    site_data=[]
    re_data=[]
    for e in name_path:
        with open(e,'r') as f:
            data2=json.load(f)
            cal_type=data2["output"]
            site_data2=data2['solver']['sublattice_occupancies']
            for i in range(len(site_data2)):
                for k in range(len(site_data2[i])):
                    if type(site_data2[i][k]) != list:
                        site_data2[i][k]=[site_data2[i][k]] 
                    if len(site_data2[i][k])==len(sub_model1[k]):

                        continue;
                    else:
                        if "sublattice_configurations" in data2['solver'].keys():
                            model_change=data2['solver']["sublattice_configurations"]
                            for j in range(len(model_change)):
                                for h in range(len(model_change[j])):
                                    if type(model_change[j][h])!= list:
                                        model_change[j][h]=[model_change[j][h]]
                            for n in range(len(model_change[i][k])):
                                num=sub_model1[k].index(model_change[i][k][n])
                                for o in range(len(sub_model1[k])):
                                    if o==num:
                                        pass;
                                    else:
                                        site_data2[i][k].insert(o,0)
                        else:
                            raise ValueError('The sublattice_occupancies are not consistent with the sublattice model')
        if len(site_data)==0:
            site_data=site_data2
            re_data=data2["values"]
            re_data=re_data[0]
        else:
            site_data=np.vstack((site_data,site_data2))
            re_data1=data2["values"]
            re_data1=re_data[0]
            re_data=np.hstack((re_data,re_data1))

    ratios=data['phases'][ordered_phase]['sublattice_site_ratios']
    symmetry=decide_symmetry(sub_model,ratios)
    mole_data=find_mole_data(site_data,ratios,symmetry)

    for i in range(len(re_data[0])):
        a=re_data[0][i]
        re_data[0][i]=float("{:.8g}".format(a))
    try_fit=np.mat(re_data).transpose()
    if 'SM' in cal_type:

        try_fit=SM_modify(try_fit,site_data,mole_data,ratios)
    end_member=find_endmember(sub_model,symmetry)
    inter2=find_inter2(sub_model)
    cross_inter=find_crossinter2(sub_model)
    if np.sum(try_fit)==0:
        para_name={}
        para_name[0]=end_member
        zero=[0]*len(end_member)
        output=getting_output(zero,zero,para_name,[0],[0],[0],symmetry,False)
        return()
    sublattice_model=copy.deepcopy(sub_model)
    if sublattice_model.count(['VA'])==1:
        sublattice_model.remove(['VA'])

    if len(sublattice_model[0])>2:
        inter3=find_inter3(sub_model)
        site_pa=paramemter(site_data,end_member,sublattice_model,inter2,cross_inter,symmetry,inter3)
        mole_pa=paramemter(mole_data,end_member,sublattice_model,inter2,cross_inter,symmetry,inter3)
    else:
        site_pa=paramemter(site_data,end_member,sublattice_model,inter2,cross_inter,symmetry)
        mole_pa=paramemter(mole_data,end_member,sublattice_model,inter2,cross_inter,symmetry)

    fitting(site_pa,mole_pa,try_fit,ordered_phase,sublattice_model,symmetry,cal_type)
  
    
    

def fit():
    for i in os.listdir():
        if '-input.json' in i or 'INPUT+MODEL.json' in i:
            model_name=i

    with open(model_name,'r') as f:

        data=json.load(f)
        num_para=1
        pre_phase=''
        # path='ESPEIrun'
        name=os.listdir()
        for i in name:
            if '_changed.tdb' in i:
                print(i)
                os.remove(i)
        for i in data['phases'].keys():
            if 'model_hints' in data['phases'][i]:
                if 'ordered_phase' in data['phases'][i]['model_hints']:
                    if data['phases'][i]['model_hints']['ordered_phase']== i:
                        ordered_phase=data['phases'][i]['model_hints']['ordered_phase']
                        disordered_phase=data['phases'][i]['model_hints']['disordered_phase']
                        produce_json(ordered_phase,data,'HM')
                        produce_json(ordered_phase,data,'SM')
                        num_para,pre_phase=change_tdb(ordered_phase,disordered_phase,num_para,pre_phase)
    name1=os.listdir()
    for i in name1:
        if 'order_disorder_result_' in i and '.json' in i:
            print(i)
            os.remove(i)

        
            



        
        
        
        

                
            
                
            