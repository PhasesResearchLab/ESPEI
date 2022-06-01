#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:07:45 2020

@author: sunhui
"""
import os


name=[]
for i in os.listdir():
    if "json" in i:
        name.append(i)

for j in name:
    out=[]
    n=0
    with open(j,'r') as c:
        change=c.readlines()
        work=False
        for i in range(len(change)):
            len_i=change[i].replace('\n','')
            len_i=len_i.replace(' ','')
            if 'values' in change[i]:
                work=True;
                len_i='    '+len_i
            if 'reference' in change[i]:
                work=False
        

            if work: 
                out[n-1]=out[n-1]+len_i
            
            else:
                out.append(change[i])
                n=n+1
    name_new='H+'+j
    with open(name_new,'w+') as f:
        for i in out:
            i=i.replace(']]],',']]],\n')
            i=i.replace(':[[[',':[\n[[')
            i=i.replace(']]]],',']]]\n        ],')
            i=i.replace('[[','             [[')
        
            f.write(i)