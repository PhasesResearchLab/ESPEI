#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:50:24 2020

@author: sunhui
"""
import json
import os
name=[]
for root, dirs, files in os.walk('input-data'):
    for i in files:
        if 'HM_MIX' in i:
            name.append(os.path.join(root, i));
for i in os.listdir():
    if '-input.json' in i:
        model_name=i
with open(model_name,'r') as m:
    model=json.load(m)

with open(name[-1],'r') as f:
    data=json.load(f)
    site_data=data['solver']['sublattice_occupancies']
    sub_model=model['phases'][data['phases'][0]]['sublattice_model']
    for i in range(len(site_data)):
        for k in range(len(site_data[i])):
            if type(site_data[i][k]) != list:
                site_data[i][k]=[site_data[i][k]]        
            if len(site_data[i][k])==len(sub_model[k]):
                continue;
            else:
                if "sublattice_configurations" in data['solver'].keys():
                    model_change=data['solver']["sublattice_configurations"]
                    for j in range(len(model_change)):
                        for h in range(len(model_change[j])):
                            if type(model_change[j][h])!= list:
                                model_change[j][h]=[model_change[j][h]]
                    for n in range(len(model_change[i][k])):
                        num=sub_model[k].index(model_change[i][k][n])
                        for o in range(len(sub_model[k])):
                            if o==num:
                                pass;
                            else:
                                site_data[i][k].insert(o,0)
                else:
                    raise ValueError('The sublattice_occupancies are not consistent with the sublattice model')
                

    print(site_data)
    print(data)


