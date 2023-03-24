# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:46:58 2022

@author: mhdkhatamirad
"""

# =============================================================================
# import required libraries and modules
# =============================================================================

import shutil
import json
import pandas as pd
import numpy as np
from scipy.stats import norm, entropy
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import KFold

# =============================================================================
# define function for writing input
# =============================================================================

def write_input(path, df, id_job, n_cutoffs, algo, dev, n_res, n_seeds, target_key):
    """
    creates the two input files necessary to run the algorithm:
    i) a .json file with calculation details, named "id_job.json", and
    ii) a .xarf file with the data set, named "id_job.xarf".
    function arguments: path(str): path to the folder where the files 
                                   will be written
               df(data frame): data set containing the values for the 
                               candidate descriptive parameters and for
                               the target for all adsorption sites
               id_job(str): job name
               n_cutoffs(int): number of cutoffs to be used in k-Means
                               clustering to generate the propositions
               algo(str): SG search algorithm (PMM_SAMPLER or EMM_SAMPLER)
                          PMM_SAMPLER uses (std(SG)-std(P))/std(P) as utility function
                          whereas EMM_SAMPLER uses the function specified in dev
               dev(str): deviation measure when using EMM_SAMPLER 
                         (e.g. cumulative_jensen_shannon_divergence)
               n_res(int): number of results, i.e., number of top-ranked
                           SGs to display
               n_seeds(int): number of seeds to use for the SG search
               target_key(str): label of the variable to be used as target quantity in SGD
    """
    df.to_csv(path+'/'+id_job+'.csv')
    with open(path+'/'+id_job+'.csv', 'r') as file_in:
        data = file_in.read().splitlines(True)
        
    file_out = open(path+'/'+id_job+'.xarf', 'w')
    file_out.write('@relation '+id_job+'\n')
    file_out.write('@attribute sites name\n')
    for variable in list(df.columns):
        file_out.write('@attribute '+variable+' numeric\n')
    file_out.write("@data\n")
    file_out.close()

    with open(path+'/'+id_job+'.xarf', 'a') as file_out:
        file_out.writelines(data[1:])
        file_out.close()
    
    input_file = {}
    input_file = {"type" : "productWorkScheme",
                  "id" : id_job,
                  "workspaces" : [ {
                                "type" : "workspaceFromXarf",
                                "id" : id_job,
                                "datafile" : id_job+".xarf",
                                "propScheme": {"type": "standardPropScheme",
                                                "defaultMetricRule": {"type": "kmeansPropRule",
                                                                       "numberOfCutoffs": n_cutoffs,
                                                                       "maxNumberOfIterations": 1000}}} ],
                    "computations" : [ {
                                "type" : "legacyComputation",
                                "id" : "subgroup_analysis",
                                "algorithm" : algo,
                                "parameters" : {
                                    "dev_measure": dev,
                                    "attr_filter" : "[]",
                                    "cov_weight" : "1.0",
                                    "num_res" : n_res,
                                    "num_seeds" : n_seeds,
                                    "targets" : "["+target_key+"]"
                                             }
                  }],
                  "computationTimeLimit" : 360000
                     }
    with open(path+'/'+id_job+'.json','w') as outfile:
        json.dump(input_file, outfile, indent=4)
        
        
# =============================================================================
# define function for getting SGD results         
# =============================================================================

def get_sg_info(data,index):
    """
    reads the SGD output .json file and returns the following information 
    on a specific identified SGs: coverage, utility function, target mean value within the SG,
                           the SG rules and the parameters enterning the rulesattributes
    function arguments: data(str): path to the output .json file
                        index(int): index of the SG for which the information is obtained
                                    note that the SGs are ordered by decreasing quality-function values
    """
    coverage=data[index].get('measurements')[0].get('value')
    utility_function=data[index].get('measurements')[1].get('value')
    target_mean=data[index].get('descriptor').get('targetLocalModel').get('means')
    list_attributes=data[index].get('descriptor').get('selector').get('attributes')
    list_operators=[]
    list_cutoffs=[]
    constraints=[]
    for i in list(range(0,len(list_attributes))):
        list_operators.append(data[index].get('descriptor').get('selector').get('constraints')[i].get('type'))
        list_cutoffs.append(round(data[index].get('descriptor').get('selector').get('constraints')[i].get('value'),4))

    list_operators = [op.replace('lessOrEquals', '<=') for op in list_operators]
    list_operators = [op.replace('greaterOrEquals', '>=') for op in list_operators]
    list_operators = [op.replace('lessThan', '<') for op in list_operators]
    list_operators = [op.replace('greaterThan', '>') for op in list_operators]
    
    for i in list(range(0,len(list_attributes))):
        if i == 0:
            constraints=list_attributes[0]+list_operators[0]+str(list_cutoffs[0])
        else:
            constraints=constraints+' & '+list_attributes[i]+list_operators[i]+str(list_cutoffs[i])
    return(coverage,utility_function,*target_mean,constraints,list_attributes, list_cutoffs)

































