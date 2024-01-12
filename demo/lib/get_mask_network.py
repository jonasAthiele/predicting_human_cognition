# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2024

@author: Jonas A. Thiele
"""

import scipy.stats
import numpy as np
from itertools import combinations


#Compute masks with specific selection of links
def get_mask_network(n_nodes, method, idx_nodes):
    
    
    # n: number of links
    # methods: 'connections', 'nodes', 'theory'
    # idx_nodes: idx of preselected nodes (intelligence theories) 
    
    if method == 'connections':
        
        n_connections = scipy.special.comb(n_nodes, 2)
        
        edges = []
        
        while len(edges) < n_connections:
            
            node1 = np.random.randint(0,100,1)
            node2 = np.random.randint(0,100,1)
            
            
            if node1 != node2:
                
                node_combi = np.array([node1, node2]).ravel()
                
                test = 0
                for n in edges:
                    
                    nT = np.ones(n.size)
                    nT.astype(int)
                    nT[0] = n[1]
                    nT[1] = n[0]
                    
                    if (node_combi == n).all():
                        test = 1
                    
                    if (node_combi == nT).all():
                        test = 1
                    
                if test == 0:
                
                    edges.append(np.array([node1, node2]).ravel())
        
        
        
        mask = np.zeros((100, 100))
        
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1))
        
        
    if method == 'nodes':
        
        nodes = []
        
        while len(nodes) < n_nodes:
            
            node = np.random.randint(0,100,1)
            test = 0
            
            if np.isin(node, np.array(nodes))[0]:
                
                test = 1
            
        
            if test == 0:
            
                nodes.append(node)
          
        
        nodes = np.array(nodes).ravel()        
        edges = np.array(list(combinations(nodes,2)))
        
        
        mask = np.zeros((100, 100))
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1)) 
        
    
    
    if method == 'theory':
        
        edges = np.array(list(combinations(idx_nodes,2)))
        
        mask = np.zeros((100, 100))
        for n in edges:
            
            mask[n[0], n[1]] = 1
            mask[n[1], n[0]] = 1
        
        mask = np.triu(mask,1)
        
        mask_network = []
        mask_network.append(mask)
        network_list = list(np.arange(0,1)) 
            
    
    return mask_network, network_list, edges