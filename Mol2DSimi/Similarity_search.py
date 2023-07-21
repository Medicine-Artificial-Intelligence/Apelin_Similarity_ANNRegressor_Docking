import math
import numpy as np
import pandas as pd
from rdkit import Chem
import sys
import os
import glob
import pickle
sys.path.append('Mol2DSimi')
from Similarity import similarity_calculate
from enrichment_factor import Enrichment_Factor
from SimiValid import  similarity_validation

class simi_search:
    """
    Similarity Searching.

    Parameters
    ----------
    query : rdkit.chem.rdchem.mol
        query molecules for similarity searching
    data : pandas.DataFrame
        Data screen with ID, active and SMILES columns.
    active_col : str
        Name of "Active" column (binary).
    ID_col : str
        Name of "ID" column .
    smiles_col : str
        Name of "SMILES" column 
    Returns
    -------
    table: pandas.DataFrame
        Data with ...
    plot: matplot
        ROC plot
        
    """
    def __init__(self,  data, smiles_col,  ID_col, model_path,
                 query_smiles='S(=O)(=O)(Nc1n(-c2c(OC)cccc2OC)c(-c2cc(C)cnc2)nn1)[C@H]([C@H](C)c1ncc(C)cn1)C',query_name = 'AMG 986',
                 active_col='Active', input_smiles=None, id_name = None):
        #self.query = query_smiles
        self.query = Chem.MolFromSmiles(query_smiles)
        self.query.SetProp("_Name",query_name)
    
        self.smiles_col = smiles_col
        self.ID_col = ID_col
        self.active_col = active_col
        self.model_path=model_path
        
        while True:
            try:
                if data != None:
                    self.data = data
                    break
                else:
                    self.data = pd.DataFrame([input_smiles], columns =[self.smiles_col])
                    self.data[self.ID_col] = id_name
                    break
            except:
                if data.empty == False:
                    self.data = data
                    break
                else:
                    self.data = pd.DataFrame([input_smiles], columns =[self.smiles_col])
                    self.data[self.ID_col] = id_name
                    break

   
            
            
        while True:
            try:
                self.data[self.active_col]
                break
            except:
                self.data[self.active_col]=0
                break
        self.raw_data = self.data.copy()
    def preprocess(self):
        simi = similarity_calculate(data = self.data, query= self.query, smile_col=self.smiles_col, active_col=self.active_col)
        simi.fit()
        self.screen_data = simi.data
        self.screen_data.index =  self.data[self.ID_col].values
        self.screen_data.drop([self.ID_col,self.smiles_col, self.active_col, 'ROMol'], axis =1, inplace = True)
        col = self.screen_data.columns
        tanimoto = []
        for i in col:
            if 'tanimoto' in i:
                tanimoto.append(i)
        self.screen_data = self.screen_data[tanimoto]
        
        
    def screen(self):
        with open(self.model_path+ '/model.pkl','rb') as f:
            self.model= pickle.load(f)
        self.pred = self.model.predict(self.screen_data)
        self.pred_df = pd.DataFrame(self.pred,columns =['Pred'])
        self.pred_df.index = self.raw_data[self.ID_col].values
        self.proba = self.model.predict_proba(self.screen_data)[:,1]
        self.proba_df = pd.DataFrame(self.proba,columns =['Proba'])
        self.proba_df.index = self.raw_data[self.ID_col].values
        self.data_filter = pd.concat([self.screen_data, self.pred_df, self.proba_df], axis =1,)
        
    def fit(self):
        self.preprocess()
        self.screen()