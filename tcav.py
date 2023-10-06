'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''

import torch
import pandas as pd

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import matplotlib.pyplot as plt

import argparse 
import jsonlines
import os 

import os
import glob
import random

import torch
import torchtext
import spacy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from torch.utils.data import DataLoader, Dataset, IterableDataset

#.... Captum imports..................
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str

from torchtext.vocab import Vocab

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

nlp = spacy.load('en')

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
        self.indToLabel = ["Negative", "Positive"]
        self.labelToInd = {
            "Negative": 0,
            "Positive": 1,
        }
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, 
            inputs: list, 
            attributes: list, 
            outfile_path: str,
        ):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        splits = len(attr.cpu().numpy()[0][::-1])//50
        for i, split in enumerate(range(splits)):
            start, end = split*50, (split+1)*50
            if i == splits-1:
                a = pd.Series(attr.cpu().numpy()[0][::-1][start:], 
                                index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1][start:])
            else: 
                a = pd.Series(attr.cpu().numpy()[0][::-1][start:end], 
                            index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0])[::-1][start:end])
            
            a.plot.barh(figsize=(10,20))
            plt.savefig(outfile_path+"_"+str(split))
                      
    def explain(self, text: str, label: int, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        preds = self.forward_func(inputs)


        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True)
        # Give a path to save
        self.visualize(
            inputs, 
            attributes, 
            outfile_path,
        )
        del attributes
        del delta
        del prediction
        del preds
        del lig
    
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint) 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels, load_in_8bit=True)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"

    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                # device=device
                                )
    exp_model = ExplainableTransformerPipeline(args.model_checkpoint, clf, device)

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            print("Explaining {}...".format(idx+1))
            exp_model.explain(obj["review"], obj["label"], os.path.join(args.output_dir,f'example_{idx}'))
            idx+=1
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analsis_dir', default='out', type=str, help='Directory where attribution figures will be saved')
    parser.add_argument('--model_checkpoint', type=str, default='microsoft/deberta-v3-base', help='model checkpoint')
    parser.add_argument('--a1_analysis_file', type=str, default='out/a1_analysis_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')    
    args = parser.parse_args()
    main(args)