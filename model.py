import argparse
import transformers
from transformers import AutoModel, AutoTokenizer 
import numpy as np
import torch
import logging
from pathlib import Path
from os.path import exists
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import csv, json
import evaluate
from datasets import Dataset
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
from sklearn.metrics import auc, roc_curve

from torch import tensor 
from transformers.pipelines import TextClassificationPipeline
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import matplotlib.pyplot as plt 
import jsonlines

labelToModelLogitIndex = {
    "Negative": 0,
    "Positive": 1, 
}

colsToRemove = {
    "imdb": [
        "text"
    ]
}

labelTag = {
    "imdb": "label"
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-cacheDir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1419542/huggingface_cache/"
)

parser.add_argument(
    "-dataset",
    choices = [
        "imdb",
    ],
    default="imdb",
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=1
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=16
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for optimizer",
    default=2e-5
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0.01
)

parser.add_argument(
    "-model",
    help="Path to model to use",
    default="microsoft/deberta-v3-large"
)

parser.add_argument(
    "-out",
    "--output_dir",
    help="Path to output directory where trained model is to be saved",
    required=True
)

parser.add_argument(
    '-seed', 
    type=int, 
    help='Random seed', 
    default=13
)

parser.add_argument(
    "-do_train",
    action="store_true",
    help="Boolean flag to train model"
)

parser.add_argument(
    "-do_predict",
    action="store_true",
    help="Boolean flag to make predictions"
)

parser.add_argument(
    "-cpu",
    "--use_cpu",
    action="store_true",
    help="Boolean flag to use cpu only"
)
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")
#---------------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#---------------------------------------------------------------------------
class ComputeMetrics:
        def __init__(self, metricName="accuracy"):
            self.metricName = metricName
            self.metric = evaluate.load(metricName)
        
        def __call__(self, evalPreds):
            predictions, labels = evalPreds
            predictions = np.argmax(predictions, axis=1)
            return self.metric.compute(predictions=predictions, references=labels)
#---------------------------------------------------------------------------
class Tokenize:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer 
        self.dataset = dataset

    def __call__(self, example):
        # return self.tokenizer(inputToPrompt(example, self.dataset), truncation=True)
        return self.tokenizer(example["text"], truncation=True)
#---------------------------------------------------------------------------
def inputToPrompt(instance, dataset):
    if dataset == "imdb":
        inpPrompt = "Review: {review}\nWhat is the sentiment of the review: negative or positive?".format(
            review=instance["text"]
        )
    else: 
        raise ValueError("[inputToPrompt] {} not supported!".format(dataset))
    return inpPrompt
#---------------------------------------------------------------------------
def writeFile(data, fileName):
    if fileName.endswith(".csv"):
        with open(fileName, 'w', newline='') as f:
            writer = csv.DictWriter(f, data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    elif fileName.endswith(".json"):
        with open(fileName, "w") as f: 
            json.dump(data, f)
    elif fileName.endswith(".jsonl"):
        with open(fileName, "w") as f: 
            for instance in data:
                f.write(json.dumps(instance))
                f.write("\n")
    else: 
        raise ValueError("[readFile] {} has unrecognized file extension!".format(fileName))
#---------------------------------------------------------------------------
def collateBatch(batch):
    return zip(*batch)
#---------------------------------------------------------------------------
def createDataLoader(ds, batchSize, collateFn=collateBatch):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        collate_fn=collateFn,
    )
# ---------------------------------------------------------------------------
class DeBertaWrapper(torch.nn.Module):
    def __init__(self, model, device="cpu"):
        super(DeBertaWrapper, self).__init__()
        self.model = model
        self.device = device
        self.model.to(device)
    
    def __call__(self, *inputs):
        inputs = torch.tensor(inputs, device=self.device).squeeze()
        return torch.tensor(self.model(inputs)["logits"])
        # return self.model(*inputs)
    
    def children(self):
        return self.model.children()
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        # logging.basicConfig(filemode='w', level=logging.ERROR)
        logging.basicConfig(filemode='w', level=logging.INFO)

    if torch.cuda.is_available() and not args.use_cpu:
        logging.info("Using GPU: cuda")
        device = "cuda"
    else: 
        logging.info("Using CPU")
        device = "cpu"

    if args.batchSize <= 0:
        raise ValueError("[main] Batch Size has to be a positive number!")
    
    data = load_dataset(args.dataset, cache_dir=args.cacheDir)
    data = data.shuffle(seed=args.seed)
    if "train" not in data.keys():
        raise RuntimeError("[main] No train split found in {} dataset!".format(args.dataset))
    if "test" not in data.keys():
        raise RuntimeError("[main] No test split found in {} dataset!".format(args.dataset))
    
    data["train"] = data["train"].select(np.random.choice(len(data["train"]), 10))
    data["test"] = data["test"].select(np.random.choice(len(data["test"]), 2))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labelToModelLogitIndex))
    model.to(device)

    tokenizedDatasets = data.map(Tokenize(tokenizer, args.dataset), batched=True, remove_columns=colsToRemove[args.dataset])
    tokenizedDatasets = tokenizedDatasets.rename_column(labelTag[args.dataset], "labels")
    dataCollator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=1024)

    if args.do_train or args.do_predict:
        trainingArgs = TrainingArguments(
            output_dir=args.output_dir, 
            num_train_epochs=args.numEpochs,
            learning_rate=args.learningRate,
            weight_decay=args.weightDecay,
            per_device_train_batch_size=args.batchSize,
            per_device_eval_batch_size=args.batchSize,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=50,
            eval_steps=50,
            save_total_limit=100,
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
            bf16=True,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True
        )

        trainer = Trainer(
            model,
            trainingArgs,
            train_dataset=tokenizedDatasets["train"],
            eval_dataset=tokenizedDatasets["test"],
            data_collator=dataCollator,
            tokenizer=tokenizer,
            compute_metrics=ComputeMetrics("accuracy")
        )

    if args.do_train:
        #Train the model
        trainer.train()

    if args.do_predict:
        #Sample 10 mispredictions randomly
        predictions = trainer.predict(tokenizedDatasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        incorrectInds = np.where(~np.equal(preds, tokenizedDatasets["test"]["labels"]))[0]
        assert len(incorrectInds) >= 10
        testData = data["test"]
        testData = testData.add_column("predicted", preds)
        if args.dataset == "imdb":
            testData = testData.rename_column("text", "review")
        allData = Dataset.from_dict(testData[incorrectInds])
        sampledData = Dataset.from_dict(testData[np.random.choice(incorrectInds, 10, replace=False)])

        allData.to_json("mispredictions.jsonl", orient="records", lines=True)
        sampledData.to_json("mispredictions_10.jsonl", orient="records", lines=True)
    
    #Finding most influential training examples for test examples

    # clf = transformers.pipeline("text-classification", 
    #     model=model, 
    #     tokenizer=tokenizer, 
    #     device=device
    # )
    # modelCheckpoints = list(os.walk(args.output_dir))[0][1]
    # extrChkpt = lambda path: int(path.split("-")[-1])
    # sorted(modelCheckpoints, key=extrChkpt)
    # appendOutputDirPath = lambda path: args.output_dir + "/" + path
    # modelCheckpoints = list(map(appendOutputDirPath, modelCheckpoints))
    # model = ExplainableTransformerPipeline(modelCheckpoints[-1], clf, device)
    # checkpoints_load_func = lambda _, path: ExplainableTransformerPipeline(path, clf, device)

    checkpoints_load_func = lambda _, path: DeBertaWrapper(AutoModelForSequenceClassification.from_pretrained(path, num_labels=len(labelToModelLogitIndex)), device)
    model = DeBertaWrapper(model, device)

    # #Generate train data in the format TracInCPFast expects
    # trainDataLoader = createDataLoader(tokenizedDatasets["train"], args.batchSize, dataCollator)

    # #Generate test data in the format TracInCPFast expects
    # testDataLoader = createDataLoader(tokenizedDatasets["test"], args.batchSize, dataCollator)

    tokenizedDatasets["train"] = tokenizedDatasets["train"].map(dataCollator)
    tokenizedDatasets["test"] = tokenizedDatasets["test"].map(dataCollator)

    tracin_cp_fast = TracInCPFast(
        model=model,
        final_fc_layer=list(model.children())[-1],
        train_dataset=(
            tokenizedDatasets["train"]["input_ids"],
            torch.tensor(tokenizedDatasets["train"]["labels"], device=device),
        ),
        # train_dataset=tokenizedDatasets["train"],
        # train_dataset=trainDataLoader,
        # checkpoints=modelCheckpoints,
        checkpoints=args.output_dir,
        checkpoints_load_func=checkpoints_load_func,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        batch_size=1,
        vectorize=False,
    )

    k = 10
    proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
        # testDataLoader, 
        (
            tokenizedDatasets["test"]["input_ids"],
            torch.tensor(tokenizedDatasets["test"]["labels"], device=device),
        ),
        k=k, 
        proponents=True,
        show_progress=True,
    )
    opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
        # testDataLoader, 
        (
            tokenizedDatasets["test"]["input_ids"],
            torch.tensor(tokenizedDatasets["test"]["labels"], device=device),
        ),
        k=k, 
        proponents=False,
        show_progress=True,
    )

    print(proponents_indices)
    print(opponents_indices)

#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()