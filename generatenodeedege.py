from datetime import datetime
import pandas as pd   
import os
import pickle as pkl
import sys
import pandas as pd
import numpy as np
from vuljavadetectmodel.inits import __inits__important as imp
import vuljavadetectmodel.data_preprocessing as dpre  
import vuljavadetectmodel.graph_subprocess as gsp  

# Extract nodes and edges data for a given function  
# SETUP
NUM_JOBS = 1  
JOB_ARRAY_NUMBER = 0 


def readjavacode(pathjavacode: str):
    # storing the current time in the variable, then use it as function id to avoid ovalaping ids in the dataset
    c = datetime.now()
    id = int(c.strftime('2024%H%M%S'))
    try:
        if os.path.exists(pathjavacode):
            with open(pathjavacode, 'r') as f:
                code = f.read()
        return code , id
    except:
        print("The code to analyse is not yet ready... verify instructions. Thanks")
        return print(f"The running id is: {id}")


def createdf(code_text, id):
    df = pd.DataFrame({"id": [id], "before": [str(code_text)], "after": [str(code_text)], 
                   "dataset": 'datasetss', "diff": [[]], "added": [[]], "removed": [[]],
                   "label": 'test'})
    return df



def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.datasetss()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = imp.get_dir(imp.processed_dir() / row["dataset"] / "before")
    savedir_after = imp.get_dir(imp.processed_dir() / row["dataset"] / "after")

    # Write Java Files
    fpath1 = savedir_before / f"{row['id']}.java"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.java"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        gsp.full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        gsp.full_run_joern(fpath2, verbose=3)



# construct the graph with the extracted node and edges.

import glob
import json
import dgl
import torch
import torch as th
from pathlib import Path
from node2vec import Node2Vec
from dgl import load_graphs, save_graphs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import torchmetrics
import networkx as nx
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from tqdm import tqdm


class datasetssDatasetNLP:
    """Override getitem for codebert."""

    def __init__(self, partition="train", random_labels=False):
        """Init."""
        self.df = dpre.datasetss()
        self.df = self.df[self.df.label == partition]
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0] .sample(len(vul), random_state=0) # brinh this back when you have over 1000 sample functions ---------->>
            self.df = pd.concat([vul, nonvul])
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        tokenized = tokenizer(text, **tk_args)
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class datasetssDatasetNLPLine:
    """Override getitem for codebert."""

    def __init__(self, partition="train"):
        """Init."""
        linedict = dpre.get_dep_add_lines_datasetss()
        df = dpre.datasetss()
        df = df[df.label == partition]
        df = df[df.vul == 1].copy()
        df = df.sample(min(1000, len(df))) # it does not work, bring back the code #---------------->>

        texts = []
        self.labels = []

        for row in df.itertuples():
            line_info = linedict[row.id]
            vuln_lines = set(list(line_info["removed"]) + line_info["depadd"])
            for idx, line in enumerate(row.before.splitlines(), start=1):
                line = line.strip()
                if len(line) < 5:
                    continue
                if line[:2] == "//":
                    continue
                texts.append(line.strip())
                self.labels.append(1 if idx in vuln_lines else 0)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in texts]
        tokenized = tokenizer(text, **tk_args)
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]



class datasetssDatasetNLPDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for datasetss."""

    def __init__(self, DataClass, batch_size: int = 32, sample: int = -1):
        """Init class from datasetss dataset."""
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test, batch_size=self.batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LitCodebert(pl.LightningModule):
    """Codebert."""

    def __init__(self, lr: float = 1e-3):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 2)
        self.accuracy = torchmetrics.Accuracy(task = "binary", num_classes = 2 ) 
        self.auroc = torchmetrics.AUROC(task = "binary", num_classes = 2) 
        from torchmetrics import MatthewsCorrCoef
        self.mcc = MatthewsCorrCoef(task = "binary", num_classes = 2)

    def forward(self, ids, mask):
        """Forward pass."""
        with torch.no_grad():
            bert_out = self.bert(ids, attention_mask=mask)
        fc1_out = self.fc1(bert_out["pooler_output"])
        fc2_out = self.fc2(fc1_out)
        return fc2_out
    
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        labels = labels.type(torch.LongTensor) 
        labels , logits = labels.to(device), logits.to(device)
        loss = F.cross_entropy(logits, labels)

        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        labels = labels.type(torch.LongTensor) 
        labels , logits = labels.to(device), logits.to(device)
        loss = F.cross_entropy(logits, labels)
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logits[:, 1], labels)
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        ids, att_mask, labels = batch
        logits = self(ids, att_mask)
        labels = labels.type(torch.LongTensor) 
        labels , logits = labels.to(device), logits.to(device)
        loss = F.cross_entropy(logits, labels)
        self.auroc.update(logits[:, 1], labels)
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)



class CodeBert:
    """CodeBert.

    Example:
    cb = CodeBert()
    sent = ["int myfunciscool(float b) { return 1; }", "int main"]
    ret = cb.encode(sent)
    ret.shape
    >>> torch.Size([2, 768])
    """

    def __init__(self):
        """Initiate model."""
        codebert_base_path = imp.external_dir() / "codebert-base"
        if os.path.exists(codebert_base_path):
            self.tokenizer = AutoTokenizer.from_pretrained(codebert_base_path)
            self.model = AutoModel.from_pretrained(codebert_base_path)
        else:
            def cache_dir() -> Path:
                """Get storage cache path."""
                path = imp.storage_dir() / "cache"
                Path(path).mkdir(exist_ok=True, parents=True)
                return path
            cache_dir = imp.get_dir(f"{cache_dir()}/codebert_model")
            print("Loading Codebert for feature engineering...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
        self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._dev)

    def encode(self, sents: list):
        """Get CodeBert embeddings from a list of sentences."""
        tokens = [i for i in sents]
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        tokens = self.tokenizer(tokens, **tk_args).to(self._dev)
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"])[1]

    
 
#---------------------------------------- Single Graph construction ----------------------------------------------
def gget_dep_add_lines_datasetss(cache=True):
    """Cache dependent added lines for datasetss."""
    saved = imp.get_dir(imp.processed_dir() / "datasetss/eval") / "statement_labels.pkl"
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = df
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = imp.dfmp(df, imp.helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
        print(pkl.dump(lines_dict, f))
    return lines_dict


def initialize_lines_and_features(gtype="pdg", feat="all"):
    """Initialize dependency and feature settings."""
    lines = gget_dep_add_lines_datasetss()
    lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
    return lines, gtype, feat

def process_item(_id, df, codebert=None, lines=None, graph_type="pdg", feat="all"):
    """Process a single dataset item."""
    savedir = imp.get_dir(
        imp.cache_dir() / f"datasetss_linevd_codebert_{graph_type}"
    ) / str(_id)
    
    if os.path.exists(savedir):
        g = load_graphs(str(savedir))[0][0]
        if "_CODEBERT" in g.ndata:
            if feat == "codebert":
                for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                    try:
                        g.ndata.pop(i, None)
                    except:
                        print(f"No {i} in nodes feature")
            return g
    
    code, lineno, ei, eo, et = dpre.feature_extraction(f"{imp.processed_dir()}/datasetss/before/{_id}.java", graph_type)
    vuln = [1 if i in lines[_id] else 0 for i in lineno] if _id in lines else [0 for _ in lineno]
    
    g = dgl.graph((eo, ei))
    if codebert:
        code = [c.replace("\\t", "").replace("\\n", "") for c in code]
        chunked_batches = imp.chunks(code, 128)
        features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
        g.ndata["_CODEBERT"] = th.cat(features)
    
    g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
    g.ndata["_LINE"] = th.Tensor(lineno).int()
    g.ndata["_VULN"] = th.Tensor(vuln).float()
    g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
    g.edata["_ETYPE"] = th.Tensor(et).long()

    emb_path = imp.cache_dir() / f"codebert_method_level/{_id}.pt"
    g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
    
    # Node embeddings with Node2Vec
    nx_graph = g.to_networkx()
    node2vec = Node2Vec(nx_graph, dimensions=768, walk_length=5, num_walks=10, workers=4)
    model = node2vec.fit(window=5, min_count=1, batch_words=8)
    embeddings = model.wv
    node_embeddings = {int(node): embeddings[str(node)] for node in nx_graph.nodes}
    embedding_matrix = th.tensor([node_embeddings[node.item()] for node in g.nodes()], dtype=th.float)
    g.ndata['node_embedding'] = embedding_matrix
    
    # Edge embeddings
    src, dst = g.edges()
    src_embeddings = g.ndata['node_embedding'][src]
    dst_embeddings = g.ndata['node_embedding'][dst]
    edge_embeddings = (src_embeddings + dst_embeddings) / 2
    g.edata['edge_embedding'] = edge_embeddings

    # Normalize node and edge features
    g.ndata['node_embedding'] = (g.ndata['node_embedding'] - th.mean(g.ndata['node_embedding'], dim=0)) / th.std(g.ndata['node_embedding'], dim=0)
    g.edata['edge_embedding'] = (g.edata['edge_embedding'] - th.mean(g.edata['edge_embedding'], dim=0)) / th.std(g.edata['edge_embedding'], dim=0)
    
    g = dgl.add_self_loop(g)
    save_graphs(str(savedir), [g])
    return g

def cache_all_items(df, codebert, lines, graph_type="pdg", feat="all"):
    """Cache all dataset items."""
    for _id in tqdm(df.sample(len(df)).id.tolist()):
        try:
            process_item(_id, df, codebert, lines, graph_type, feat)
        except Exception as e:
            print(f"Error processing item {_id}: {e}")

def cache_codebert_method_level(df, codebert, _id):
    """Cache method-level embeddings using CodeBERT."""
    savedir = imp.get_dir(imp.cache_dir() / "codebert_method_level")
    batch_texts = df.before.tolist()
    texts = ["</s> " + ct for ct in batch_texts]
    embedded = codebert.encode(texts).detach().cpu()
    th.save(embedded, savedir / f"{_id}.pt")


#----------------------------- GNN model ---------------------------

import vuljavadetectmodel.inits.Losss as sceloss  


class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512,
        embtype: str = "codebert",
        embfeat: int = -1,  # Keep for legacy purposes
        num_heads: int = 4,
        lr: float = 1e-3,
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce", # "sce", # 
        multitask: str = "linemethod",
        stmtweight: int = 5,
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initialization."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()

        self.test_step_outputs = []

        # Set params based on embedding type
        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"

        # Loss
        if self.hparams.loss == "sce":
            self.loss = sceloss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight]) #.cuda() ---------------??????
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary", num_classes=2)
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", num_classes=2)

        # GraphConv Type
        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        # model: gat2layer
        if "gat" in self.hparams.model:
            self.gat = gnn(**gnn1_args)
            self.gat2 = gnn(**gnn2_args)
            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: contains femb
        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(embfeat * 2, self.hparams.hfeat)

        # Transform codebert embedding
        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        # Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass."""
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
        else:
            g2 = g
            h = g.ndata[self.EMBED]
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # model: contains femb
        if "+femb" in self.hparams.model:
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        # Transform h_func if wrong size
        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Edge masking (for GNNExplainer)
        if test and len(e_weights) > 0:
            g.ndata["h"] = h
            g.edata["ew"] = e_weights
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            h = self.mlpdropout(F.elu(self.fconly(hdst)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        h_func = self.fc2(
            h_func
        )  # Share weights between method-level and statement-level tasks

        if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:
            return h, h_func  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_VULN"].long()
                labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_VULN"].long()
                labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, labels_func
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss1 = self.loss(logits[0], labels)
        
        logits1 = logits[0]
        
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
            acc_func = self.accuracy(logits, labels_func)
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True, batch_size=batch_idx)
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_loss_func", loss2, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_auroc", self.auroc(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("train_mcc", self.mcc(preds, labels), prog_bar=True, batch_size=batch_idx)
        
        if not self.hparams.methodlevel:
            self.log("train_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("train_auroc_func", self.auroc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("train_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits, labels, labels_func = self.shared_step(batch)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
            
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_idx)
        self.log("val_auroc", self.auroc(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        self.log("val_mcc", self.mcc(preds, labels), prog_bar=True, batch_size=batch_idx)

        if not self.hparams.methodlevel:
            self.log("val_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("val_auroc_func", self.auroc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            self.log("val_mcc_func", self.mcc(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
    
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, labels_func = self.shared_step(batch, test=True)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None

        metrics = {
            "test_loss": loss,
            "test_acc": self.accuracy(preds, labels),
            "test_auroc": self.auroc(preds, labels),
            "test_mcc": self.mcc(preds, labels)
        }
        
        if not self.hparams.methodlevel:
            metrics["test_acc_func"] = self.accuracy(preds_func, labels_func)
            metrics["test_auroc_func"] = self.auroc(preds_func, labels_func)
            metrics["test_mcc_func"] = self.mcc(preds_func, labels_func)
           
        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Test epoch end.""" # investigate how to can save list of metric per epoch from here, then use it to make plot. 
        avg_metrics = {
            key: th.mean(th.stack([x[key] for x in self.test_step_outputs]))
            for key in self.test_step_outputs[0].keys()
        }
        # print(f"what is insight self.test_step_outputs {self.test_step_outputs}")
        self.test_step_outputs.clear()
        self.log_dict(avg_metrics)
        return
        
    def configure_optimizers(self):
        """Configure optimizers."""
        return AdamW(self.parameters(), lr=self.lr)
    
    
def modelpredict(model, g, id):
    all_preds_ = []
    all_labels_ = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    path = f"{imp.cache_dir()}/datasetss_linevd_codebert_pdg+raw/{id}"   
    g = load_graphs(path)[0][0]
    code_lines = np.array(g.ndata['_LINE'])
    batch = g
    with torch.no_grad():
        logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
        if labels is not None:
            preds_ = torch.softmax(logits[0], dim=1).cpu().numpy()
            labels_f = labels.cpu().numpy()
            all_preds_.extend(preds_)
            all_labels_.extend(labels_f)   
    all_preds_ = np.array(all_preds_)
    all_labels_ = np.array(all_labels_) 
    rounded_preds = np.where(all_preds_[:, 1] >= 0.2, 1, 0)
    # Use argmax to select the class
    # predicted_classes = np.argmax( all_preds_, axis=1)
    predicted_classes = rounded_preds
    array_predict = f"{predicted_classes}"
    code_lines = f"{code_lines}"
    code_line = code_lines[1:-1]
    array_string = array_predict[1:-1]
    predict_status = list(map(int, array_string.split()))
    lines_list = list(map(int, code_line.split()))
    result_dict = {"code line": lines_list, "vul status": predict_status}
    output = pd.DataFrame(result_dict)
    vulnerable_lines = output[output['vul status'] == 1]
    if not vulnerable_lines.empty:
        output = vulnerable_lines
    else:
        output = pd.DataFrame({
        "code line": ["No potential vulnerable line found"],  
        "vul status": ["Thank you!!!"]
        })
    return output
