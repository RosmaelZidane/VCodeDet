# this code refers to  the main_data_class
# contrucg graphs, train and test the model

import os
import json

import dgl
import torch
import torch as th
import numpy as np
import pandas as pd
import pickle as pkl
import torchmetrics
import networkx as nx
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from pathlib import Path
from node2vec import Node2Vec
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tsne_torch import TorchTSNE as TSNE
from dgl import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import matthews_corrcoef, ndcg_score, f1_score, precision_score
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score 





import inits.__inits__important as imp    # all funtion from here should stard by "imp"
import inits.Losss as sceloss  
import data_preprocessing as dpre  ### Always usde this fucntion if you have to use a function from data_preprocessing.py
import graph_subprocess as gsp     # always use this definition if there is a function to use from graph_subprocess.py



### Prepare the embedding model : CodeBERT


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

run_id = imp.get_run_id()
savepath = imp.get_dir(imp.processed_dir() / "codebert" / run_id)

# # ----------- Finetune CodeBert--------------
# model = LitCodebert()
# data = datasetssDatasetNLPDataModule(datasetssDatasetNLP, batch_size=64)
# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
# trainer = pl.Trainer(
#     accelerator="auto",
#     max_epochs= 10,
#     default_root_dir=savepath,
#     num_sanity_val_steps=0,
#     callbacks=[checkpoint_callback],
# )
# print("[INFO -------------->] Train the CodeBERT model on the corpus ")
# print("[INFO ----------->]")
# print("[INFO --->] Finetune CodeBERT")
# trainer.fit(model, data)
# trainer.test(model, data)


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
            print("Loading Codebert...")
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

# remove this function at the end
def plot_embeddings(embeddings, words):
    """Plot embeddings.

    import sastvd.helpers.datasets as svdd
    cb = CodeBert()
    df = svdd.datasetss()
    sent = " ".join(df.sample(5).before.tolist()).split()
    plot_embeddings(cb.encode(sent), sent)
    """
    tsne = TSNE(n_components=2, n_iter=2000, verbose=True)
    Y = tsne.fit_transform(embeddings)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(words, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
    
    
    
#---------------------------------------- Graph construction ----------------------------------------------


class datasetssDataset:
    """Represent datasetss as graph dataset."""

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default"):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(imp.processed_dir() / "datasetss/before/*nodes*"))
        ]
        self.df = dpre.datasetss(splits=splits)
        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        # Balance training set
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Correct ratio for test set
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0) ##----->>>
            self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = imp.dfmp(
            self.df, datasetssDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        

    def itempath(_id):
        """Get itempath path from item id."""
        return imp.processed_dir() / f"datasetss/before/{_id}.java"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"datasetss/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(datasetssDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(datasetssDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(datasetssDataset.itempath(_id)))
            return False

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"datasetssDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"



def get_sast_lines(sast_pkl_path):
    """Get sast lines from path to sast dump."""
    ret = dict()
    ret["cppcheck"] = set()
    ret["rats"] = set()
    ret["flawfinder"] = set()

    try:
        with open(sast_pkl_path, "rb") as f:
            sast_data = pkl.load(f)
        for i in sast_data:
            if i["sast"] == "cppcheck":
                if i["severity"] == "error" and i["id"] != "syntaxError":
                    ret["cppcheck"].add(i["line"])
            elif i["sast"] == "flawfinder":
                if "CWE" in i["message"]:
                    ret["flawfinder"].add(i["line"])
            elif i["sast"] == "rats":
                ret["rats"].add(i["line"])
    except Exception as E:
        print(E)
        pass
    return ret




#-----draw a dgl graph --------------

def draw_dgl_graph(dgl_graph):
    """Convert DGL graph to NetworkX graph"""
    nx_graph = dgl_graph.to_networkx()
    pos = nx.spring_layout(nx_graph) 
    nx.draw(nx_graph, pos, with_labels=True, 
            node_color='skyblue', node_size=400, 
            edge_color='black', linewidths=1, 
            font_size=8)
 
    plt.show()


class datasetssDatasetLineVD(datasetssDataset):
    """IVDetect version of datasetss."""

    def __init__(self, gtype="pdg", feat="all", **kwargs):
        """Init."""
        super(datasetssDatasetLineVD, self).__init__(**kwargs)
        lines = dpre.get_dep_add_lines_datasetss()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines
        self.graph_type = gtype 
        self.feat = feat

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = imp.get_dir(
            imp.cache_dir() / f"datasetss_linevd_codebert_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            
            if "_CODEBERT" in g.ndata:
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                        try:
                            g.ndata.pop(i, None)
                        except:
                            print(f"No {i} in nodes feature")
                return g
            
        code, lineno, ei, eo, et = dpre.feature_extraction(
            datasetssDataset.itempath(_id), self.graph_type
        )
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
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
        
        # # Node embeddings step 
        nx_graph = g.to_networkx() 
        node2vec = Node2Vec(nx_graph, dimensions=768, walk_length= 5, num_walks= 10, workers=4)
        model = node2vec.fit(window = 5, min_count = 1, batch_words = 8)
        embeddings = model.wv
        node_embeddings = {int(node): embeddings[str(node)] for node in nx_graph.nodes}
        embedding_matrix = torch.tensor([node_embeddings[node.item()] for node in g.nodes()], dtype=torch.float)
        g.ndata['node_embedding'] = embedding_matrix
        
        # edges embedding
        src, dst = g.edges()
        src_embeddings = g.ndata['node_embedding'][src]
        dst_embeddings = g.ndata['node_embedding'][dst]
        edge_embeddings = (src_embeddings + dst_embeddings) / 2
        g.edata['edge_embedding'] = edge_embeddings
        
        # normalise nodes and edge features
        g.ndata['node_embedding'] = (g.ndata['node_embedding'] - th.mean(g.ndata['node_embedding'], dim = 0))/th.std(g.ndata['node_embedding'], dim = 0)
        g.edata['edge_embedding'] = (g.edata['edge_embedding'] - th.mean(g.edata['edge_embedding'], dim = 0))/th.std(g.edata['edge_embedding'], dim = 0)
        
        
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g

    def cache_items(self, codebert):
        """Cache all items."""
        for i in tqdm(self.df.sample(len(self.df)).id.tolist()):
            try:
                self.item(i, codebert)
            except Exception as E:
                print(E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = imp.get_dir(imp.cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = imp.chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches):
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            if set(batch_ids).issubset(done):
                continue
            texts = ["</s> " + ct for ct in batch_texts]
            embedded = codebert.encode(texts).detach().cpu()
            assert len(batch_texts) == len(batch_ids)
            for i in range(len(batch_texts)):
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        try:
            m = self.item(self.idx2id[idx])
        except:
            print(f"the null id is the from -------------------> {self.idx2id[idx]}")
            print(f"------------idx-------------- {idx}")  
            
        return self.item(self.idx2id[idx]) 

    

class datasetssDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for datasetss."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
    ):
        """Init class from datasetss dataset."""
        super().__init__()
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        self.train = datasetssDatasetLineVD(partition="train", **dataargs)
        self.val = datasetssDatasetLineVD(partition="val", **dataargs)
        self.test = datasetssDatasetLineVD(partition="test", **dataargs)
        codebert = CodeBert()
        self.train.cache_codebert_method_level(codebert)
        self.val.cache_codebert_method_level(codebert)
        self.test.cache_codebert_method_level(codebert)
        self.train.cache_items(codebert)
        self.val.cache_items(codebert)
        self.test.cache_items(codebert)
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
                
     
    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.DataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=10,
        ) 

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val), num_workers=10)))
            return self.node_dl(g)
        return GraphDataLoader(self.val, shuffle = False, batch_size=self.batch_size, num_workers=10)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, shuffle = False, batch_size=32, num_workers=10)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, shuffle = False, batch_size=32, num_workers=10)
    


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
    
    
    
#  train the classifie

model = LitGNN( 
               hfeat=  512,
               embtype= "codebert",
               methodlevel=False,
               nsampling=True,
               model= "gat2layer",
               loss="ce",
               hdropout=0.3,
               gatdropout=0.2,
               num_heads=4,
               multitask="linemethod", 
               stmtweight=1,
               gnntype="gat",
               scea=0.4,
               lr=1e-4,
               )

  # Load data
samplesz = -1
data = datasetssDatasetLineVDDataModule(
    batch_size=64,
    sample=samplesz,
    methodlevel=False,
    nsampling=True,
    nsampling_hops=2,
    gtype= "pdg+raw",
    splits="default"
    )
max_epochs = 250 #200 # 100 # 

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
metrics = ["train_loss", "val_loss", "val_auroc"]
trainer = pl.Trainer(
    accelerator= "auto",
    devices= "auto",
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback], 
    max_epochs=max_epochs,
    )
print(f"[INFO ---] Training the model")
trainer.fit(model, data) 

# Save the model checkpoint before testing
checkpoint_path = f"{imp.outputs_dir()}/checkpoints/best-Model.ckpt"
# trainer.save_checkpoint(checkpoint_path)
model = LitGNN.load_from_checkpoint(checkpoint_path)
trainer.test(model, data)


def calculate_metrics(model, data):
    """
    Calculate ranking metrics: MRR, N@5, MFR,
    and classification metrics: F1-Score, Precision.
    """
    def mean_reciprocal_rank(y_true, y_scores):
        order = np.argsort(y_scores, axis=1)[:, ::-1]
        rank = np.argwhere(order == y_true[:, None])[:, 1] + 1
        return np.mean(1.0 / rank)

    def precision_at_n(y_true, y_scores, n=5):
        order = np.argsort(y_scores, axis=1)[:, ::-1]
        top_n = order[:, :n]
        return np.mean(np.any(top_n == y_true[:, None], axis=1))

    def mean_first_rank(y_true, y_scores):
        order = np.argsort(y_scores, axis=1)[:, ::-1]
        rank = np.argwhere(order == y_true[:, None])[:, 1] + 1
        return np.mean(rank)

    # Extract function-level predictions and true labels
    all_preds_ = []
    all_labels_ = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    for batch in data.test_dataloader():
        with torch.no_grad():
            logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
            if labels is not None:  
                preds_ = torch.softmax(logits[0], dim=1).cpu().numpy()
                labels_f = labels.cpu().numpy()
                all_preds_.extend(preds_)
                all_labels_.extend(labels_f)

    all_preds_ = np.array(all_preds_)
    all_labels_ = np.array(all_labels_)

    # Compute ranking metrics
    MRR = mean_reciprocal_rank(all_labels_, all_preds_)
    N5 = precision_at_n(all_labels_, all_preds_, n=5)
    MFR = mean_first_rank(all_labels_, all_preds_)

    predicted_classes = np.argmax(all_preds_, axis=1)
    f1_c = f1_score(all_labels_, predicted_classes, average="macro")
    precision = precision_score(all_labels_, predicted_classes, average="macro")
    accuracy = accuracy_score(all_labels_, predicted_classes, normalize= True )
    recall = recall_score(all_labels_, predicted_classes, average= "macro") # average=None, zero_division=np.nan
    roc_ = roc_auc_score(all_labels_, predicted_classes, average= "macro")
    mcc_ = matthews_corrcoef(all_labels_, predicted_classes)
    
    
    prediction = pd.DataFrame({"true label": all_labels_,
                          "Predicted_label": predicted_classes})
    # print(f"Predict label {predicted_classes}")
    # print(f"true label {all_labels_}")
    
    return {
        "accuracy": accuracy,
        "Precision": precision,
        "F1-Score": f1_c,
        "recall" : recall,
        "roc_auc" : roc_,
        "mcc": mcc_,
        "MRR": MRR,
        "N@5": N5,
        "MFR": MFR,
    }, prediction

metrics = calculate_metrics(model, data)[0]
dfm = pd.DataFrame([metrics])
dfm.to_csv(f"{imp.outputs_dir()}/evaluation_metrics.csv", index=False)
prediction_ = calculate_metrics(model, data)[1]
prediction_.to_csv(f"{imp.outputs_dir()}/predict_label.csv", index = False)
print(f"Metris {metrics} \n-----------------------------> Done ")







# #  Predict in a single function by prodiding  the graph in the path
# all_preds_ = []
# all_labels_ = []
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# model.eval()
# path = "/home/GNN2025/VJavaDet/vuljavadetectmodel/storage/cache/datasetss_linevd_codebert_pdg+raw/90000000000"
    
# g = load_graphs(path)[0][0]
# batch = g
# # print(f"batch {batch.number_of_nodes()} \n {batch.ndata['_VULN']}")
# with torch.no_grad():
#     logits, labels, labels_func = model.shared_step(batch.to(device), test=True)
#     if labels is not None:
#         preds_ = torch.softmax(logits[0], dim=1).cpu().numpy()
#         labels_f = labels.cpu().numpy()
#         all_preds_.extend(preds_)
#         all_labels_.extend(labels_f)   
#     all_preds_ = np.array(all_preds_)
#     all_labels_ = np.array(all_labels_) 
# # Adjust the predictions based on a custom threshold (e.g., 0.3)
# threshold = 0.01
# adjusted_preds = np.where(all_preds_ <= threshold, 0, all_preds_)
# # Use argmax to select the class
# final_preds = np.argmax(adjusted_preds, axis=1)
# predicted_classes = np.argmax( all_preds_, axis=1)
# f1_c = f1_score(all_labels_, predicted_classes, average="macro")
# precision = precision_score(all_labels_, predicted_classes, average="macro")
# accuracy = accuracy_score(all_labels_, predicted_classes, normalize= True )
# recall = recall_score(all_labels_, predicted_classes,average = "macro")
# # roc_ = roc_auc_score(all_labels_, predicted_classes, average= "macro")
# mcc_ = matthews_corrcoef(all_labels_, predicted_classes)
# print(f"--->>> Predict label {all_preds_}")
# print(f"--->>++ {final_preds}")
# print(f"-->>> Predicted class {predicted_classes}")
# print(f"--->>> true label {all_labels_}")