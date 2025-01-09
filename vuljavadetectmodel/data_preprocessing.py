# this is the data preparation file
import os
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import pickle as pkl
from multiprocessing import Pool
#----------------

try:
    from  vuljavadetectmodel.inits import __inits__important as imp
except:
    import inits.__inits__important as imp
# import inits.__inits__important as imp    # all funtion from here should stard by "imp"


def _c2dhelper(item):
    """Given item with func_before, func_after, id, and dataset, save gitdiff."""
    savedir = imp.get_dir(imp.cache_dir() / item["dataset"] / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if os.path.exists(savepath):
        return
    if item["func_before"] == item["func_after"]:
        return
    ret = imp.code2diff(item["func_before"], item["func_after"])
    with open(savepath, "wb") as f:
        pkl.dump(ret, f)
        
def rdg(edges, gtype):
    """Reduce graph given type."""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
        ]


def get_codediff(dataset, iid):
    """Get codediff from file."""
    savedir = imp.get_dir(imp.cache_dir() / dataset / "gitdiff")
    savepath = savedir / f"{iid}.git.pkl"
    try:
        with open(savepath, "rb") as f:
            return pkl.load(f)
    except:
        return []

def allfunc(row):
    """Return a combined function (before + after commit) given the diff.

    diff = return raw diff of combined function
    added = return added line numbers relative to the combined function (start at 1)
    removed = return removed line numbers relative to the combined function (start at 1)
    before = return combined function, commented out added lines
    after = return combined function, commented out removed lines
    """
    readfile = get_codediff(row["dataset"], row["id"])

    ret = dict()
    ret["diff"] = "" if len(readfile) == 0 else readfile["diff"]
    ret["added"] = [] if len(readfile) == 0 else readfile["added"]
    ret["removed"] = [] if len(readfile) == 0 else readfile["removed"]
    ret["before"] = row["func_before"]
    ret["after"] = row["func_before"]

    if len(readfile) > 0:
        lines_before = []
        lines_after = []
        for li in ret["diff"].splitlines():
            if len(li) == 0:
                continue
            li_before = li
            li_after = li
            if li[0] == "-":
                li_before = li[1:]
                li_after = "        // " + li[1:]
            if li[0] == "+":
                li_before = "       // " + li[1:]
                li_after = li[1:]
            lines_before.append(li_before)
            lines_after.append(li_after)
        ret["before"] = "\n".join(lines_before)
        ret["after"] = "\n".join(lines_after)

    return ret


#--------------------------------------------------
def datasetss(minimal=True, sample=False, return_raw=False, splits="default"):
    """Read datasetss Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = imp.get_dir(imp.cache_dir() / "minimal_datasets")
    if minimal:
        try:
            df = pd.read_parquet(
                savedir / f"minimal_datasetss_{sample}.pq", engine="fastparquet"
            ).dropna()

            md = pd.read_csv(imp.cache_dir() / "datasetss/datasetss_metadata.csv", low_memory=False)
            md.groupby("project").count().sort_values("id")

            default_splits = imp.external_dir() / "datasetss_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            if "crossproject" in splits:
                project = splits.split("_")[-1]
                md = pd.read_csv(imp.cache_dir() / "datasetss/datasetss_metadata.csv", low_memory=False)
                nonproject = md[md.project != project].id.tolist()
                trid, vaid = imp.train_test_split(nonproject, test_size=0.1, random_state=1)
                teid = md[md.project == project].id.tolist()
                teid = {k: "test" for k in teid}
                trid = {k: "train" for k in trid}
                vaid = {k: "val" for k in vaid}
                cross_project_splits = {**trid, **vaid, **teid}
                df["label"] = df.id.map(cross_project_splits)

            return df
        except Exception as E:
            print(E)
            pass
    filename = "sample_Graph_V_projectKB_csv.csv" if sample else "Graph_V_projectKB_csv_version.csv" #  "Graph_V_projectKB_csv_version.csv"#
    df = pd.read_csv(imp.external_dir() / filename) 
    #df = df.rename(columns={"index": "id"})
    
    df["dataset"] = "datasetss" # change this to kbdataset

    # Remove comments
    df["func_before"] = imp.dfmp(df, imp.remove_comments, "func_before", cs=500) # , cs=500
    df["func_after"] = imp.dfmp(df, imp.remove_comments, "func_after", cs=500) # , cs=500
    #print(df["func_after"][0])
   
    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["commit_ID","func_before", "func_after", "id", "dataset"]
    imp.dfmp(df, _c2dhelper, columns=cols, ordr=False, cs=300)

    # Assign info and save
    df["info"] = imp.dfmp(df, allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)
    df['after'] = df['after'].apply(lambda x: '\n'.join(x.split('\n')[:-1]) if x.split('\n')[-1].startswith('\\') else x)
    df['before'] = df['before'].apply(lambda x: '\n'.join(x.split('\n')[:-1]) if x.split('\n')[-1].startswith('\\') else x)
    # POST PROCESSING
    dfv = df[df.vul == 1]
    # No added or removed but vulnerable 
    
    #---------------------------------removed this in comment----------------------------------------------------
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    
    # Remove functions with abnormal ending (no } or ;)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # Remove samples with mod_prop > 0.5
    # #
    # dfv["mod_prop"] = dfv.apply(
    #     lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    # )
    # dfv = dfv.sort_values("mod_prop", ascending=0)
    # dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 2, axis=1)] # 5
    # print(f"First element in after: \n{dfv['func_after'][0]}")
    # Filter by post-processing filtering
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # Make splits
    df = imp.train_val_test_split_df(df, "id", "vul")

    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"minimal_datasetss_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    
    metadata_cols = df.columns[:17].tolist() + ["project"]
    #df['after'] = df['after'].apply(lambda x: '\n'.join(x.split('\n')[:-1]) if x.split('\n')[-1].startswith('\\') else x)
    df[metadata_cols].to_csv(imp.cache_dir() / "datasetss/datasetss_metadata.csv", index=0)

    return df

def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes
def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl.id = nl.lineNumber
    nl = drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours.

    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours




def assign_line_num_to_local(nodes, edges, code):
    """Assign line number to local variable in CPG."""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
    ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        types = [i for i in types if i != 359]
        types = [i for i in types if i != 374]
        if len(types) == 0:
            continue
        assert len(types) == 1, f"Incorrect Type Assumption. --> {types}"
        block = onehop_labels[k]
        assert len(block) == 1, f"Incorrect block Assumption. ---> {types}"
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map







def get_node_edges(filepath, verbose=0):
    """Get node and edges given filepath (must run after run_joern).

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/datasetss/before/53.c"
    """
    outdir = Path(filepath).parent
    outfile = outdir / Path(filepath).name
    #print(outfile)
    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
        edges = edges.fillna("")
    #print(edges)
    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        if "lineNumber" not in nodes.columns:
            nodes["lineNumber"] = ""
        nodes = nodes.fillna("")
        try:
            nodes = nodes[
                ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
            ]
        except Exception as E:
            if verbose > 1:
                imp.debug(f"Failed {filepath}: {E}")
            return None
    #print(nodes)
    # Assign line number to local variables
    with open(filepath, "r") as f:
        code = f.readlines()
    lmap = assign_line_num_to_local(nodes, edges, code)
    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )
    nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
        nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    #edges = edges[edges.etype != "AST"]
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]
    # 
    #print(edges)
    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            # nodes = nodes.append(
            #     {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
            #     ignore_index=True,
            # )
            nodes1 = {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum}
            nodes1 = pd.DataFrame([nodes1])
            nodes = pd.concat([nodes, nodes1], ignore_index=True)
    return nodes, edges


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """Extract graph feature (basic).

    _id = svddc.datasetssDataset.itempath(177775)
    _id = svddc.datasetssDataset.itempath(180189)
    _id = svddc.datasetssDataset.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get CPG
    n, e = get_node_edges(_id)
    n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = rdg(e, graph_type.split("+")[0])
    n = drop_lone_nodes(n, e)
    #print(n)
    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        try:
            func_name = n[n.lineNumber == 1].name.item()
        except:
            print(_id)
            func_name = ""
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code
    else:
        n.code = "</s>" + " " + n.code

    # Return plain-text code, line number list, innodes, outnodes
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes



def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """Get lines that are dependent on added lines.

    Example:
    df = svdd.datasetss()
    filepath_before = "/home/david/Documents/projects/singularity-sastvd/storage/processed/datasetss/before/177775.c"
    filepath_after = "/home/david/Documents/projects/singularity-sastvd/storage/processed/datasetss/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    """
    before_graph = feature_extraction(filepath_before)[0]
    after_graph =  feature_extraction(filepath_after)[0]

    # Get nodes in graph corresponding to added lines
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # Get lines dependent on added lines in added graph
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # Filter by lines in before graph
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def helper(row):
    """Run get_dep_add_lines from dict.

    Example:
    df = svdd.datasetss()
    added = df[df.id==177775].added.item()
    removed = df[df.id==177775].removed.item()
    helper({"id":177775, "removed": removed, "added": added})
    """
    before_path = str(imp.processed_dir() / f"datasetss/before/{row['id']}.java")
    after_path = str(imp.processed_dir() / f"datasetss/after/{row['id']}.java")
    try:
        dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
    except Exception:
        dep_add_lines = []
    return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]

def get_dep_add_lines_datasetss(cache=True):
    """Cache dependent added lines for datasetss."""
    saved = imp.get_dir(imp.processed_dir() / "datasetss/eval") / "statement_labels.pkl"
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = datasetss()
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = imp.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
        print(pkl.dump(lines_dict, f))
    return lines_dict