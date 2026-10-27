"""ReactionT5 forward-prediction backend for SynFlowNet.

ReactionT5 predictions on reactants acts as "extra reactions" to access larger chemical space.
ReactionT5 applied to (state, building block) yields distinct, valid product returned by beam search. 
Because beam search is deterministic and every prediction is cached, the same (state, bb, k) always 
maps to the same child, which keeps the GFlowNet transition function well defined.

Reverse direction: a learned model cannot be run backwards, so every forward T5
transition is recorded in a ProvenanceStore (product -> parent, bb) and the backward
step / backward masks are read from it.
"""

import sqlite3
import threading
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def canonical(smi: str) -> Optional[str]:
    """Canonical, stereo-free SMILES (SynFlowNet's graphs drop stereochemistry)."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def largest_fragment(smi: str) -> str:
    frags = smi.split(".")
    if len(frags) == 1:
        return smi
    mols = [(Chem.MolFromSmiles(f), f) for f in frags]
    mols = [(m.GetNumHeavyAtoms(), f) for m, f in mols if m is not None]
    return max(mols)[1] if mols else smi


class ReactionT5Forward:
    """Thin, cached wrapper around the Huggingface model ReactionT5."""

    def __init__(
        self,
        model_name: str,
        num_slots: int = 3,
        num_beams: int = 10,
        max_length: int = 300,
        device: str = "cuda",
        allowed_atoms: Optional[Sequence[str]] = None,
        max_heavy_atoms: Optional[int] = None,
        cache_size: int = 200_000,
    ):

        self.num_slots = num_slots
        self.num_beams = max(num_beams, num_slots)
        self.max_length = max_length
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.allowed_atoms = set(allowed_atoms) if allowed_atoms is not None else None
        self.max_heavy_atoms = max_heavy_atoms
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        self._cache: "OrderedDict[Tuple[str, str], List[str]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @staticmethod
    def _key(reactants: Sequence[str]) -> Tuple[str, ...]:
        cans = tuple(canonical(r) or r for r in reactants)
        # order-invariant for two reactants: A + B and B + A are the same reaction
        return tuple(sorted(cans)) if len(cans) == 2 else cans

    def _postprocess(self, products: List[str], seq_scores: List[float], reactant_smiles: Tuple[str, ...], lam=0.2) -> List[str]:
        ### TODO!!!: Add the filtering steps per SyntheMol_NP
        out: List[str] = []
        #for s in raw:
        scored_beam = list(zip(products, seq_scores))
        out = self._select(reactant_smiles=reactant_smiles, scored_beam=scored_beam, products_from_smarts=None, k=self.num_slots, lam=lam)
        return out

    def _select(self, reactant_smiles, scored_beam, k, lam):
        """beam: [(product_smiles, logprob)]
        One reaction with multiple possible reactants
        """
        r_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        r_fp = sum(self._fp(m) for m in r_mols)
        r_set = {Chem.MolToSmiles(m) for m in r_mols}

        # List smiles, list quality score, list difference between fp of product and reactant
        smis, qs, deltas = [], [], [] 

        for smi, score in sorted(scored_beam, key=lambda x: -x[1]): 

            # Check if it matches the products from the smarts string
            
            mol = Chem.MolFromSmiles(smi)
            # Validity check
            if mol is None:
                continue

            # Remove products similar to reactant
            can = Chem.MolToSmiles(mol)
            if can in r_set or can in smis:                     
                continue        

            # Filter out molecules with unallowed atoms
            if self.allowed_atoms is not None and any(a.GetSymbol() not in self.allowed_atoms for a in mol.GetAtoms()):
                continue

            # Remove too big molecules
            if self.max_heavy_atoms is not None and mol.GetNumHeavyAtoms() > self.max_heavy_atoms:
                continue
           
            smis.append(can)
            qs.append(score.detach().cpu().numpy())     
            deltas.append(self._fp(mol) - r_fp)

        if not smis:
            return []
        
        q = np.array(qs)
        q = (q - q.min()) / (np.ptp(q) + 1e-9)
        D = np.stack(deltas)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        S = np.clip(D @ D.T, 0, 1)

        # Choose k products with Maximal Marginal Relevance
        # Pick candidate greedily, with score being lamb*quality + (1-lamb)*(max sim to what is already picked)
        picked = [int(q.argmax())]
        while len(picked) < min(k, len(smis)):
            score = lam * q - (1 - lam) * S[:, picked].max(axis=1)
            score[picked] = -np.inf
            picked.append(int(score.argmax()))

        return [smis[i] for i in picked]      
    
        
    def predict_batch(self, reactant_tuples: Sequence[Sequence[str]], batch_size: int = 32) -> List[List[str]]:
        """Returns, for each reactant tuple (one SMILES for unimolecular, two for bimolecular),
        up to num_slots distinct valid products."""
        keys = [self._key(rt) for rt in reactant_tuples]
        todo = sorted({k for k in keys if k not in self._cache})
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            texts = [f"REACTANT:{'.'.join(k)}REAGENT:" for k in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    max_length=self.max_length,
                    early_stopping=True,
                )
            dec = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            seq_scores = gen.sequences_scores
            with self._lock:
                for j, k in enumerate(chunk):
                    self._cache[k] = self._postprocess(products=dec[j * self.num_beams : (j + 1) * self.num_beams],
                                                       seq_scores= seq_scores[j * self.num_beams : (j + 1) * self.num_beams],
                                                       reactant_smiles=k)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        results = []
        for k in keys:
            self._cache.move_to_end(k)
            results.append(self._cache[k])
        return results

    def predict(self, reactants: Sequence[str], slot: int) -> Optional[str]:
        prods = self.predict_batch([tuple(reactants)])[0]
        return prods[slot] if slot < len(prods) else None


class ProvenanceStore:
    """Records T5 edges: (product, slot) -> (parent, bb_idx).

    Uses an in-process dict by default. Pass `db_path` to share edges between
    DataLoader worker processes (num_workers > 0) through SQLite.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._mem: Dict[Tuple[str, int], Tuple[str, int]] = {}
        self._db_path = db_path
        self._conn = None
        if db_path is not None:
            self._connect()

    def _connect(self):
        self._conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS edges (product TEXT, slot INTEGER, parent TEXT, bb INTEGER, "
            "PRIMARY KEY(product, slot))"
        )
        self._conn.commit()

    def __getstate__(self):  # make picklable for worker processes
        d = self.__dict__.copy()
        d["_conn"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self._db_path is not None:
            self._connect()

    def add(self, product: str, slot: int, parent: str, bb: int) -> Tuple[str, int]:
        """Stores the edge if (product, slot) is new; returns the edge actually stored."""
        key = (product, slot)
        if key in self._mem:
            return self._mem[key]
        if self._conn is not None:
            self._conn.execute("INSERT OR IGNORE INTO edges VALUES (?,?,?,?)", (product, slot, parent, bb))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT parent, bb FROM edges WHERE product=? AND slot=?", (product, slot)
            ).fetchone()
            self._mem[key] = (row[0], int(row[1]))
        else:
            self._mem[key] = (parent, bb)
        return self._mem[key]

    def get(self, product: str, slot: int) -> Optional[Tuple[str, int]]:
        key = (product, slot)
        if key in self._mem:
            return self._mem[key]
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT parent, bb FROM edges WHERE product=? AND slot=?", (product, slot)
            ).fetchone()
            if row is not None:
                self._mem[key] = (row[0], int(row[1]))
                return self._mem[key]
        return None


# One backend per process, shared by the task's ctx and the env's ctx (they are separate objects).
_BACKENDS: Dict[str, Tuple[ReactionT5Forward, ProvenanceStore]] = {}


def get_t5_backend(**kwargs) -> Tuple[ReactionT5Forward, ProvenanceStore]:
    db_path = kwargs.pop("provenance_db", None)
    key = repr(sorted(kwargs.items())) + str(db_path)
    if key not in _BACKENDS:
        warnings.warn(f"Loading ReactionT5 backend: {kwargs.get('model_name')}")
        _BACKENDS[key] = (ReactionT5Forward(**kwargs), ProvenanceStore(db_path))
    return _BACKENDS[key]