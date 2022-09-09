import string
from typing import Any, List, Tuple, Union, Dict
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.loop import add_remaining_self_loops
from captum.attr._utils.typing import (
    TargetType,
)

from .base_explainer import WalkBase
EPS = 1e-15



class CustomExplainer(WalkBase):
    r"""
    An implementation of GradCAM on graph in
    `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        explain_graph (bool, optional): Whether to explain graph classification model.
            (default: :obj:`False`)

    .. note::
        For node classification model, the :attr:`explain_graph` flag is False.
        For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self, model: nn.Module):
        super().__init__(model, explain_graph=True)
        self.interpreter = GNNInterpreter(model, featurizer, log=False)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs)\
            -> Union[Tuple[None, List, List[Dict]], Tuple[List, List, List[Dict]]]:
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype: (None, list, list)

        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        self.model.eval()
        super().forward(x, edge_index)

        labels = tuple(i for i in range(kwargs.get('num_classes')))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        if kwargs.get('edge_masks'):
            edge_masks = kwargs.pop('edge_masks')
            hard_edge_masks = [self.control_sparsity(mask, kwargs.get('sparsity')).sigmoid() for mask in edge_masks]
        else:
            edge_masks = []
            hard_edge_masks = []
            for ex_label in ex_labels:
                attr = self.calculate_nodes_attribution(
                    smiles=kwargs.get('smiles'), 
                    label=ex_label, 
                    replace_atoms_with=kwargs.get('replace_atoms_with'), 
                    replace_atom_alg=kwargs.get('replace_atom_alg'), 
                    calculate_atom_weight_alg=kwargs.get('calculate_atom_weight_alg')
                )
                # attr = normalize(attr)
                mask = torch.from_numpy(attr).type(torch.float32).squeeze()
                mask = (mask[self_loop_edge_index[0]] + mask[self_loop_edge_index[1]]) / 2
                edge_masks.append(mask.detach())
                mask = self.control_sparsity(mask, kwargs.get('sparsity'))
                mask = mask.sigmoid()
                hard_edge_masks.append(mask.detach())

        # Store related predictions for further evaluation.
        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(x.type(torch.float32), edge_index, hard_edge_masks, **kwargs)

        return edge_masks, hard_edge_masks, related_preds

    def calculate_nodes_attribution(self, smiles: string, label: int, replace_atoms_with: string, replace_atom_alg: string, calculate_atom_weight_alg: string) -> List[int]:
        nodes_weights = self.interpreter.calc_atoms_weight(smiles, label, replace_atoms_with, replace_atom_alg, calculate_atom_weight_alg)

        return nodes_weights


import torch
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Atom, AtomValenceException
from rdkit import RDLogger
import torch.nn.functional as F

RDLogger.DisableLog('rdApp.*')

from ..dataset.mol_dataset import x_map

#returns tuple(nodes, edges)
def featurizer(mol: Chem.Mol):
    if mol is None:
        raise "Featurizer molecule cannot be none"

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(
            str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]

    return (x, edge_index)

class GNNInterpreter:
    def __init__(self, model, featurizer, log=False):
        self.model = model
        self.featurizer = featurizer
        self.log = log
        self.organic_atoms = [Atom(symbol) for symbol in  ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']]
    
    def _log(self, msg):
        if self.log:
            print(msg)
    
    def calc_atoms_weight(self,  
                            smiles, # molecule to explain model on
                            label, # what label prediction to explain
                            replace_atoms_with='all', # others: zero or atom label
                            replace_atoms_alg='number', # others: atom
                            calculate_atoms_weight_alg='signed', # others: absolute
                            ):
        self.mol = Chem.MolFromSmiles(smiles)
        self.label = label.item()

        atom_weights = []
        for atom in self.mol.GetAtoms():
            atom_weights.append(self._calculate_atom_weight(atom, replace_atoms_with, replace_atoms_alg, calculate_atoms_weight_alg))

        return np.array(atom_weights, dtype=np.float)

    def _calculate_atom_weight(self, atom, replace_atom_with, replace_atom_alg, calculate_atom_weight_alg):
        model_preds = []

        # get model preds
        if replace_atom_with == 'all':
            for organic_atom in self.organic_atoms:
                model_preds.append(self._get_replaced_data_prediction(atom, organic_atom, replace_atom_alg))
        elif replace_atom_with == 'zero':
            raise "Not obvious how to implement zero replacement when one-hot-encoding not used"
            # model_preds.append(self._get_replaced_data_prediction(atom, None, replace_atom_alg))
        else:
            organic_atom = Atom(replace_atom_with)
            model_preds.append(self._get_replaced_data_prediction(atom, organic_atom, replace_atom_alg))

        model_preds = np.array(model_preds)
        original_pred = self.get_original_pred()

        #calculate weight
        if calculate_atom_weight_alg == 'absolute':
            prepared_diff = np.abs(original_pred - model_preds)
        elif calculate_atom_weight_alg == 'signed':
            prepared_diff = original_pred - model_preds

        not_nans = prepared_diff[~np.isnan(prepared_diff)]
        atom_weight = not_nans.mean() if len(not_nans) != 0 else np.nan

        return atom_weight

    def _get_replaced_data_prediction(self, atom, replacement, replace_atom_alg):
        if replacement:
            try:
                replaced_mol = self._getReplacedMol(atom, replacement, replace_atom_alg)
            except AtomValenceException:
                self._log(f"Replacing {atom.GetSymbol()} with {replacement.GetSymbol()} failed.")
                return np.nan

            nodes, edges = self.featurizer(replaced_mol)

            self._log(f'before replacement {atom.GetSymbol()} at {atom.GetIdx()}: Chem.MolToSmiles(self.mol)')
            self._log(f'after  replacement {atom.GetSymbol()} at {atom.GetIdx()}: Chem.MolToSmiles(replaced_mol)')
        else:
            nodes, edges = self.featurizer(self.mol)
            nodes[atom.GetIdx()][:1] = torch.zeros((1,)) #TODO: how to zero?? atom is just a number from 0 to 119

        x = nodes.type(torch.float32)
        edge_index = torch.LongTensor(edges)
        batch = torch.zeros(x.shape[0], dtype=torch.int64)

        predictions = self.model(x, edge_index, batch)

        return predictions[0][self.label].item()

    def get_original_pred(self, return_tensor=False):
        nodes, edges = self.featurizer(self.mol)
        x = nodes.type(torch.float32)
        edge_index = torch.LongTensor(edges)
        batch = torch.zeros(x.shape[0], dtype=torch.int64)
        output =  self.model(x, edge_index, batch)[0]

        return output[self.label] if return_tensor else output[self.label].item()

    # throws
    def _getReplacedMol(self, atom, replacement, replace_atom_alg):
        molRW = RWMol(self.mol)

        if replace_atom_alg == 'number':
            molRW.GetAtoms()[atom.GetIdx()].SetAtomicNum(replacement.GetAtomicNum())
        elif replace_atom_alg == 'atom':
            molRW.ReplaceAtom(atom.GetIdx(), replacement, preserveProps=True)

        molRW.UpdatePropertyCache()

        return molRW