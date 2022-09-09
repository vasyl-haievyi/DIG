import os
import torch
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from benchmarks.xgraph.gnnNets import get_gnnNets
from benchmarks.xgraph.utils import check_dir, fix_random_seed, Recorder, perturb_input
from benchmarks.xgraph.dataset import get_dataset, get_dataloader, SynGraphDataset
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method import CustomExplainer

from torch_geometric.utils import add_remaining_self_loops


@hydra.main(config_path="config", config_name="config")
def pipeline(config):
    assert( config.datasets.dataset_name in ['bbbp', 'bace'])

    config.models.param = config.models.param[config.datasets.dataset_name]
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)
    config.record_filename = os.path.join(config.record_filename, f"{config.datasets.dataset_name}.json")
    print(OmegaConf.to_yaml(config))
    fix_random_seed(config.random_seed)
    recorder = Recorder(config.record_filename)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    # bbbp warning
    dataset = get_dataset(config.datasets.dataset_root,
                          config.datasets.dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.models.param.graph_classification:
        dataloader_params = {'batch_size': config.models.param.batch_size,
                             'random_split_flag': config.datasets.random_split_flag,
                             'data_split_ratio': config.datasets.data_split_ratio,
                             'seed': config.datasets.seed}
        loader = get_dataloader(dataset, **dataloader_params)
        test_indices = loader['test'].dataset.indices
    else:
        node_indices_mask = (dataset.data.y != 0) * dataset.data.test_mask
        node_indices = torch.where(node_indices_mask)[0]

    model = get_gnnNets(input_dim=dataset.num_node_features,
                        output_dim=dataset.num_classes,
                        model_config=config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_saving_dir,
                                         config.datasets.dataset_name,
                                         f"{config.models.gnn_name}_"
                                         f"{len(config.models.param.gnn_latent_dim)}l_best.pth"))['net']
    model.load_state_dict(state_dict)

    model.to(device)
    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          'Custom')
    check_dir(explanation_saving_dir)

    explainer = CustomExplainer(model)

    index = 0
    x_collector = XCollector()

    all_test_indicies = len(dataset[test_indices])

    for i, data in enumerate(dataset[test_indices]):
        index += 1
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.to(device)

        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt')):
            edge_masks = torch.load(os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
            edge_masks = [edge_mask.to(device) for edge_mask in edge_masks]

            print(f"load example {test_indices[i]}.")
            edge_masks, hard_edge_masks, related_preds = \
                explainer(data.x, data.edge_index,
                            sparsity=config.explainers.sparsity,
                            num_classes=dataset.num_classes,
                            smiles=data.smiles,
                            edge_masks=edge_masks)

        else:
            edge_masks, hard_edge_masks, related_preds = \
                explainer(data.x, data.edge_index,
                            sparsity=config.explainers.sparsity,
                            num_classes=dataset.num_classes,
                            smiles=data.smiles,
                            replace_atoms_with=config.explainers.replace_atoms_with, 
                            replace_atom_alg=config.explainers.replace_atom_alg, 
                            calculate_atom_weight_alg=config.explainers.calculate_atom_weight_alg)
            edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
            torch.save(edge_masks, os.path.join(explanation_saving_dir, f'example_{test_indices[i]}.pt'))
        prediction = model(data).argmax(-1).item()

        x_collector.collect_data(hard_edge_masks, related_preds, label=prediction)

        print(f'Pocessed {i} / {all_test_indicies}')


    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv: .4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    experiment_data = {
        'fidelity': x_collector.fidelity,
        'fidelity_inv': x_collector.fidelity_inv,
        'sparsity': x_collector.sparsity,
    }

    if x_collector.accuracy:
        print(f'Accuracy: {x_collector.accuracy}')
        experiment_data['accuracy'] = x_collector.accuracy
    if x_collector.stability:
        print(f'Stability: {x_collector.stability}')
        experiment_data['stability'] = x_collector.stability

    recorder.append(experiment_settings=['custom', f"{config.explainers.sparsity}", f"{config.explainers.replace_atoms_with} + {config.explainers.replace_atom_alg} + {config.explainers.calculate_atom_weight_alg}"],
                    experiment_data=experiment_data)

    recorder.save()


if __name__ == '__main__':
    import sys
    sys.argv.append('explainers=custom_explainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explanation_result_dir={os.path.join(os.path.dirname(__file__), 'results')}")
    sys.argv.append(f"record_filename={os.path.join(os.path.dirname(__file__), 'result_jsons')}")
    pipeline()
