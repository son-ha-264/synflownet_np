import argparse
import pickle
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem
from tqdm import tqdm

from synflownet.envs.synthesis_building_env import ReactionTemplateEnvContext


def precompute_bb_masks(path, building_blocks_path=None, templates_path=None, add_hs=False):

    if building_blocks_path is None:
        building_blocks_path = Path(__file__).parent / "enamine_bbs.txt"

    with open(building_blocks_path, "r") as f:
        building_blocks = f.readlines()

    #ctx = ReactionTemplateEnvContext(building_blocks=building_blocks)
    # IMPORTANT: add_hs here MUST match task.reactions_task.add_hs used at training time, otherwise the
    # masks (HasSubstructMatch) and the env's run_reactants disagree and nearly every reaction is invalid.
    ctx_kwargs = {"building_blocks": building_blocks, "add_hs": add_hs}
    if templates_path is not None:
        with open(templates_path, "r") as f:
            ctx_kwargs["reaction_templates"] = [line.strip() for line in f if line.strip()]
    ctx = ReactionTemplateEnvContext(**ctx_kwargs)

    print("Precomputing building blocks masks for each reaction and reactant position...")
    #masks = np.zeros((2, ctx.num_bimolecular_rxns, ctx.num_building_blocks))
    #for rxn_i in tqdm(range(ctx.num_bimolecular_rxns)):
    
    ### NEW: Specify only 'template' reactions and not T5
    masks = np.zeros((2, ctx.num_template_bimolecular_rxns, ctx.num_building_blocks))
    for rxn_i in tqdm(range(ctx.num_template_bimolecular_rxns)):  

        reaction = ctx.bimolecular_reactions[rxn_i]
        reactants = reaction.rxn.GetReactants()
        for bb_j, bb in enumerate(ctx.building_blocks_mols):
            if bb is None:
                print(bb_j, building_blocks[bb_j])
            if bb.HasSubstructMatch(reactants[0]):
                masks[0, rxn_i, bb_j] = 1
            if bb.HasSubstructMatch(reactants[1]):
                masks[1, rxn_i, bb_j] = 1

    print(f"Saving precomputed masks of shape={masks.shape} to {path}")
    with open(path, "wb") as f:
        pickle.dump(masks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        type=str,
        default=Path(__file__).parent / "precomputed_bb_masks_synthemol_bbs.pkl",
    )
    parser.add_argument("--building_blocks", type=str, default=None, help="path to building-blocks .txt")
    parser.add_argument("--templates", type=str, default=None, help="path to reaction-templates .txt (SMIRKS)")
    parser.add_argument(
        "--add_hs",
        action="store_true",
        help="add explicit Hs before substructure matching; MUST match task.reactions_task.add_hs at training time",
    )
    args = parser.parse_args()
    precompute_bb_masks(
        args.out_path,
        building_blocks_path=args.building_blocks,
        templates_path=args.templates,
        add_hs=args.add_hs,
    )
    print("Done!")
    with open(args.out_path, "rb") as f:
        masks = pickle.load(f)
    print(masks.shape)
