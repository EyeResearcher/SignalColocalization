import argparse
from itertools import product
import json
from pathlib import Path
import time

import tqdm
from colocalize.pipeline import run_analysis
from colocalize.datasets import TissueChannel,ReferenceSet, SignalChannel, AnalysisConfig
from colocalize.transforms import MaxProjection

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep background parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the sweep configuration file")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    return parser.parse_args()

def load_config(config_path) -> dict:

    with open(config_path, "r") as f:
        return json.load(f)

def set_reference_config(names, channels, models):
    reference_sets = []
    for name, channel, model in zip(names, channels, models):
        reference_set = ReferenceSet(name=name, channel=channel,model=model)
        reference_sets.append(reference_set)
    return reference_sets

def set_signal_channel_config(names, channels, **kwargs) -> list[SignalChannel]:
    signal_channels = []
    for name, channel in zip(names, channels):
        signal_channel = SignalChannel(name=name, channel=channel, **kwargs)
        signal_channels.append(signal_channel)
    return signal_channels
def set_run_config(run_name, 
                   params,
                   input_dir : Path,
                   output_dir : Path, 
                   dendrimer : str, 
                   exclude="Map_A01.oir") -> AnalysisConfig:
    
    dendrimer_dir = input_dir / f"DAPI-{dendrimer}-RGC"
    output_dir = output_dir / f"DAPI-{dendrimer}-RGC_counts_sweep" / run_name

    config = AnalysisConfig(
                input_dir=dendrimer_dir,
                exclude = exclude,
                output_dir=output_dir,

                tissue_channel=TissueChannel(name="DAPI", channel=0),
                reference_sets=[
                                ReferenceSet(
                                    name="BRN3A",
                                    channel=2,
                                    model="cpdino_RPBMS-tiled"
                                    ),
                                ],
                signal_channels=[
                    SignalChannel(
                        name=dendrimer,
                        channel=1,
                        **params
                    )
    ],
                transforms=[MaxProjection()],
                device="cuda",
                save_masks=False,
                save_segmentation=False
                ,  
                progress=True,
                load_masks=True,
                mask_dir=Path(f"C:\\Users\\mzinn1\\Desktop\\DendrimerColocalization\\TiledRetina\\DAPI-{dendrimer}-RGC_counts\\masks"),
                )
    return config

def main():
    args = parse_args()
    config_path = args.config
    output_dir = args.output_dir
    input_dir = args.input_dir  

    # Here you would add the code to perform the sweep using the configuration file
    # and save the results to the output directory
    config = load_config(config_path)
    config_items_list = [(key, value) for key, value in config.items()]
    param_dendrimer_combos = list(product(["GD", "HD"], config_items_list))

    for i, (dendrimer, (run_name, run_params)) in enumerate(param_dendrimer_combos):
        
        
        run_config = set_run_config(run_name=run_name, 
                                    params=run_params, 
                                    input_dir=Path(input_dir), 
                                    output_dir=Path(output_dir), 
                                    dendrimer=dendrimer)
        if Path(run_config.output_dir / "cells.csv").exists():
            print(f"Skipping combination {i+1}/{len(param_dendrimer_combos)}: dendrimer={dendrimer}, run_name={run_name} because output already exists.")
            continue
        print(f"Processing combination {i+1}/{len(param_dendrimer_combos)}: dendrimer={dendrimer}, run_name={run_name}")
        start_time = time.time()
        analysis_result = run_analysis(run_config)

        
        end_time = time.time()
        print(f"Finished processing combination: dendrimer={dendrimer}, run_name={run_name} in {end_time - start_time:.2f} seconds")



if __name__ == "__main__":
    main()