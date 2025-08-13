"""
GNINA Runner Module
===================

This module provides functionality for integrating GNINA—a molecular docking and scoring tool—into the ProtFlow framework. It enables automated setup, execution, and output processing for GNINA docking jobs, making it easy to include GNINA in protein–ligand interaction pipelines.

Detailed Description
--------------------
The `GNINA` class encapsulates all logic necessary to run GNINA on a set of input protein–ligand complexes. It supports docking of ligands into protein structures, manages autobox setup around ligand binding sites, and parses docking score outputs into a structured format. The class is intended to integrate seamlessly into the broader ProtFlow ecosystem.

Key Features:
- Automatically prepares receptor and ligand input files for GNINA.
- Handles ligand and receptor chain processing using Biopython and ProtFlow tools.
- Submits and monitors docking jobs, either locally or on HPC clusters using jobstarter backends.
- Extracts docking scores and generates per-pose output PDBs for downstream analysis.

Usage
-----
Typical usage involves instantiating the `GNINA` class and invoking the `run` method with required inputs:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from protflow.runners.gnina import GNINA

    poses = Poses()
    jobstarter = JobStarter()

    gnina = GNINA()
    results = gnina.run(
        poses=poses,
        prefix="gnina_experiment",
        ligand_chain="Z",
        jobstarter=jobstarter,
        options="autobox_add=4 exhaustiveness=8",
        overwrite=True
    )

    print(results)

Further Notes
-------------
- Input Requirements: Each pose must include a ligand chain specified by `ligand_chain`. This chain is extracted to guide the autoboxing step.
- Output: Docked ligand structures and docking scores are stored in an organized directory tree. A DataFrame with all scores and output paths is returned.
- Chain Handling: Ligand and receptor chains are separated and reassembled post-docking using the `ChainRemover` and `add_chain` utilities from ProtFlow.

This module is ideal for users who want to incorporate GNINA docking into automated pipelines for structure-based drug design or protein engineering.

Author
------
David Stoll, Adrian Tripp

Version
-------
0.1.0
"""

"""
GNINA Module
=================

This module provides the functionality to integrate GNINA within the ProtFlow framework. It offers tools to run GNINA, handle its inputs and outputs, and process the resulting data in a structured and automated manner.

Detailed Description
--------------------
The `GNINA` class encapsulates the functionality necessary to execute GNINA runs. It manages the configuration of paths to essential scripts and Python executables, sets up the environment, and handles the execution of the diffusion processes. It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of GNINA into larger computational workflows. It supports the automatic setup of job parameters, execution of GNINA commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps.

Usage
-----
To use this module, create an instance of the `GNINA` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `GNINA` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from GNINA import GNINA

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the GNINA class
    GNINA = GNINA()

    # Run the diffusion process
    results = GNINA.run(
        poses=poses,
        prefix="experiment_1",
        jobstarter=jobstarter,
        nseq=10,
        model_type="ligand_mpnn",
        options="some_option=some_value",
        pose_options=["pose_option=pose_value"],
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
- Edge Cases: The module handles various edge cases, such as empty pose lists and the need to overwrite previous results. It ensures robust error handling and logging for easier debugging and verification of the process.
- Customizability: Users can customize the process through multiple parameters, including the number of sequences, specific options for the GNINA script, and options for handling pose-specific parameters.
- Integration: The module seamlessly integrates with other components of the ProtFlow framework, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate GNINA into their protein design and analysis workflows. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, especially those related to job management in HPC environments.

Author
------
Markus Braun, Adrian Tripp

Version
-------
0.1.0
"""
# general imports
import os
import logging
from glob import glob


# dependencies
import pandas as pd


# custom
import protflow.config
from protflow.poses import description_from_path
from protflow.poses import Poses
import protflow.utils.biopython_tools as bpt
from protflow.poses import Poses
from protflow.jobstarters import JobStarter
from protflow.runners import Runner, RunnerOutput, parse_generic_options, options_flags_to_string
from protflow.tools.protein_edits import ChainRemover
from protflow.utils.biopython_tools import load_structure_from_pdbfile, save_structure_to_pdbfile

##helper functions


def weighted_average(poses, group_col, score_col, weight_col, inverse: bool = False, weight_exponent: float = 1):
    """
    Calculate the weighted average of a score column within groups, 
    using a specified column where the weight is determined based on its squared values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_col (str): Column name to group by.
    score_col (str): Column to calculate the weighted average for.
    weight_col (str): Column used for weight calculation.
    inverse (bool): If True, lower values in weight_col result in higher weights. 
                    If False, higher values in weight_col result in higher weights.
    weight_exponent (float): Exponent for weight calculation.

    Returns:
    pd.DataFrame: A DataFrame with the group column and weighted averages.
    """
        
    # Compute weight based on the square of the weight_col values
    if inverse:
        poses.df['weight'] = 1 / (poses.df[weight_col] ** weight_exponent)  # Lower values → Higher weights
    else:
        poses.df['weight'] = poses.df[weight_col] ** weight_exponent  # Higher values → Higher weights

    # Calculate weighted average per group
    weighted_avg = poses.df.groupby(group_col).apply(lambda g: (g[score_col] * g['weight']).sum() / g['weight'].sum())

    # Rename and reset index for merging
    return weighted_avg.reset_index(name=f"{score_col}_weighted")

class GNINA(Runner):
    """
    GNINA Class
    ================

    The `GNINA` class provides the necessary methods to execute GNINA runs within the ProtFlow framework. This class is responsible for managing the configuration, execution, and output processing of GNINA tasks.

    Detailed Description
    --------------------
    The `GNINA` class integrates GNINA into the ProtFlow pipeline by setting up the environment, running the diffusion process, and collecting the results. It ensures that the inputs and outputs are handled efficiently, making the data readily available for further analysis.

    Key Features:
    - Manages paths to essential scripts and executables.
    - Configures and executes GNINA processes.
    - Collects and processes output data into a structured DataFrame format.
    - Handles various edge cases and supports custom configurations through multiple parameters.

    Usage
    -----
    To use this class, initialize it with the appropriate script and Python paths, along with an optional job starter. The main functionality is provided through the `run` method, which requires parameters such as poses, prefix, and additional options for customization.

    Example
    -------
    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from GNINA import GNINA

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the GNINA class
        GNINA = GNINA()

        # Run the diffusion process
        results = GNINA.run(
            poses=poses,
            prefix="experiment_1",
            jobstarter=jobstarter,
            nseq=10,
            model_type="ligand_mpnn",
            options="some_option=some_value",
            pose_options=["pose_option=pose_value"],
            overwrite=True
        )

        # Access and process the results
        print(results)

    Notes
    -----
    This class is designed to work within the ProtFlow framework and assumes that the necessary configurations and dependencies are properly set up. It leverages shared data structures and configurations from ProtFlow to provide a seamless integration experience.

    Author
    ------
    David Stoll, Markus Braun, Adrian Tripp

    Version
    -------
    0.1.0
    """
    def __init__(self, script_path:str=protflow.config.GNINA_PATH, pre_cmd:str=protflow.config.GNINA_PRE_CMD, jobstarter:JobStarter=None) -> None:
        """
        Initializes the GNINA class.

        Parameters:
            script_path (str, optional): The path to the GNINA script. Defaults to the configured script path in ProtFlow.
            python_path (str, optional): The path to the Python executable to run the GNINA script. Defaults to the configured Python path in ProtFlow.
            jobstarter (JobStarter, optional): An instance of the JobStarter class to manage job submissions. If not provided, it will use the default job starter configuration.

        Detailed Description
        --------------------
        The `__init__` method sets up the necessary paths and configurations for running GNINA. It searches for the provided script and Python
        paths to ensure they are correct and sets them as instance attributes. Additionally, it initializes the job starter, which manages the execution
        of jobs in high-performance computing (HPC) environments. This method ensures that all configurations are correctly set up before running any
        GNINA tasks.
        """


        self.script_path = self.search_path(script_path, "GNINA_PATH")
        self.name = "gnina.py"
        self.pre_cmd = pre_cmd
        self.index_layers = 1
        self.jobstarter = jobstarter
        
    def __str__(self):
        return "gnina.py"

    def run(self, poses: Poses, prefix: str, ligand_chain: str = None, ligand_path: str = None, options: str = None, pose_options: object = None, overwrite: bool = False, jobstarter: JobStarter = None) -> Poses:
        """ 
        Execute the GNINA process with given poses and jobstarter configuration.

        This method sets up and runs the GNINA process using the provided poses and jobstarter object. It handles the configuration, execution, and collection of output data, ensuring that the results are organized and accessible for further analysis.

        Parameters:
            poses (Poses): The Poses object containing the protein structures.
            prefix (str): A prefix used to name and organize the output files.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.
            nseq (int, optional): The number of sequences to generate for each input pose. Defaults to None.
            model_type (str, optional): The type of model to use. Defaults to 'ligand_mpnn'.
            options (str, optional): Additional options for the GNINA script. Defaults to None.
            pose_options (object, optional): Pose-specific options for the GNINA script. Defaults to None.
            fixed_res_col (str, optional): Column name in the poses DataFrame specifying fixed residues. Defaults to None.
            design_res_col (str, optional): Column name in the poses DataFrame specifying residues to be redesigned. Defaults to None.
            return_seq_threaded_pdbs_as_pose (bool, optional): If True, return sequence-threaded PDBs as poses. Defaults to False.
            preserve_original_output (bool, optional): If True, preserve the original output files. Defaults to True.
            overwrite (bool, optional): If True, overwrite existing output files. Defaults to False.

        Returns:
            Poses: The updated Poses object containing the results of the GNINA process.

        Raises:
            FileNotFoundError: If required files or directories are not found during the execution process.
            ValueError: If invalid arguments are provided to the method.

        Examples:
            Here is an example of how to use the `run` method:

            .. code-block:: python

                from protflow.poses import Poses
                from protflow.jobstarters import JobStarter
                from GNINA import GNINA

                # Create instances of necessary classes
                poses = Poses()
                jobstarter = JobStarter()

                # Initialize the GNINA class
                GNINA = GNINA()

                # Run the diffusion process
                results = GNINA.run(
                    poses=poses,
                    prefix="experiment_1",
                    jobstarter=jobstarter,
                    nseq=10,
                    model_type="ligand_mpnn",
                    options="some_option=some_value",
                    pose_options=["pose_option=pose_value"],
                    overwrite=True
                )

                # Access and process the results
                print(results)

        Further Details:
            - **Setup and Execution:** The method ensures that the environment is correctly set up, directories are prepared, and necessary commands are constructed and executed.
            - **Output Management:** The method handles the collection and processing of output data, ensuring that results are organized and accessible for further analysis.
            - **Customization:** Extensive customization options are provided through parameters, allowing users to tailor the process to their specific needs.

        This method is designed to streamline the execution of GNINA processes within the ProtFlow framework, making it easier for researchers and developers to perform and analyze protein design simulations.
        """


        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        
        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip GNINA.
        scorefile = os.path.join(work_dir, f"GNINA_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
       
        # define docked ligand output directory
        ligand_out_dir = os.path.join(work_dir, "docked_ligands")
        os.makedirs(ligand_out_dir, exist_ok=True)

        # prepare autobox


        # TODO : ligand_path is not used, but should be used in creating the ligand_chain opts!!!!

        # TODO: check if ligand_path is str -> if yes, check if ligand_path in poses.df.columns -> if yes, create list of ligand paths out of column
        # TODO: if ligand path is list --> use list directly
        # TODO: if ligand path is str not in poses.df.columns --> create a list with length of poses, each element is ligand_path (lig_path_ligands)
        # TODO: lig_path_ligands = [lig_path for lig_path in ligand_path]



        # TODO: autobox generation independent of ligand (if none is present), add autobox cmd options to docstrings
        if ligand_path is not None:
            # if ligand_path is a string, check if it is in poses.df columns
            if isinstance(ligand_path, str):
                if ligand_path in poses.df.columns:
                    lig_path_ligands = poses.df[ligand_path].tolist()
                else:
                    # if ligand_path is a string but not in poses.df columns, create a list with length of poses
                    lig_path_ligands = [ligand_path for _ in poses.poses_list()]
            elif isinstance(ligand_path, list):
                # if ligand_path is a list, use it directly
                lig_path_ligands = ligand_path
            else:
                raise ValueError(f"ligand_path must be a string or a list, not {type(ligand_path)}")
                        
            ligand_chain_opts = [f"--autobox_ligand {ligand} --ligand {ligand}" for ligand in lig_path_ligands]
            poses.save_poses(os.path.join(work_dir, 'apo_structures'))
            # TODO: if no ligand is present in the pose (and ligand_path is used as input), copy poses to no_ligand/apo_structures
            
        else:
            poses = self.prepare_ligand_autobox(poses, ligand_chain=ligand_chain, prefix=prefix)
            ligand_chain_opts = [f"--autobox_ligand {ligand} --ligand {ligand}" for ligand in poses.df[f"{prefix}_ligand_location"]]
            
            
        # parse pose_options
        pose_options = self.prep_pose_options(poses, pose_options)

        # combine pose_options and ligand_chain options (priority goes to ligand_chain):
        pose_options = [options_flags_to_string(*parse_generic_options(pose_opt, pose_opt_cols_opt, sep="--"), sep="--") for pose_opt, pose_opt_cols_opt in zip(pose_options, ligand_chain_opts)]
                 
        # write GNINA cmds:
        cmds = [self.write_cmd(pose, output_dir=ligand_out_dir, options=options, pose_options=pose_opts)
        for pose, pose_opts in zip(poses.df[f"{prefix}_noligand_location"].tolist() if ligand_chain else poses.df["poses"].to_list(), pose_options)]

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="gnina",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(work_dir=work_dir, ligand_chain=ligand_chain)
                
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def prepare_ligand_autobox(self, poses: Poses, ligand_chain, prefix):

        chain_remover = ChainRemover(jobstarter=protflow.jobstarters.LocalJobStarter())
        original_work_dir = poses.work_dir
        new_work_dir = os.path.join(poses.work_dir, prefix)
        poses.set_work_dir(new_work_dir)
        poses.df[f"input_location"] = poses.df["poses"]
        poses = chain_remover.run(poses=poses, prefix=f"{prefix}_separate_ligand", preserve_chains=ligand_chain)
        poses.df["poses"] = poses.df[f"input_location"]
        poses = chain_remover.run(poses=poses, prefix=f"{prefix}_noligand", chains=ligand_chain)
        poses.set_work_dir(original_work_dir)
        return poses


    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        """
        Writes the command to run GNINA.py.

        This method constructs the command necessary to run the GNINA script, incorporating various options and parameters. It ensures that the command is correctly formatted and includes all required arguments.

        """
                      
        # parse options
        opts, flags = parse_generic_options(options, pose_options)

        # check if interfering options were set
        forbidden_options = ['r', 'receptor', 'log' , 'o', 'out']
        if opts and any(opt in opts for opt in forbidden_options):
            raise KeyError(f"options and pose_options must not contain any of {forbidden_options}")

        # TODO: add a check if autobox was defined, e.g. via residues (useful if no ligand chain was defined and autobox should be created from cmds)

        # convert to string
        options = options_flags_to_string(opts, flags, sep="--")

        # construct output paths
        out_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.pdb")
        score_dir = os.path.join(os.path.dirname(output_dir), "scores")
        os.makedirs(score_dir, exist_ok=True)
        out_log = os.path.join(score_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.score")

        # write command and return.
        return f"{self.script_path} --receptor {pose_path} --out {out_file} --log {out_log} {options}"

def collect_scores(work_dir:str, ligand_chain:str, ligand_name: str = None) -> pd.DataFrame:


    def extract_gnina_table(file_path):
        description = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r') as file:
            lines = file.readlines()


        # Find where the table header starts (line containing "mode |")
        for i, line in enumerate(lines):
            if line.strip().startswith("mode |"):
                start_idx = i + 3  # Skip header and separator lines
                break

        if not start_idx:
            raise ValueError(f"Table not found in {file_path}")

        # Load table using whitespace as delimiter
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,
            skiprows=start_idx,
            names=["rank", "affinity", "intramol", "CNN_pose", "CNN_affinity"]
        )

        df['input_description'] = description
        df['poses_description'] = [f"{description}_{str(r).zfill(4)}" for r in df['rank']]
        return df

    def add_ligand_to_pose(pose_path:str, ligand_path:str, ligand_chain:str, out_dir:str) -> str:
        out = bpt.add_chain(target=load_structure_from_pdbfile(pose_path), reference=load_structure_from_pdbfile(ligand_path), copy_chain=ligand_chain)
        pose_path = os.path.join(out_dir, description_from_path(ligand_path))
        save_structure_to_pdbfile(out, pose_path)
        return pose_path

    ligand_out_dir = os.path.join(work_dir, "scores")

    # read score files
    scorefiles = glob(os.path.join(ligand_out_dir, "*.score"))
    if not scorefiles:
        raise FileNotFoundError(f"No .score files were found in the output directory of gnina {work_dir}. Gnina might have crashed (check output log), or path might be wrong!")

    scores = pd.concat([extract_gnina_table(scorefile) for scorefile in scorefiles]).reset_index(drop=True)

    # split multimodel ligand pdbs into separate files and collect data
    separate_ligands_dir = os.path.join(work_dir, "separate_ligands")
    os.makedirs(separate_ligands_dir, exist_ok=True)

    ligands = glob(os.path.join(work_dir, "docked_ligands", "*.pdb"))
    ligand_paths = []
    poses_descriptions = []


    ### replace following loop with protflow function "load structure from pdb " or similar --> flag load all models = true
    ### --> biopython 

    for ligand in ligands:
        base = os.path.splitext(os.path.basename(ligand))[0]
        matching_scores = scores[scores["input_description"] == base].reset_index(drop=True)

        # Load all models from the PDB file
        structure = load_structure_from_pdbfile(ligand, all_models=True)
        models = list(structure.get_models())

        if len(models) != len(matching_scores):
            logging.warning(f"Mismatch: {ligand} has {len(models)} models, but {len(matching_scores)} scores")

        for model, (_, row) in zip(models, matching_scores.iterrows()):
            pose_desc = row["poses_description"]
            # Save each model as a separate PDB file
            out_path = os.path.join(separate_ligands_dir, f"{pose_desc}.pdb")
            for chain in model:
                chain.id = ligand_chain
                if ligand_name:
                    for res in chain.get_residues():
                        old_id = res.id
                        res.id = (f"H_{ligand_name}", old_id[1], old_id[2])

            save_structure_to_pdbfile(model, out_path)
            ligand_paths.append(out_path)
            poses_descriptions.append(pose_desc)

    ligand_df = pd.DataFrame({"ligand_path": ligand_paths,"poses_description": poses_descriptions})
    ### hardcoded ligandpath is ok because not poses df

    input_pdb_dir = os.path.join(work_dir, "apo_structures")
    input_pdbs = glob(os.path.join(input_pdb_dir, "*.pdb"))
    input_df = pd.DataFrame({"input_path": input_pdbs, "input_description": [description_from_path(pose) for pose in input_pdbs]})
    scores = scores.merge(ligand_df, on="poses_description")
    scores = scores.merge(input_df, on="input_description")
    # add ligands to
    output_dir = os.path.join(work_dir, "output_pdbs")
    os.makedirs(output_dir, exist_ok=True)
    location = [add_ligand_to_pose(pose_path=row['input_path'], ligand_path=row['ligand_path'], ligand_chain=ligand_chain, out_dir=output_dir) for _, row in scores.iterrows()]
    scores['location'] = location
    scores["description"] = scores["poses_description"]
    return scores
