"""
Obrms Module
============

This module provides the functionality to integrate Obrms (Open Babel RMSD calculation tool) within the ProtFlow framework. It offers tools to run Obrms, handle its inputs and outputs, and process the resulting RMSD data in a structured and automated manner.
Use this module instead of the RMSD metric if the compared structures are not named in identically.

Detailed Description
--------------------
The `Obrms` class encapsulates the functionality necessary to execute Obrms runs for calculating Root Mean Square Deviation (RMSD) between molecular structures. 
It manages the configuration of paths to the Obrms executable, sets up the environment, and handles the execution of RMSD calculation processes. 
It also includes methods for collecting and processing output data, ensuring that the results are organized and accessible for further analysis within the ProtFlow ecosystem.

The module is designed to streamline the integration of Obrms into larger computational workflows. It supports the automatic setup of job parameters, 
execution of Obrms commands, and parsing of output files into a structured DataFrame format. This facilitates subsequent data analysis and visualization steps for structural comparison tasks.

Usage
-----
To use this module, create an instance of the `Obrms` class and invoke its `run` method with appropriate parameters. The module will handle the configuration, execution, and result collection processes. Detailed control over the RMSD calculation process is provided through various parameters, allowing for customized runs tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `Obrms` class within a ProtFlow pipeline:

.. code-block:: python

    from protflow.poses import Poses
    from protflow.jobstarters import JobStarter
    from obrms import Obrms

    # Create instances of necessary classes
    poses = Poses()
    jobstarter = JobStarter()

    # Initialize the Obrms class
    obrms = Obrms()

    # Run the RMSD calculation process
    results = obrms.run(
        poses=poses,
        prefix="rmsd_experiment",
        ref_path="/path/to/reference.pdb",
        test_column="poses",
        options="--additional_option",
        jobstarter=jobstarter,
        overwrite=True
    )

    # Access and process the results
    print(results)

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as missing output files, parsing errors in Obrms output, and the need to overwrite previous results. 
    It ensures robust error handling and logging for easier debugging and verification of the RMSD calculation process.
    - Integration: The module seamlessly integrates with other components of the ProtFlow framework, 
    leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to incorporate RMSD calculations into their protein structure analysis workflows. 
Use it instead of the RMSD metric if the atom naming is not identical between the compared structures.

Notes
-----
This module is part of the ProtFlow package and is designed to work in tandem with other components of the package, 
especially those related to job management in HPC environments. The Obrms executable must be available in the system PATH or properly configured.

Authors
-------
David Stoll

Version
-------
0.1.0    
"""

import os
import logging
import pandas as pd
from protflow.poses import Poses
from protflow import require_config, load_config_path, runners
from protflow.runners import Runner, RunnerOutput
from protflow.jobstarters import JobStarter

class Obrms(Runner):
    """
    Obrms Class
    ===========

    The `Obrms` class is a specialized class designed to facilitate the execution of Obrms (Open Babel RMSD calculation) within the ProtFlow framework. It extends the `Runner` class and incorporates specific methods to handle the setup, execution, and data collection associated with RMSD calculation processes.

    Detailed Description
    --------------------
    The `Obrms` class manages all aspects of running Obrms calculations for determining Root Mean Square Deviation between molecular structures. It handles the configuration of the Obrms executable, prepares the environment for RMSD calculations, and executes the comparison commands. Additionally, it collects and processes the output data, organizing it into a structured format for further analysis.

    Key functionalities include:
        - Setting up paths to the Obrms executable (assumes it's available in PATH).
        - Configuring job starter options for parallel execution of multiple comparisons.
        - Handling the execution of Obrms commands with reference and test structure pairs.
        - Collecting and processing RMSD output data into a pandas DataFrame.
        - Managing output directories and file organization for structured results.

    Returns
    -------
    An instance of the `Obrms` class, configured to run RMSD calculations and handle outputs efficiently.

    Raises
    ------
        FileNotFoundError: If required files or directories are not found during the execution process.
        ValueError: If invalid arguments are provided to the methods or if no RMSD values are returned.
        RuntimeError: If expected output files are missing after job completion.

    Examples
    --------
    Here is an example of how to initialize and use the `Obrms` class:

    .. code-block:: python

        from protflow.poses import Poses
        from protflow.jobstarters import JobStarter
        from obrms import Obrms

        # Create instances of necessary classes
        poses = Poses()
        jobstarter = JobStarter()

        # Initialize the Obrms class
        obrms = Obrms()

        # Run the RMSD calculation process
        results = obrms.run(
            poses=poses,
            prefix="rmsd_analysis",
            ref_path="/path/to/reference_structure.pdb",
            test_column="poses",
            options="--additional_options",
            jobstarter=jobstarter,
            overwrite=True
        )

        # Access and process the results
        print(results)

    Further Details
    ---------------
        - Edge Cases: The class includes handling for various edge cases, such as missing output files, parsing errors in Obrms output, and the need to overwrite previous results.
        - Customization: The class provides extensive customization options through its parameters, allowing users to tailor the RMSD calculation process to their specific needs.
        - Integration: Seamlessly integrates with other ProtFlow components, leveraging shared configurations and data structures for a unified workflow.

    The Obrms class is intended for researchers and developers who need to perform structural comparisons and RMSD calculations as part of their protein analysis workflows. It simplifies the process, allowing users to focus on analyzing results and advancing their research.
    """
    
    def __init__(self, script_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter|None = None) -> None:
        """
        Initialize the Obrms class with necessary configurations.

        This method sets up the Obrms class, configuring paths to the Obrms executable and setting up the environment for executing RMSD calculation processes.

        Parameters:
            obrms_path (str, optional): The path to the Obrms executable. Defaults to the value specified in `protflow.config.OBRMS_PATH`.
            pre_cmd (str, optional): Command to prepend before running Obrms (e.g., module loading commands). Defaults to the value specified in `protflow.config.OBRMS_PRE_CMD`.
            jobstarter (JobStarter, optional): An instance of the JobStarter class, which manages job execution. Defaults to None.

        Raises:
            ValueError: If no path is set for the Obrms executable.

        Examples:
            Here is an example of how to initialize the `Obrms` class:

            .. code-block:: python

                from protflow.jobstarters import JobStarter
                from obrms import Obrms

                # Initialize the Obrms class with default configurations
                obrms = Obrms()

                # Initialize the Obrms class with a custom path and jobstarter
                custom_obrms_path = "/path/to/custom/obrms"
                jobstarter = JobStarter()
                obrms = Obrms(obrms_path=custom_obrms_path, jobstarter=jobstarter)

        Further Details:
            - **Configuration:** This method sets the path for the Obrms executable and any pre-command needed. If these are not correctly set in the configuration, a ValueError is raised.
            - **Initialization:** The method initializes necessary attributes, including the executable path, pre-command, jobstarter, and other configurations needed for running RMSD calculations.

        This method prepares the Obrms class for running RMSD calculations, ensuring that all necessary configurations and paths are correctly set up.
        """
        # setup config
        config = require_config()
        self.script_path = script_path or load_config_path(config, "OBRMS_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "OBRMS_PRE_CMD", is_pre_cmd=True)

        # runner setup
        self.name = "obrms"
        self.index_layers = 1
        self.jobstarter = jobstarter
        

    def __str__(self):
        return "obrms"

    def run(self, poses: Poses, prefix: str, ref_path: str, test_column: str, options: str = "", overwrite: bool = False, jobstarter: JobStarter = None) -> Poses:
        work_dir = os.path.join(poses.work_dir, prefix, "obrms")
        os.makedirs(work_dir, exist_ok=True)
        logging.info(f"Running obrms in {work_dir} on {len(poses.df)} poses")

        scorefile = os.path.join(work_dir, f"{prefix}_obrms.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile, overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()

        # Build output paths and commands
        output_txts = []
        cmds = []
        for _, row in poses.df.iterrows():
            test_path = row[test_column]
            description = row["poses_description"]
            out_path = os.path.join(work_dir, f"{description}.obrms")
            cmd = f"{self.script_path} -f {ref_path} {test_path} > {out_path}"
            cmds.append(cmd)
            output_txts.append((description, test_path, out_path))

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Submit jobs via JobStarter
        logging.info(f"Submitting {len(cmds)} obrms jobs via JobStarter")
        jobstarter.start(cmds=cmds, jobname="obrms", output_path=work_dir)

        # Parse results
        results = []
        for description, test_path, out_path in output_txts:
            if not os.path.exists(out_path):
                raise RuntimeError(f"Missing obrms output: {out_path}")
            with open(out_path) as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("RMSD"):
                        try:
                            rmsd = float(line.strip().split()[-1])
                            results.append({
                                "description": description,
                                "input_poses": test_path,
                                "rmsd": rmsd
                            })
                        except Exception as e:
                            logging.warning(f"Failed to parse line in {out_path}: {line.strip()} ({e})")

        if not results:
            raise ValueError("Obrms runner did not return RMSD values.")

        df = pd.DataFrame(results)
        df["location"] = df["input_poses"]
        self.save_runner_scorefile(df, scorefile)
        return RunnerOutput(poses=poses, results=df, prefix=prefix).return_poses()
