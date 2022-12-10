# **MGKDB**
### **What is MGKDB?**
MGKDB is a tool for storing, accessing, and learning from a wealth of fusion results.  The main focus of MGKDB will be in the pedestal region of tokamaks, with a large portion of data coming from simulation results.  Currently, GENE is the only code compatible with the database, though compatibility with more codes will be added in the future.

### **What does MGKDB look like?**
MGKDB is structured using **MongoDB**.  MongoDB is a non-relational, sometimes referred to as NoSQL, database structure.  Relational databases, such as the common MySQL database, contains rigidly structure tables to store data.  Non-relational databases, including MongoDB, use a loose format for data collections, based on JSON formatting, meaning that each document (entry) in a collection can hold a unique structure.  MongoDB identifies each document with a unique object ID, given by "_id".  

MGKDB uses **gridfs**, a MongoDB tool for uploading and storing files.  GridFS stores files in chunks, requiring a unique system for storage.  First, the file is broken into chunks.  A file-identifier document is then created in the collection *fs.files*, containing a unique "_id" and a "filepath" that dispalys the original filepath.  The chunks of the files are stored as documents in the collection *fs.chunks*.  All documents for chunks of one file receive unique "_id" values, but contain the same "files_id" value that calls back to the unique file-identifier "_id" value.

An example MGKDB document is shown below.

### **What is MGKDB being developed with?**
MGKDB is being developed in **Python 3.6** using **Anaconda 4.5.9** and **PyMongo** packages.  PyMongo is included in Anaconda releases, but if a custom pacakge list was selected, you may need to install PyMongo.  PyMongo is necessary to connecto a MongoDB via Python.  As of yet, no other versions of Python or Anaconda have been tested.  Any other scritps necessary for running MGKDB are included in this repository.

### **Naming conventions in MGKDB**
* In MGKDB linear runs are labeled 'linear' and nonlinear runs are labeled 'nonlin'.
* MGKDB refers to each folder of runs as a 'run collection'.  Upon submitting a folder to MGKDB, each run in the folder will be given a "run_collection_name" value, set as the folder name.  It may be helpful to include 'linear' or 'nonlin' in the folder name.
	* All runs relating to each other should be kept in a single folder with suffixes to identify them, e.g., each run from a linear scan or continuations from checkpoints
	* Runs in the folder are identified by their suffix, e.g. '_0005' or '_rho.95', shown in MGKDB as "run_suffix."


## **Instructions**
1. Clone the repository.
2. Open mgk_uploader.py.  Near the top you will find the code listed below, sectioned off by ####...####.  This section is where you edit variables to fit your needs.  
	* ** *Required* ** fields are: ```user```, ```output_folder```, and ```multiple_runs``` 
		* ```user```: your name.
		* ```output_folder```: output folder where run(s) are located.  Use '.' if run folders are located in current working directory.
		* ```multiple_runs```: True if uploading multiple sets of runs, False if uploading one set.
	* *Desired* fields are: ```input_heat```, ```confidence```, and ```keywords```  
		* ```input_heat```: in MW, is for simulations based on experiments with known input heat.
		* ```confidence```: will allow MGKDB to guage the checks that went into setting up your simulation.  Low values are for simulations that were quickly thrown together, with little or no prior checks performed.  High values are for simulations for which a wide array of numerical and physical checks were performed.
		* ```keywords```: are very helpful to provide MGKDB with metadata and allows for smart searching through the database.  MGKDB will automatically fill in 'linear' or 'nonlin'.  Please include as many relavent keywords as possible!
3. Run mgk_uploader.py and MGKDB will automate finding parameters and quantities of interest and upload them to the database.  A message will display if your run was uploaded successfully.  If a document in MGKDB contains the same folder name, you will be prompted with whether you would like to remove the original document and reupload the folder or keep the folder already in MGKDB.  This is the current 'update' functionality, though proper update functionality is soon to come!

*Thank you for contributing!*
### **Example mgk_uploader.py file**
```python 
########################################################################

### REQUIRED ###
user = 'C. Blackmon'
output_folder = '.'     ### Set as '.' for current directory ###
multiple_runs = True    ### Automate scanning through a directory of numerous runs ###
#################

### DESIRED ###
if not multiple_runs:
    confidence = '8'     ### '1'-'10', '1': little confidence, '10': well checked ###
else:
    confidence = 'None'  ### Set if  same for all runs, else set as 'None' ###
input_heat = '15MW'      ### Set if input heat is known, else set as 'None' ###
keywords = 'ETG, pedestal, GENE'  ### enter any relevant keywords, i.e., ETG, ITG, pedestal, core ###
###############

########################################################################
```
### **Example MGKDB document**
```json
{
"_id":"5bc7b6e55d25ca2a6cfdce5c",
"user":"C. Blackmon",
"run_collection_name":"78697.51005_sauter_linear_kyscan_r95",
"run_suffix":"_0001",
"keywords":"ETG, pedestal, GENE, linear",
"confidence":"None",
"gamma":0.11,
"omega":-1.115,
"ky":5,
"kx":"None",
"omt":3.4604,
"omn":1.2798,
"scan_id":"None",
"scanlog_id":"5bc7b6e55d25ca2a6cfdce57",
"scaninfo_id":"5bc7b6e55d25ca2a6cfdce5a",
"codemods_id":"5bc7b6e25d25ca2a6cfdcd1e",
"submit_id":"None",
"parameters_id":"5bc7b6e45d25ca2a6cfdce1b",
"eqdisk_id":"None",
"efit_id":"None",
"autopar_id":"None",
"energy_id":"5bc7b6e35d25ca2a6cfdcd5a",
"nrg_id":"5bc7b6e35d25ca2a6cfdcd99",
"omega_id":"5bc7b6e45d25ca2a6cfdcddc",
"s_alpha_id":"None"
}
```
## Future of MGKDB
* Integration into GENE diagnostic tool
* Intuitive interface for user access
* Build simplified and/or quasilinear models of turbulence



	