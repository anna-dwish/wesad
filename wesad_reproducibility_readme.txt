Following OPTION 1 will just produce the output from the CSV we already ran using generate_wesad_data.py. 
Following OPTION 2 will recreate the CSV with our transformations from the raw wesad data. It will take a LONG time. We anticipate that this process can run from 5-6 HOURS so please run it as soon as possible if you choose this.

OPTION 1: RESULTS REPRODUCIBILITY

1. ssh netID@dcc-slogin.oit.duke.edu
2. cd /hpc/group/sta440-f20/netID
3. Upload the following files into the same folder: quick_submit_wesad_models.sh, generate_wesad_models.R
4. srun --pty bash -i
5. sbatch quick_submit_wesad_models.sh
6. Check the wesad_models.txt file for the classification accuracy for our combined data model, the classification accuracy for our wrist-only model, and the quantification of heterogeneity across individuals.
7. Please email ard51@duke.edu if you encounter any issues, we will be quick to respond!

OPTION 2: DATA & RESULTS REPRODUCIBILITY

1. ssh netID@dcc-slogin.oit.duke.edu
2. cd /hpc/group/sta440-f20/netID
3. Upload the following files into the same folder: submit_wesad_data.sh, submit_wesad_models.sh, generate_wesad_data.py, & generate_wesad_models.R
4. srun --pty bash -i
5. sbatch submit_wesad_data.sh (see **TIPS** for checking status of this job & receiving email updates)
6. Verify merged_wesad.csv is in your folder.
7. sbatch submit_wesad_models.sh
8. Check the wesad_models.txt file for the classification accuracy for our combined data model, the classification accuracy for our wrist-only model, and the quantification of heterogeneity across individuals.
9. Please email ard51@duke.edu if you encounter any issues, we will be quick to respond!

**TIPS**
1. You can reference the status of the csv submission in the status_csv.txt file in the same directory you run our bash script in. It will finish after the last subject, subject 17, has finished.
2. In both the the create_wesad_csv.sh and create_wesad_models.sh bash script, please note that you may modify it to send you an email when the job is complete by modifying the #SBATCH --mail-user=ard51@duke.edu line to send to your NetID email instead.