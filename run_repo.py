"""Sends "simulation.py" to spinnaker server, then retrieves the job results.

It is a revisitation of the script given in form of jupyter notebook in EBRAINS spinnaker lab.
"""
import nmpi
from rich import print as pprint
import argparse
import os 
import time

# Replace None with your username, password and collab_id
username = "djanloo"
password = "ebra_in_758"
collab_id = "nmc-test-djanloo"

pprint("[blue]Attempting connection[/blue]...")
if username is None or password is None or collab_id is None:
    pprint("[red]username/password/collab_id not set yet[/red]")
    exit()

# Connection to server
client = nmpi.Client(
                    username=username, 
                    password=password,                     
                    )
pprint("[green]Connection done[/green]")

pprint(f"Authorization endpoint: [green]{client.authorization_endpoint}[/green]/login")
pprint("It may be necessary to grant access visiting this URL by browser [blue]once[/blue]")
pprint(f"Using the repository [green] {collab_id} [/green] for quotas.\nStarting the job at {time.ctime()}")

job = client.submit_job(source="https://github.com/djanloo/sPyNNaker.git",
                        platform=nmpi.SPINNAKER,
                        collab_id=collab_id,
                        # config=dict(spynnaker_version="1!7.1.0"), 
                        # #NOTE: specifying the spynnker version does not work 
                        # because the remote system is outdated
                        # and no more compatible with version 1!6.0.1
                        command="setup_and_run.py",
                        wait=True)
print(job["log"])

filenames = client.download_data(job, local_dir=os.getcwd())
pprint("Fetched the files",filenames)
image_filenames = [name for name in filenames if name.endswith(".png")]
pprint(image_filenames)

os.system(r'echo -e \\a') # Makes a beep