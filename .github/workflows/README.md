
<img width="1038" height="303" alt="Screenshot 2025-12-04 at 8 48 26 PM" src="https://github.com/user-attachments/assets/a41f64b9-8968-4f14-8e1c-a7639a5771f4" />



# Jetstream2 Runner

The Cyverse Discovery Environment (DE) lacks in powerful GPUs needed for rapid machine learning training. To solve this problem, we have set up a `runner` on github which connects the DE to powerful GPUs VMs hosted on Jetstream2 cloud computer. Two simultaneous processes can occur on the GPU machine. On a single request, the user gets the entire GPU. If another request comes in, it will share all resources relatively equally. If a third request comes in, it is queued until a workflow is available.

<br>
<br>

## How to use the GPU runner in Cyverse DE

<img width="1038" height="266" alt="Screenshot 2025-12-04 at 8 48 50 PM" src="https://github.com/user-attachments/assets/01abd8f1-d20d-488f-b298-21a5980cb03a" />




### 1. Launch a Cyverse App
Users will likely be using the `ESIIL ML Challenge 2025` jupyterlab app in the DE. 

<br>
<br>

### 2. Authenticate with Gihhub
In the terminal of jupyterlab, authenticate to github by typing:

`gh auth login` 

Follow the prompts to connect the DE app with your Github account

<br>
<br>

### 3. Clone the Repository to the Cyverse App
The ML code and the runner are located in the github repository [HDR-SMood-Challenge-sample](https://github.com/jeffgillan/HDR-SMood-Challenge-sample). Because there will be multiple people using the runner, each user needs to have their own branch in the repository. You will be cloning and pushing on your dedicated branch, **not the main branch**. Best practices would be to clone the repo to the directory `~/data-store`.

Clone using the Terminal:

`git clone --branch <branch-name> https://github.com/jeffgillan/HDR-SMood-Challenge-sample.git`

<br>

Clone using the Git Widget:

<img width="566" height="276" alt="Screenshot 2025-12-03 at 8 14 10 AM" src="https://github.com/user-attachments/assets/6cbc9b16-bdd8-4f64-8087-59427c6c2fb5" />


<br>
<br>
<br>

### 4. Edit the Config File
In the root of the repo is a file `training_config.json` which contains the parameters of your training run. 

```
{
  "script_path": "baselines/training/BioClip2/train.py",
  "epochs": 2,
  "batch_size": 32,
  "num_workers": 8,
  "cyverse_output_path": "/iplant/home/<cyverser-username>/hackathon"
}
```

In order to request the GPU runner, you will make some kind of change in the `training_config.json` and push the changes back up to your branch in the github repository. By detecting some kind of change in the json file, github will trigger the workflow on the GPU machine. Once done processing, the model.pth file will be transferred to the "cyverse_output_path" specified in the json file. The GPU machine is a shared resource so if multiple people try to do training runs at the same time, the runs will be queued in the order that github receives them. 

<br>

#### Editing the 'training_config.json' file
Using the terminal, you can open a document editor by typing `nano training_config.json`. Make edits then save the file by pressing crtl+s.

Alternatively, you can edit the json file by right clicking on the file >>> Open With >>> Editor

<img width="479" height="276" alt="Screenshot 2025-12-03 at 8 18 11 AM" src="https://github.com/user-attachments/assets/a5e48e5d-fc32-4387-a8b7-5247a24192cc" />


<br>
<br>

### 5. Create Output Folder in Cyverse Datastore

In the Cyverse Datastore, create a directory in your personal account. This directory will be the path for 'cyverse_output_path'. 

<br>
<br>

### 6. Share the Output Folder 
**!!Important!!** You need to share (write access) your 'cyverse_output_path' directory with the username `jkentg`. This is very important, otherwise the model weights will not be delivered to your output directory. 

<img width="464" height="405" alt="Screenshot 2025-12-03 at 8 26 32 AM" src="https://github.com/user-attachments/assets/a241c32d-4104-4907-a3c2-77a5679a4c5c" />

<img width="405" height="167" alt="Screenshot 2025-12-03 at 8 28 44 AM" src="https://github.com/user-attachments/assets/58b542bd-65b3-4aad-a8c6-d47f37ba0938" />


<br>
<br>

### 7. Push changes to your branch of the github repository

Using the terminal:

`git add training_config.json`

`git commit -m 'runner submission'`

`git push`

<br>
<br>

Using the Git Widgit:

<img width="187" height="489" alt="Screenshot 2025-12-03 at 8 22 03 AM" src="https://github.com/user-attachments/assets/0323f78e-0966-49e3-90cd-e5c2a96da1be" />

<br>
<br>

### 8. Monitoring the Workflow
Once you have submitted a GPU training run (through a change and push of `training_config.json`) you can monitor the training run by going to [Actions Tab in the Github Repository](https://github.com/jeffgillan/HDR-SMood-Challenge-sample/actions). 

<br>
<br>

### 9. Results

The workflow should have deposited the output model weights file `model_<data/time stamp>.pth` into the directory "cyverse_output_path". 

You can transfer the model.pth files back to your jupyterlab working directory by typing:

`cp /iplant/home/<cyverser-username>/hackathon/model_<data/time stamp>.pth ~/data-store/HDR-SMood-Challenge-sample`

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

# Runner Administration

The runner is a self-hosted runner. In the repo, go to _Settings_ >> _Actions_ >> _Runners_ >> _New self-hosted runner_.  It will include instructions on how to download, install, and test run the runner on the host computer. [Additional documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners). 

The customized code for what the runner does and how it does it is specified in [process_training.yml](process_training.yml)

### Start the service
Install the service that will keep running listening all the time, and will restart automatically when the VM is rebooted or unshelved. 

`cd ~/actions-runner`

`sudo ./svc.sh install`

`sudo ./svc.sh start`

`sudo ./svc.sh status`


### Verify in GitHub
Go to: Your repo → Settings → Actions → Runners
You should see your runner with green "Idle" status.


## Host Computer
The runner is hosted on a Jestream2 VM using the ACCESS allocation for HDR. It is a GPU XL machine with an A100 Nvidia GPU, 40GB of GPU RAM, 32vCPU cores, 120GB of system RAM, 280GB of disk storage. 

### Install gocommands to transfer finished data to Cyverse Datastore

```
GOCMD_VER=$(curl -L -s https://raw.githubusercontent.com/cyverse/gocommands/main/VERSION.txt); \
curl -L -s https://github.com/cyverse/gocommands/releases/download/${GOCMD_VER}/gocmd-${GOCMD_VER}-linux-amd64.tar.gz | tar zxvf -
```
Move gocmds to the system Path `sudo mv gocmd /usr/local/bin`

`gocmd init`
 
## Install uv
`curl -LsSf https://astral.sh/uv/install.sh | sh`

