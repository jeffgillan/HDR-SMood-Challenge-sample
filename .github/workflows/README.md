# Jetstream2 Runner

The Cyverse Discovery Environment (DE) lacks in powerful GPUs needed for rapid machine learning training. To solve this problem, we have set up a `runner` on github which connects the DE to powerful GPUs VMs hosted on Jetstream2 cloud computer. 

<br>
<br>

## How to use the GPU runner in Cyverse DE
Users will likely be using the `ESIIL ML Challenge 2025` jupyterlab app in the DE. 

<br>
<br>

In the terminal of jupyterlab, authenticate to github by typing:

`gh auth login` 

Follow the prompts to connect the DE app with your Github account

<br>
<br>

The ML code and the runner are located in the github repository [HDR-SMood-Challenge-sample](https://github.com/jeffgillan/HDR-SMood-Challenge-sample). Because there will be multiple people using the runner, each user needs to have their own branch in the repository. You will be cloning and pushing on your dedicated branch, **not the main branch**. Best practices would be to clone the repo to the directory `~/data-store`.

Clone using the Terminal:

`git clone --branch <branch-name> https://github.com/jeffgillan/HDR-SMood-Challenge-sample.git`

<br>

Clone using the Git Widget:



<br>
<br>

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

Alternatively, you can edit the json file by right clicking on the file .......

<br>

#### Notes on cyverse_output_path

In the Cyverse Datastore, create a directory in your personal account. This directory will be the path for 'cyverse_output_path'. Then you need to share (write access) your 'cyverse_output_path' directory with the username `jkentg`. This is very important, otherwise the model weights will not be delivered to your output directory. 

<br>
<br>

#### Push changes to your branch of the github repository

Using the terminal:

`git add training_config.json`

`git commit -m 'runner submission'`

`git push`

<br>
<br>

Using the Git Widgit:



#### Monitoring the Workflow
Once you have submitted a GPU training run (through a change and push of `training_config.json`) you can monitor the training run by going to [Actions Tab in the Github Repository](https://github.com/jeffgillan/HDR-SMood-Challenge-sample/actions). 

<br>
<br>
<br>
<br>
<br>

# Runner Administration

The runner is a self-hosted runner. In the repo, go to _Settings_ >> _Actions_ >> _Runners_ >> _New self-hosted runner_.  It will include instructions on how to download, install, and test run the runner on the host computer. [Additional documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners). 

The customized code for what the runner does and how it does it is specified in [process_training.yml](.github/workflows/process_training.yml)

### Start the service
`cd ~/actions-runner`

`sudo ./svc.sh start`

### Verify it started
`sudo ./svc.sh status`


### Verify in GitHub
Go to: Your repo → Settings → Actions → Runners
You should see your runner with green "Idle" status.
