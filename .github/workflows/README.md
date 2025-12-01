# Jetstream2 Runner

The Cyverse Discovery Environment (DE) lacks in powerful GPUs needed for rapid machine learning training. To solve this problem, we have set up a `runner` on github which connects the DE to powerful GPUs VMs hosted on Jetstream2 cloud computer. 

## How to use the GPU runner in Cyverse DE
Users will likely be using the `ESIIL ML Challenge 2025` jupyterlab app in the DE. 


In the terminal of jupyterlab, authenticate to github by typing:

`gh auth login` 

Follow the prompts to connect the DE app with your Github account

<br>
<br>

The ML code and the runner are located in the github repository Clone [HDR-SMood-Challenge-sample](https://github.com/jeffgillan/HDR-SMood-Challenge-sample) Because there will be multiple people using the runner, each user needs to have their own branch in the repository. You will be cloning and pushing on your dedicated branch, not the main branch. 

Best practices would be to clone the repo to the directory `~/data-store`.

`git clone --branch <branch-name> https://github.com/jeffgillan/HDR-SMood-Challenge-sample.git`

Navigate into the cloned repo directory 

`cd HDR-SMood-Challenge-sample`

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

In order to request the GPU runner, you will make some kind of change in the `training_config.json` and push the changes back up to your branch in the github repository. By detecting some kind of change in the json file, github will trigger the workflow on the GPU machine. Once done processing, the model.pth file will be transferred to the "cyverse_output_path" specified in the json file. 

<br>


<br>
<br>
<br>
<br>
<br>

# Runner Administration

### Start the service
`cd ~/actions-runner`

`sudo ./svc.sh start`

### Verify it started
`sudo ./svc.sh status`


### Verify in GitHub
Go to: Your repo → Settings → Actions → Runners
You should see your runner with green "Idle" status.
