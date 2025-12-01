# Jetstream2 Runner

The Cyverse Discovery Environment (DE) lacks in powerful GPUs needed for rapid machine learning training. To solve this problem, we have set up a `runner` on github which connects the DE to powerful GPUs VMs hosted on Jetstream2 cloud computer. 

## How to use the GPU runner in Cyverse DE
Users will likely be using the `ESIIL ML Challenge 2025` jupyterlab app in the DE. 


In the terminal of jupyterlab, authenticate to github by typing:

`gh auth login` 

Follow the prompts to connect the DE app with your Github account

<br>
<br>

clone the [HDR-SMood-Challenge-sample](https://github.com/jeffgillan/HDR-SMood-Challenge-sample) git repository to the jupyterlab app. 

`git clone https://github.com/jeffgillan/HDR-SMood-Challenge-sample.git`


`cd ~/actions-runner`

# Runner Administration

### Start the service
`sudo ./svc.sh start`

### Verify it started
`sudo ./svc.sh status`


### Verify in GitHub
Go to: Your repo → Settings → Actions → Runners
You should see your runner with green "Idle" status.
