# LG Automotive Simulator RL Project
This project creates a wrapper around LG Sillicon Valley Lab Automotive Simulator (LGSVL SIM) to make it an OpenAI compatible environment where you can test your DRL algortihms. In fact, OpenAI baselines are included in this project as a submodule (you will have to clone it yourself - see bellow) and some of their latest agents where used to park a vehicle.

## Installation and Setup

### 1) Running the LG-SIM Server
This project uses the LG automotive simulator as the environment (Openai compatible). The easiest way to install LG SIM is to follow the instructions on their Github: https://github.com/lgsvl/simulator

For this particular project I used a locally built executable (based on their 2019-04 release with a python API) with parkinglot.

### 2) Running the client (training code, benchmark code)
My code requires:

* Python 3
* OpenAI Gym (https://github.com/openai/gym)
* OpenAI Baselines (https://github.com/openai/baselines)

### run: 
```python 
python setup.py install
```
Make sure everything went ok and run

```python 
python baselines_run.py
```

#### NOTE: Soon I will pack all in a Dockerfile to install all these dependencies in a clean and isolated manner. 

## Arguments and Config Files
**baselines_run.py** script uses a clear defined section to setup all the main OpenAI Baselines arguments.
```python
    ##########################################################
    #            DEFINE YOUR "BASELINES" PARAMETERS HERE 
    ##########################################################
    env =  'LG-SIM-ENV-v0'
    alg = 'her'
    network = 'default'
    num_timesteps = '1e4'
    save_folder = models_folder + '/' + env +'/'+ alg + '/' + network 
    save_file = save_folder + '/' + str(currentDT)
    logger_path = save_file + '_log'
    load_path = save_folder +'/'+ '20190419-113140' 
    ##########################################################
        
```
 **settings.py** is used to define the folder where all the training logs and final/trained agent weights will reside.
 and a configuration file. 

Hyperparameter Tuning
Check OpenAI Baselines for more information on how to change their default hyperparameters. This will let you specify a set of hyperparameters to test different from default.

## Benchmark Results
By default the training results will be placed on
***/run/models/"algorithm"/"network"***

### A2C
To reproduce my results, run a LG SIM server and on ***baselines_run.py*** use the following setup:
```python
    ##########################################################
    #            DEFINE YOUR "BASELINES" PARAMETERS HERE 
    ##########################################################
    env =  'LG-SIM-ENV-v0'
    alg = 'a2c'
    network = 'mlp'
    num_timesteps = '1e5'
    save_folder = models_folder + '/' + env +'/'+ alg + '/' + network 
    save_file = save_folder + '/' + str(currentDT)
    logger_path = save_file + '_log'
    load_path = save_folder +'/'+ '20190419-113140' 
    ##########################################################
        
```
#### NOTE: 
1) If you run it for the first time, make sure to comment out the loading path argument (see below)
2) If you want to use the "default" network (mlp) you can comment out the "--network" argument (see below)
3) logger_path will only work with some of the OpenAI baseline algorithms. If you chose one algortihm and it throws you an error regarding the "logger" just comment out the argument (see below)
4) For more information about what these arguments do and if there are more arguments that you can add to "tune" your agent, please refer to OpenAI baselines README.MD files for the algorithm/agent you are using.

```python
DEFAULT_ARGUMENTS = [
        '--env=' + env,
        '--alg=' + alg,
    #    '--network=' + network,
        '--num_timesteps=' + num_timesteps,    
    #    '--num_env=0',
        '--save_path=' + save_file,
    #    '--load_path=' + load_path,
    #    '--logger_path=' + logger_path,
        '--play'
    ]
```

### ACKTR
To reproduce my results, run a LG SIM server and follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'acktr'
```

### PPO2
To reproduce my results, run a LG SIM server and follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'ppo2'
```
### TRPO_MPI
To reproduce my results, run a LG SIM server and follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'trpo_mpi'
```

### ON-POLICY Hindsight Experience Replay (HER) over DDPG
To reproduce my results, run a LG SIM server and follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'her'
```
#### NOTE:
1) To know which algorithms you can use, simply take a look at the /open_ai_baselines/baselines folder.
2) Some algorithms will fail since are NOT suited for this problem. For example, DDPG was implemented for discrete actions spaces and will not take a "BOX" as an action space. Try the one you are interested in and find out why it will not run. Sometimes it will take just a few changes and other times, as metioned before, it might not even be meant for this type of problem. 