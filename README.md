# Reinforcement Learning Project

## Part 1: Pong
### Run the code
Install the dependencies:
```
cd Part1
pip install -r requirements.txt
```

**Train the agents (optional)**
For training the REINFORCE model run:
```
python REINFORCE.py
```
This will train and save the model in the specified `SAVE_PATH` directory. Please note this model did not achieve a good performance in our experiments.

For training the DQN model, run the `DQN.ipynb` notebook. 
This will train and evaluate the agent over time, saving the resulting model in the specified `SAVE_PATH` directory.

Our resulting best model (DQN) is saved as `checkpoint_best.zip`.


### Videos
![](Part1/DQN_best.gif)
- The video shows our **best agent** (right), trained with DQN, playing against the default opponent.


## Part 2: Pong Tournament
### Run the code
Install the dependencies:
```
cd PONG-TOURNAMENT
pip install -r requirements.txt
```

**Train the agents (optional)**
The `adversarial_training.ipynb` notebook and `selfPlay.py` file can be run to train models in an adversarial setting (left and right models) or against itself (one model). 
Please note that we have considered these experiments to be "failed attempts" for not performing as expected.

To train our best performing method, first run:
```
python train_right_paddle.py
```
This will train and save the model logs and checkpoints, as well as the final version, in the specified `ROOT_DIR`.

To transfer the model's "flipped weights" into a left model, modify the `SOURCE_MODEL` directory and run:
```
python flip_weights.py
```
This will save the resulting model in the specified `TARGET_MODEL` directory.

For further training for right against left, run:
```
python train_both_sides.py
```
Checkpoints and videos will be saved in the desired `CHECKPOINT_DIR` and `VIDEOS_DIR`.

### Videos
![](PONG-TOURNAMENT/gifs/adversarial.gif)
![](PONG-TOURNAMENT/gifs/selfPlay.gif)
![](PONG-TOURNAMENT/gifs/weightFlipping.gif)
- First video shows the agents competing after being trained with Adversarial Training.
- Second video shows the agents competing after being trained with Self-Play.
- Third video shows the agents competing after being trained with Weight Flipping. This is our **best agent**.

### Use our best left & right agents
The checkpoints are located in `PONG-TOURNAMENT/left_model.zip` and `PONG-TOURNAMENT/right_model.zip`.
Download them and load them as simple PPO models for the tournament.


## Part 3: Donkey Kong
### Run the code
**Install the dependencies:**
```
cd DKONG
pip install -r requirements.txt
```
Edit the `config.yml` file to your liking, setting the `wandb` and paths.

**Start the training (optional):**
```
python main.py
```
This will train and evaluate the agent over time, creating video files that are saved on the `videos` folder.

**Render a video with the best model:**
```
python render_checkpoint_video.py
```
This will save a .mp4 video in `videos/best_model_video_eval/` with our best agent playing and hopefully completing the screen.


### Videos
![](DKONG/gifs/jumping.gif)
![](DKONG/gifs/hammer.gif)
![](DKONG/gifs/good_one.gif)
- First video shows how the agent initially got stuck jumping barrels in the floor level. This was when default barrel reward was used.
- Second video shows how the agent learned to use the hammer, but then ignored the last ladder to climb.
- Third video shows the **final agent** that was able to climb to the top and complete the first screen.
