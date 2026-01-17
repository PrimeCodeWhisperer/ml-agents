Instructions setup:
First download Unity at https://unity.com/download 
Then open GitHub and download the repository
After launch, open it and sign up with your Unity ID or create a new one.
On Unity, click on Add -> Add project from disk and open the file named "Project" in our repository
After that run the Project file on Unity

Open any code editor terminal, in this case VS Code
In the code terminal open the ml-agents file found in the repository
Make sure the python version is Python 3.10.11, you can check this by running python -V

To activate the virtual environment, paste: "python -m venv venv" 
in the terminal and then add venv\Scripts\Activate

Now to run the 3DBall example type this into the code terminal:
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBallRun1

To use TensorBoard and see the results:
First download TensorBoard by: pip install tensorboard
Then after it's installed: tensorboard --logdir results
Then open the link that it returns. 
In TensorBoard open the specific results you want to see.

Create a .env file with inside having:
UNITY_ENV_PATH=[path]
CONFIG_PATH=[path]
TRAINING_DATA_PATH=[path]

To run more than one training without needing to run the previous command, you can use
"python run_training.py [amount]". The results of the training runs will automatically be added to the global csv file (using google sheets) after you created a credentials.json file which contains your credentials to the google sheets.
To run this however you must include the build folder from unity into your top directory. Make sure for MacOS you choose .app and for windows the .exe

To run the learning algorithm on the global csv file run either "python random_forrest" or "python linear_regression.py" after downloading and adding the global csv file of the training run data into the .env file

Dependencies and how to download them:
(Use the pyton -m for safe install)
python -m pip install gspread
python -m pip install oauth2client
python -m pip install pandas
python -m pip install python-dotenv
python -m pip install pyyaml



Description of Data Labels:

RunID: A unique ID for every training run
Scene: The scene name of the game that is being trained
Batch-size: Number of samples being processed in a training step
Algorithm: Name of the algorithm used
Current-time: The date and time of when the training run starts
Mean-reward: The last output of the mean reward from the last set of trainings
Training-time: The time from start of training till the training ends or is stopped
Total-steps: The total amount of steps
Steps-per-second: The amount of steps over seconds
Avg-system-cpu-percent: The average CPU usage percentage from the whole system
Avg-tracked-cpu-percent: The average tracked CPU usage percentage from just the training process only.
Avg-ram-usage: The average ram usage during the training run in MB.
Max-ram-uage: The maximum ram usage during the training in MB.
Num-cores: The number of cpu cores available on the system.
Total-ram: The total of available ram on the system.
Policy-loss: Loss value of how much the policy network is changing during updates
Value-loss: Loss value from the value function.
Learning-rate: The effective learning rate applied.
Operating System: The type of operating system that the training uses.
Hyperparameter-learning-rate: The input learning rate from the hyperparameter.
Cpu model: The type of CPU that the training uses.
Steps-to-threshhold: The amount of steps the training run took to get a mean reward of 100.
Threshold reached: A boolean that describes if the training run ever had a mean reward of 100.

