This is the accompanying code for Hansen, Miao and Xing (2024).
The python code for producing each of the figures in the paper can be found under src.
To produce figure 1, use parameters....

Dependencies:
- numpy
- matplotlib

Each code can easily be modified to use your preferred parameters.

The plots are saved as 'png' files under "plots".
The solutions are saved as 'npz' files under "results", and can be loaded to produce your own graphs.
You can run all 4 plots at once using "src/main.sbatch", which prints the output and error files under "logs".
A simple example is also provided under 'notebooks/example.ipynb'.