This is a project for learning about Keras and it's usefulness.


Training
===========================================
To start the training of the model, first of all run the shellscript
install_everything.sh to configure the dependencies.
then you can run the python script 'character_lstm.py'.


Running in Background
==========================================
After configuration using install_everything.sh, you can run the shell script
run_in_back.sh to start running the training process in background. It will
save the trained model and the weights.

Testing the pre-trained model
=========================================
To test the pre-trained, open the ipython console and import * from
test_and_load.py. this will load the model and learned weights of the network.
then to test stirng functions, run the function run_main() in the console to
see the output with carying diversities.
