###############################################################################
#
# This script will install everything that is needed so that there won't be
# anymore problems while configuration
#
# Author: Shubham Dokania
# E-mail: skd.1810@gmail.com
#
###############################################################################

sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose -y

# Installing Theano
git clone https://github.com/Theano/Theano.git
cd Theano
sudo python setup.py develop

cd ..

git clone https://github.com/fchollet/keras.git
cd keras
sudo python setup.py install
cd ..

sudo pip install Cython
sudo apt-get install libhdf5-dev -y
sudo pip install h5py
