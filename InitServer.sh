# PASS AS ARGUMENTS
# $1 : YOUR PHONE NUMBER FOR TWILIO CLIENT
# $2 : YOUR TWILIO ACCOUNT SID
# $3 : YOUR TWILIO AUTHENTICATION TOKEN
################################################################################
#                              Install anaconda                                #
################################################################################
sudo apt-get update
sudo apt-get install tmux
sudo apt-get install build-essential
sudo apt-get install git
sudo apt install make
sudo apt install vim
sudo apt-get install curl
cd /tmp
# Download appropriate version of anaconda
curl –O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# I go irresponsibly not checking the .sh file
bash Anaconda3-2020.11-Linux-x86_64.sh
################################################################################
#                         Create env vars for twilio                           #
################################################################################
phone=$1
twilio_account_sdi=$2
twilio_auth_tok=$3
# Initialise environment variables
setmyphone=$"export MYPHONE=${phone}"
settwacc=$"export TWILIO_ACCOUNT_SID=${twilio_account_sdi}"
settwauth=$"export TWILIO_AUTH_TOKEN=${twilio_auth_tok}"
echo "$setmyphone" >> ~/.bashrc
echo "$settwacc" >> ~/.bashrc
echo "$settwauth" >> ~/.bashrc
# Activate environment
source ~/.bashrc
################################################################################
#                          Create local environment                            #
################################################################################
conda create --name Datagen
conda deactivate base
conda activate Datagen
pip install twilio
cd ~/DATAGEN
