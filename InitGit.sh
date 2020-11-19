# PASS AS ARGUMENTS
# $1 : USERNAME OF YOUR REPO
# $2 ; PASSWORD OF YOUR GITHUB ACCOUNT
# $3 : NAME OF YOUR REPO
################################################################################
#                        Define the url of your repo                           #
################################################################################
myuser=$1
mypsswrd=$2
myrepo=$3
myurl=$"https://${myuser}:${mypsswrd}@github.com/${myuser}/${myrepo}.git"
################################################################################
#                         Initialise and clone repo                            #
################################################################################
# Create directory for clonning
mkdir ~/DATAGEN
cd ~/DATAGEN
# Init git
git init
# Clone repo
git clone $myurl
################################################################################
#                 Save url of your repo as environment variable                #                           #
################################################################################
echo "export REPOURL=${myurl}" >> ~/.bashrc
source ~/.bashrc
echo "alias gitpush='git push $REPOURL --all'" >> ~/.bashrc
source ~/.bashrc
