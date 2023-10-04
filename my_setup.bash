#!/bin/bash

#AUTHOR: djanloo
#VERSION: 2.0

# DESCRIPTION: this is an attempt to reproduce the << super easy >> developer installation chain 
# described at http://spinnakermanchester.github.io/development/devenv.html
#
# My necessity is to apply minor (really, really small) edits on the C code of neuron dynamics

# Installation options
UPGRADE_JAVA=false
BUILD_JAVASPINNAKER=false
DELETE_OLD=true
CLONE_STABLES=true
BUILD_EDITED=true # Whether to build the default sPyNNaker or the edited one

# Versions (this is a desperate attempt to match the remote setup)

# SPALLOC_V='d7bd5cd758438bab3c3f74f21aa6da073cd47033'
# PACMAN_V='437160765f1ff2ee579a82b6ec79a88789e3d2b4'
# SPINNFRONTENDCOMMON_V='1f3e229c10667d6919526d570020064810a8fa5f'
# SPINNMACHINE_V='bc2225e2ae37bf25d38bc1799b75f2ac84e86bc8'
# SPINNMAN_V='5afe0e3cec9ee50adc44cf2f3e322e6eab9508ef'
# SPINNUTILS_V='2ff6b301ba601a0838d1cdb61dc43f2cba7671ad'


SPALLOC_V='master'
PACMAN_V='master'
SPINNFRONTENDCOMMON_V='master'
SPINNMACHINE_V='master'
SPINNMAN_V='master'
SPINNUTILS_V='master'


gitclone () {
    REPO=$1
    VERSION=$2
    if [ -d "$REPO" ]; then
        pecho Repo $REPO not cloned since it already exists
    else
        pecho cloning $REPO
        git clone https://github.com/SpiNNakerManchester/$REPO.git --branch master --single-branch &> /tmp/last_gitclone.txt 
        cd $REPO
        pecho checking out $REPO to version $VERSION
        git checkout $VERSION  
    fi
    LAST_ERROR=$?
    if [ $LAST_ERROR -ne 0 ]; then
            cat /tmp/last_gitclone.txt
            echo "$LAST_ERROR" > $INITDIR/last_error.txt
            exit_env $LAST_ERROR
        fi
    cd ..
}

##################### UTILS #####################
# Echo with current position
pecho() {
    echo -e "\e[34m$(pwd)\e[0m>> " $@
}

# Warning with current position
pwarn() {
    echo -e  "\e[34m$(pwd)\e[0m>>  \e[31mWARNING\e[0m:" $@
}

# Prints a line
echoline() {
    TITLE=$@
    printf '%.0s-' {1..50};printf "\e[41m$TITLE $(date +"%H:%M")\e[0m";printf '%.0s-' {1..50};printf '\n'
}

exit_env () {
    pecho $?
    return
    # exit $?
}


## COPIED FROM setup.bash
dosetupinstall() {
    DIR=$1
    if [ -f "$DIR/setup.py" ]; then
        pecho "Setting up $DIR"
        # (cd $DIR; python setup.py install > /tmp/last_setup.tmp 2>&1 )
        (cd $DIR; pip install .  --no-dependencies > /tmp/last_setup.tmp 2>&1 )

        LAST_ERROR=$?
        if [ $LAST_ERROR -ne 0 ]; then
            cat /tmp/last_setup.tmp
            echo "$LAST_ERROR" > $INITDIR/last_error.txt
            exit_env $LAST_ERROR
        fi
    else
        pecho "Skipping setting up $DIR as no setup.py found"
        ls -al $DIR
    fi
}
## COPIED FROM setup.bash
domvn() {
    DIR=$1
    pecho "Building $DIR"
    ${MAVEN} -f $DIR package -Dmaven.test.skip=true &> "$MAKE_LOG_FOLDER/$(basename $DIR).java.txt" 2>&1
    LAST_ERROR=$?
    if [ $LAST_ERROR -ne 0 ]; then
        cat "$MAKE_LOG_FOLDER/$(basename $DIR).java.txt"
        echo "$LAST_ERROR" > $INITDIR/last_error.txt
        exit_env $LAST_ERROR
    fi
}

## COPIED FROM setup.bash
domake() {
    DIR=$1
    GOAL=$2
    pecho "Building $GOAL in $DIR"
    make -C $DIR $GOAL &> "$MAKE_LOG_FOLDER/$(basename $DIR).$GOAL.txt" #/tmp/last_make.tmp 2>&1
    LAST_ERROR=$?
    if [ $LAST_ERROR -ne 0 ]; then
        cat  "$MAKE_LOG_FOLDER/$(basename $DIR).$GOAL.txt"
        echo "$LAST_ERROR" > $INITDIR/last_error.txt
        exit_env $LAST_ERROR
    fi
}

clean_downloads() {
    rm spinnaker_tools -R -f
    rm PyNN -R -f
    rm SpiNNutils -R -f
    rm SpiNNUtils -R -f
    rm SpiNNMachine -R -f
    rm PACMAN -R -f
    rm SpiNNMan -R -f
    rm spalloc -R -f
    rm spinn_common -R -f
    rm spalloc_server -R -f
    rm SpiNNFrontEndCommon -R -f
    # rm sPyNNaker -R -f
    rm sPyNNaker8 -R -f
    rm JavaSpiNNaker -R -f
}

upgrade_java() {
    # Upgrades Java to a stable version (tested: 13)
    # Upgrades Maven to a stable version (tested 3.6.3)
    mkdir java_stuff
    cd java_stuff

    # Downloads java and maven
    wget https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.6.3/apache-maven-3.6.3-bin.tar.gz &> /tmp/download_maven.tmp
    wget https://download.java.net/java/GA/jdk13.0.1/cec27d702aa74d5a8630c65ae61e4305/9/GPL/openjdk-13.0.1_linux-x64_bin.tar.gz &> /tmp/download_java.tmp

    wait
    pecho after downloading maven and java:
    ls -al
    echo

    # Extracts compressed files
    tar -xvf apache-maven-3.6.3-bin.tar.gz &> /tmp/extract_maven.tmp
    tar -xvf openjdk-13.0.1_linux-x64_bin.tar.gz &> /tmp/extract_java.tmp

    # Sets environment: JAVA
    JAVA_HOME="${PWD}/jdk-13.0.1"
    PATH="$JAVA_HOME/bin:$PATH"
    export PATH

    # Sets environment: MAVEN
    unset M2_HOME
    MAVEN_HOME="${PWD}/apache-maven-3.6.3"
    PATH="$MAVEN_HOME/bin:$PATH"
    export PATH

    MAVEN="$MAVEN_HOME/bin/mvn"
    cd ..
}


##################### Some preliminaries #####################
echoline Started installation script by djanloo

export INITDIR=$(pwd)

# Edit where to place logs
# Avoiding this can cause errors
mkdir LOGS
chmod 777 LOGS

export C_LOGS_DICT=$(pwd)/LOGS/logs.sqlite3
touch $C_LOGS_DICT
chmod 777 $C_LOGS_DICT

export MAKE_LOG_FOLDER=${PWD}/LOGS

pecho Retrieving configuration files
mkdir ../config_files
cd $VIRTUAL_ENV
cp "$(<$INITDIR/config_files_list.txt)" $INITDIR/../config_files
cd -

cd ..
pecho Remote spynnaker version: $(python -c "import spynnaker; print(spynnaker.__version__)")
cd -

pecho Result of pip freeze:
pip freeze
pip freeze > pipfreeze.txt

pecho Exporting environment variables to ${INITDIR}/env_var.txt
printenv > ${INITDIR}/env_var.txt

cd $INITDIR
pecho Installing requirements:
pip install -r requirements.txt
cd -

# pecho Installing "manual" requirements
# pip install "numpy>=1.13,<1.9999" "scipy>=0.16.0" matplotlib
# pip install "appdirs>=1.4.2,<2.0.0" "pylru>=1" lxml jsonschema sortedcollections futures pytz tzlocal "requests>=2.4.1"
# pip install csa "quantities>=0.12.1" "lazyarray>=0.2.9,<=0.4.0" "neo>=0.5.2,< 0.7.0"


##################### SETUP #####################

# Move where the repos are
cd $VIRTUAL_ENV
mkdir AUX_INSTALL
export AUX_INSTALL_FOLDER=$VIRTUAL_ENV/AUX_INSTALL
cd AUX_INSTALL

# If it doesn't work with the default
# then what am I trying to do?
if [ "$BUILD_EDITED" = true ] ; then
    # Delete the current version
    pecho Removing default sPyNNaker package
    rm sPyNNaker -R -f
    # Substitute my version
    pecho Copying modified sPyNNaker package
    cp -R $INITDIR ./
    # Rename to prevent from name mismatch
    mv sPyNNaker/spynnaker-custom sPyNNaker/spynnaker
else
    rm -rf sPyNNaker
    gitclone sPyNNaker
    pwarn Default sPyNNaker package will be used in the installation
fi

## Upgrade pip and other packages (is it necessary?)
# pip install packaging
# pip -V

## Not needed anymore due to 17-JUL-23 upgrade
# pip install --upgrade pip setuptools wheel

### CLONING REPOS
# These are mine
# git clone https://github.com/djanloo/SpiNNUtils.git
##################### CLONING ###################


if [ "$DELETE_OLD" = true ] ; then
    pwarn DELETE_OLD is set to true: deleting all folders..
    clean_downloads
    pecho now listing files:
    ls -al
fi


echoline CLONING
if [ "$CLONE_STABLES" = true ] ; then
    gitclone spinnaker_tools &
    gitclone spinn_common &
    gitclone SpiNNutils $SPINNUTILS_V &
    # git clone https://github.com/djanloo/SpiNNUtils.git
    gitclone SpiNNMan $SPINNMAN_V &
    gitclone PACMAN $PACMAN_V &
    gitclone spalloc $SPALLOC_V &
    gitclone SpiNNFrontEndCommon $SPINNFRONTENDCOMMON_V &
    gitclone SpiNNMachine $SPINNMACHINE_V &
    gitclone JavaSpiNNaker &
else
    pwarn Repos were not upgraded
fi
wait

pecho Listing files:
ls -al
echo
echoline INSTALL NON-COMPILED STUFF

# dosetupinstall SpiNNutils
dosetupinstall SpiNNUtils

wait
dosetupinstall SpiNNMachine
wait
dosetupinstall PACMAN

echo

# Build the C Code
echoline COMPILATION

export SPINN_DIRS=$(pwd)/spinnaker_tools
export NEURAL_MODELLING_DIRS=$(pwd)/sPyNNaker/neural_modelling

pecho SPINN_DIRS was set to $SPINN_DIRS
pecho NEURAL_MODELLING_DIRS was set to $NEURAL_MODELLING_DIRS

domake $SPINN_DIRS clean
domake $SPINN_DIRS

pecho after compiling SPINN_DIRS was set to $SPINN_DIRS
pecho after compiling NEURAL_MODELLING_DIRS was set to $NEURAL_MODELLING_DIRS
echo

# Install spinnaker_tools
pecho setting spinnaker_tools
cd spinnaker_tools
source setup
make
cd ..
pecho done setting spinnaker_tools
export PATH=$SPINN_DIRS/include:$PATH
pecho SPINN_DIRS was set to $SPINN_DIRS
pecho now PATH is $PATH
echo
domake spinn_common clean
domake spinn_common install-clean
domake spinn_common
domake spinn_common install

if [ -e SpiNNMan/c_models ]; then
    domake SpiNNMan/c_models clean
    domake SpiNNMan/c_models
fi

wait

domake SpiNNFrontEndCommon/c_common/front_end_common_lib install-clean
domake SpiNNFrontEndCommon/c_common/ clean
domake SpiNNFrontEndCommon/c_common/
domake SpiNNFrontEndCommon/c_common/ install


if [ "$UPGRADE_JAVA" = true ] ; then
    pecho Updating JAVA...
    # Check java and maven versions
    pecho BEFORE upgrade of java and maven:
    java -version
    mvn -version
    upgrade_java
    pecho AFTER upgrade of java and maven:
    java -version
    ${MAVEN} -version
else
    pwarn Java and Maven have not been upgraded
    pecho Setting MAVEN=mvn
    MAVEN="mvn"
fi

echoline INSTALLATION OF COMPILED STUFF

 # Setup packages to make sure the right libraries are installed
dosetupinstall SpiNNMan
rm SpiNNMan -R -f

dosetupinstall spalloc
rm spalloc -R -f

dosetupinstall SpiNNFrontEndCommon
rm SpiNNFrontEndCommon -R -f

domake sPyNNaker/neural_modelling/ clean
domake sPyNNaker/neural_modelling/
dosetupinstall sPyNNaker

# # sPyNNaker8 is probably outdated
# dosetupinstall sPyNNaker8
# rm sPyNNaker8 -R -f

## NOTE: these two are not required for my purposes
# dosetupinstall SpiNNakerGraphFrontEnd
# rm SpiNNakerGraphFrontEnd -R -f

if ["$BUILD_JAVASPINNAKER" = true] ; then
    # Makes the project
    domvn JavaSpiNNaker
    # mv JavaSpiNNaker /home/spinnaker/spinnaker3.8/lib/python3.8/ -f
    pecho done maven
else
    pwarn JavaSPiNNaker was not installed
fi

pecho Linking PyNN...
python -m spynnaker.pyNN.setup_pynn

LAST_ERROR=$?
if [ $LAST_ERROR -ne 0 ]; then
    echo INSTALLATION FAILED
    echo "$LAST_ERROR" > $INITDIR/last_error.txt
    exit_env $LAST_ERROR
fi
pecho Done installing.

############### CHECK ###############
cd $VIRTUAL_ENV
echoline INSTALLATION CHECK
pecho listing files:
ls -al
pecho Remote spynnaker version: $(python -c "import spynnaker; print(spynnaker.__version__)")
pecho Remote spinnutils version: $(python -c "import spinn_utilities; print(spinn_utilities.__version__)")
pecho Result of pip freeze:
pip freeze
############### SIMULATION #############
export DJANLOO_NEURAL_SIMULATOR=spiNNaker
cd $INITDIR

# mv sPyNNaker/simulation.py ./
# clean_downloads # in case simulation goes well it does not return all the src anyway

pecho generating check for file download in ${INITDIR}
touch check_files_download.txt

echoline SIMULATION
pecho Simulating...

python3 vogels-abbott.py

