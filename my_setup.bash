#!/bin/bash

#AUTHOR: djanloo
#VERSION: 2.0

# DESCRIPTION: this is an attempt to reproduce the << super easy >> developer installation chain 
# described at http://spinnakermanchester.github.io/development/devenv.html
#
# My necessity is to apply minor (really, really small) edits on the C code of neuron dynamics

# Installation options
UPGRADE_JAVA=false
CLONE_NIGHTLY=false #TODO

# the_world_is_flat=true
# # ...do something interesting...
# if [ "$the_world_is_flat" = true ] ; then
#     echo 'Be careful not to fall off!'
# fi


# Echo with current position
pecho() {
    echo -e "\e[34m$(pwd)\e[97m>> " $@
}

# Warning with current position
pwarn() {
    echo -e  "\e[34m$(pwd)\e[97m>> \e[31mWARNING\e[97m:" $@
}

# Prints a line
echoline() {
    TITLE=$@
    printf '%.0s-' {1..50};printf "\e[41m$TITLE\e[49m";printf '%.0s-' {1..50};printf '\n'
}

## COPIED FROM setup.bash
dosetupinstall() {
    DIR=$1
    if [ -f "$DIR/setup.py" ]; then
        pecho "Setting up $DIR"
        (cd $DIR; pip install . > /tmp/last_setup.tmp 2>&1 )
        LAST_ERROR=$?
        if [ $LAST_ERROR -ne 0 ]; then
            cat /tmp/last_setup.tmp
            clean_downloads
            exit $LAST_ERROR
        fi
    else
        pecho "Skipping setting up $DIR as no setup.py found"

## COPIED FROM setup.bash
domvn() {
    DIR=$1 tags
    pecho "Building $DIR"
    ${MAVEN} -f $DIR package -Dmaven.test.skip=true &> "$MAKE_LOG_FOLDER/$(basename $DIR).java.txt" 2>&1
    LAST_ERROR=$?
    if [ $LAST_ERROR -ne 0 ]; then
        cat "$MAKE_LOG_FOLDER/$(basename $DIR).java.txt"
        clean_downloads
        exit $LAST_ERROR
    fi
}
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
        clean_downloads
        exit $LAST_ERROR
    fi
}

clean_downloads() {
    rm spinnaker_tools -R -f
    rm PyNN -R -f
    rm SpiNNUtils -R -f
    rm SpiNNMachine -R -f
    rm SpiNNStorageHandlers -R -f
    rm PACMAN -R -f
    rm SpiNNMan -R -f
    rm DataSpecification -R -f
    rm spalloc -R -f
    rm spinn_common -R -f
    rm spalloc_server -R -f
    rm SpiNNFrontEndCommon -R -f
    rm sPyNNaker -R -f
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

    # Check after install
    echo
    pecho AFTER upgrade of java and maven:
    echo
    java -version
    echo
    ${MAVEN} -version
    echo
    cd ..
}
############# Some preliminaries ############
export INITDIR=$(pwd)

# Edit where to place logs
# Avoiding this can cause errors
mkdir LOGS
chmod 777 LOGS
export C_LOGS_DICT=$(pwd)/LOGS/logs.sqlite3
export MAKE_LOG_FOLDER=${PWD}/LOGS

pecho Started installation routine by djanloo

pecho Retrieving configuration files
mkdir ../config_files
cp $(<config_files_list.txt) ../config_files

pecho Remote spynnaker version: $(python -c "import spynnaker; print(spynnaker.__version__)")
pecho Result of pip freeze:
pip freeze

pecho Exporting environment variables to ${INITDIR}/env_var.txt
printenv > ${INITDIR}/env_var.txt

# Now I try to reconstruct what they did one their own version
# of setup.bash

pecho Copying modified sPyNNaker package
# Move where the repos are
cd /home/spinnaker/spinnaker
# Delete the current version
rm sPyNNaker -R -f
# Substitute my version
cp -R $INITDIR ./

## Upgrade pip and other packages
pip install packaging
pip -V

## Not needed anymore due to 17-JUL-23 upgrade
# pip install --upgrade pip setuptools wheel

### CLONING REPOS
# These are mine
# git clone https://github.com/djanloo/SpiNNUtils.git
pwarn Repos were not upgraded

echo
pecho ended cloning stage at $(date)
pecho listing files:
ls -al
echo
echoline INSTALL NON-COMPILED STUFF
pecho starting make stage at $(date)
echo
pecho installing non-compiled stuff before

dosetupinstall SpiNNUtils
rm SpiNNUtils -R -f

wait

dosetupinstall SpiNNMachine
rm SpiNNMachine -R -f

dosetupinstall SpiNNStorageHandlers
rm SpiNNStorageHandlers -R -f

dosetupinstall DataSpecification
rm DataSpecification -R -f

dosetupinstall PACMAN
rm PACMAN -R -f

echo

# Build the C Code
echoline COMPILATION

export SPINN_DIRS=${PWD}/spinnaker_tools
export NEURAL_MODELLING_DIRS=${PWD}/sPyNNaker/neural_modelling
domake $SPINN_DIRS clean
domake $SPINN_DIRS
pecho SPINN_DIRS was set to $SPINN_DIRS
pecho NEURAL_MODELLING_DIRS was set to $NEURAL_MODELLING_DIRS
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

# Check java and maven versions
echo
pecho BEFORE upgrade of java and maven:
echo
java -version
echo
mvn -version
echo

if [ "$UPGRADE_JAVA" = true ] ; then
    pecho Updating JAVA...
    upgrade_java
else
    pwarn Java and Maven have not been upgraded
    pecho Setting MAVEN=mvn
    MAVEN="mvn"
fi

echoline INSTALLATION OF COMPILED STUFF
pecho "starting setup stage for compiled stuff"

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

dosetupinstall sPyNNaker8
rm sPyNNaker8 -R -f

# dosetupinstall SpiNNakerGraphFrontEnd
# rm SpiNNakerGraphFrontEnd -R -f

# # Makes the project
# domvn JavaSpiNNaker
# # mv JavaSpiNNaker /home/spinnaker/spinnaker3.8/lib/python3.8/ -f
# echo done maven
# echo


pecho linking PyNN...
python -m spynnaker.pyNN.setup_pynn
echo

LAST_ERROR=$?
if [ $LAST_ERROR -ne 0 ]; then
    clean_downloads
    echo INSTALLATION FAILED
    exit $LAST_ERROR
fi
pecho Done installing.

############### SIMULATION #############
cd $INITDIR

mv sPyNNaker/simulation.py ./
# clean_downloads # in case simulation goes well it does not return all the src anyway

pecho generating check for file download in ${INITDIR}
touch check_files_download.txt

echoline SIMULATION
pecho Simulating...

python3 simulation.py

