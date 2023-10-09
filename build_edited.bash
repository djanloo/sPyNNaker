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

export MAKE_LOG_FOLDER=${PWD}/MAKE_LOGS
mkdir $MAKE_LOG_FOLDER


# Move where the repos are
cd $VIRTUAL_ENV
mkdir AUX_INSTALL
export AUX_INSTALL_FOLDER=$VIRTUAL_ENV/AUX_INSTALL
cd AUX_INSTALL


# Delete the current version
pecho Removing default sPyNNaker package
rm sPyNNaker -R -f
# Substitute my version
pecho Copying modified sPyNNaker package
cp -R $INITDIR ./
# Rename to prevent from name mismatch
mv sPyNNaker/spynnaker-custom sPyNNaker/spynnaker


# Build the C Code
echoline COMPILATION

export SPINN_DIRS=$AUX_INSTALL_FOLDER/spinnaker_tools
export NEURAL_MODELLING_DIRS=$AUX_INSTALL_FOLDER/sPyNNaker/neural_modelling

pecho SPINN_DIRS was set to $SPINN_DIRS
pecho NEURAL_MODELLING_DIRS was set to $NEURAL_MODELLING_DIRS


domake sPyNNaker/neural_modelling/ clean
domake sPyNNaker/neural_modelling/
dosetupinstall sPyNNaker

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

###########################################
export DJANLOO_NEURAL_SIMULATOR=spiNNaker
cd $INITDIR

