if [ "$#" -ne 1 ]
then
    echo "Usage: trans <destination>"
else
    dest=$1
    trubar --conf trubar-config.yaml translate -s ../orangecontrib/pumice -d $dest/orangecontrib/pumice msgs.jaml
fi
