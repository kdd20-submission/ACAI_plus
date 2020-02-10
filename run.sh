BASEDIR="";
MODEL_REMAINDER="EPOCHS400_BATCHSIZE512";


declare -A MODELS;
MODELS["BASELINE_ORIG_ACAI"]="baseline_orig_acai";
MODELS["VAE_PLUS"]="vae_acai";
MODELS["IDEC_PLUS"]="idec_acai";

DATASET="CIFAR";
dataset="cifar";

# Run specified model
python setup.py install;

for model in "${!MODELS[@]}"; do
    for i in 1 2 3 4 5; do
        echo ${BASEDIR}/configs/$DATASET/${DATASET}_"${model}"_"${MODEL_REMAINDER}"_SEED$i.yaml
        python ${BASEDIR}/acaiplus/models/"${MODELS[$model]}".py -c ${BASEDIR}/configs/$DATASET/${DATASET}_"${model}"_"${MODEL_REMAINDER}"_SEED$i.yaml > ${BASEDIR}/RESULTS/${dataset}/${DATASET}_"${model}"$
    done
done
