#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

URL_en="https://storage.googleapis.com/vietai_public/best_vi_translation/v2/train.en"
URL_vi="https://storage.googleapis.com/vietai_public/best_vi_translation/v2/train.vi"

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src=en
tgt=vi
lang=en-vi
prep=mtet_en_vi
tmp=$prep/tmp

mkdir -p $prep $tmp

cd $tmp
echo "Downloading data from ${URL_en}..."
wget "$URL_en"
echo "Downloading data from ${URL_vi}..."
wget "$URL_vi"

cd ../..

TRAIN=$tmp/train.en-vi
BPE_CODE=$prep/code
rm -f $TRAIN

for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done


echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for f in $src $tgt; do
      echo "apply_bpe.py to ${f}..."
      python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/train.$f > $prep/train.$f
done




split -l 4000000 $prep/train.en
rm $prep/train.en
mv xaa $prep/train.en
mv xab $prep/valid.en


split -l 4000000 $prep/train.vi
rm $prep/train.vi
mv xaa $prep/train.vi
mv xab $prep/valid.vi