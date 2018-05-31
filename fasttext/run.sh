echo 'Preprocessing data...'
echo "" > message.train
awk '{print "__label__1 "$0}' ./data/spam_mama_helper.csv > message.train
awk '{print "__label__1 "$0}' ./data/spam_other_src.csv >> message.train

cat ./data/normal.csv | awk '{print "__label__0 "$0}' | gshuf > all_data

head -1000 all_data > message.valid
head -4000 all_data > message.train

echo 'Training model...'
./fasttext/fasttext supervised -input message.train  -output model

echo 'Validate model...'
./fasttext/fasttext test model.bin message.valid

echo 'Predicting...'
./fasttext/fasttext predict model.bin message.test | sed 's/__label__//' > message.predict

awk 'NR == FNR{a[NR] = $1} NR != FNR{if(a[FNR]==1) {print "SPAM" "\t" $0} else{print "NOT-SPAM" "\t" $0} }' message.predict message.test > result.txt

# Test the classifier interactively
#./fastText/fasttext predict model.bin -

echo 'Done!'


