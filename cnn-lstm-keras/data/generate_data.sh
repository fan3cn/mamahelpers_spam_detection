echo "" > pos.tmp
echo "" > neg.tmp

awk '{print 0"\t"$0 }' normal.csv | gshuf > pos.tmp
awk '{print 1"\t"$0 }' spam_other_src.csv | gshuf > neg.tmp
awk '{print 1"\t"$0 }' spam_mama_helper.csv | gshuf >> neg.tmp

head -1000  pos.tmp > all.tmp
cat neg.tmp >> all.tmp

gshuf all.tmp > all.data

#rm -rf *.tmp
# split data
#num=`cat all.data | wc -l`
#echo $num
#train_num=$[8*$num/10]
#test_num=$[2*$num/10]
#head -$train_num all.data > X.train.tmp
#tail -$test_num all.data > X.test.tmp

awk -F '\t' '{print $1} ' all.data > Y.train
awk -F '\t' '{print $2} ' all.data > X.train

awk -F '\t' '{print $2}' all.data > all_text

rm -rf *.tmp

