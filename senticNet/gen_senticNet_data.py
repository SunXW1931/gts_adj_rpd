
path = '../senticNet/senticnet_word.txt'
new_path = '../senticNet/senticnet_word_0.9.txt'
fp = open(path, 'r')
w_fp = open(new_path, 'w')
word_dic = {}
for line in fp:
    line = line.strip()
    word, score = line.split('\t')
    word_dic[word] = float(score)
fp.close()

for key in list(word_dic.keys()):
    value = word_dic[key]
    if abs(value) < 0.9:
        del word_dic[key]

for word in word_dic:
    w_fp.write(word+'\t'+str(word_dic[word])+'\n')

w_fp.close()