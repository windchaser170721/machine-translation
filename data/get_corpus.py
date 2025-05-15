import json

def process():
    """
    training.txt 18000
    validation.txt 500
    testing.txt 2636
    """
    data = []
    with open("data/txt/training.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去掉首尾空白字符
            if not line:
                continue  
            
            parts = line.split("\t")
            if len(parts) >= 2:
                english_sentence = parts[0].strip()
                chinese_sentence = parts[1].strip()
                data.append([english_sentence, chinese_sentence])
            else:
                print("发现格式异常的行：", line) 
    with open("data/validation.json", "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

def get_corpus():
    files = ['training', 'validation', 'testing']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        corpus = json.load(open('./data/json/' + file + '.json', 'r'))
        for item in corpus:
            ch_lines.append(item[1] + '\n')
            en_lines.append(item[0] + '\n')

    with open(ch_path, "w") as fch:
        fch.writelines(ch_lines)

    with open(en_path, "w") as fen:
        fen.writelines(en_lines)

    # lines of Chinese: 21136
    print("lines of Chinese: ", len(ch_lines))
    # lines of English: 21136
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")


if __name__ == "__main__":

    process()
    # get_corpus()