from bpe_tokenizer import Tokenizer


def main():
    # 读取manual.txt文件内容
    with open('bpe/manual.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # 创建并训练tokenizer
    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size=1024)

    # 编码并解码验证一致性
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)

    # 检查是否一致
    print("Text match:", text == decoded_text)



if __name__ == '__main__':
    main()