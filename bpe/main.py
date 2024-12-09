import pdb
from bpe_tokenizer import Tokenizer
from transformers import GPT2Tokenizer


def main():
    # 读取manual.txt文件内容
    with open('/home/pku0033/LLM-Homework1/bpe/manual.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # 创建并训练tokenizer
    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size=1024)

    # 编码并解码验证一致性
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)

    # 检查是否一致
    print("用它来encode再decode manual.txt，检查与原始manual.txt是否完全一致？")
    print("答案：", text == decoded_text)

    pdb.set_trace()
    # 加载GPT-2的tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 句子一
    sentence_1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."

    # 句子二
    sentence_2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"

    # 使用训练的tokenizer进行编码
    encoded_bpe_1 = tokenizer.encode(sentence_1)
    encoded_bpe_2 = tokenizer.encode(sentence_2)

    # 使用GPT-2的tokenizer进行编码
    encoded_gpt2_1 = gpt2_tokenizer.encode(sentence_1)
    encoded_gpt2_2 = gpt2_tokenizer.encode(sentence_2)

    # 打印输出（只显示token数量和前几个token）
    print("BPE Tokenizer for Sentence 1:", encoded_bpe_1[:10], "Total tokens:", len(encoded_bpe_1))
    print("GPT-2 Tokenizer for Sentence 1:", encoded_gpt2_1[:10], "Total tokens:", len(encoded_gpt2_1))

    print("BPE Tokenizer for Sentence 2:", encoded_bpe_2[:10], "Total tokens:", len(encoded_bpe_2))
    print("GPT-2 Tokenizer for Sentence 2:", encoded_gpt2_2[:10], "Total tokens:", len(encoded_gpt2_2))


if __name__ == '__main__':
    main()