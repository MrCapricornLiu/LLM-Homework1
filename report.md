# 1. Tokenization（10分）
# 1.1 实现BPE，训练Tokenizer（6分）
## 【1分】在实验报告中简要概述一下BPE算法和基于BPE算法训练LLM tokenizer的流程。
### BPE算法概述
BPE算法 (Byte Pair Encoding) 是一种基于统计的方法，最初用于文本压缩，但被广泛应用于自然语言处理（NLP）中作为词汇切分的工具。其核心思想是通过不断合并最频繁的字符对，逐步构建一个子词级别的词汇表，从而提高模型处理词语的灵活性和效率。

### BPE算法的基本步骤：  
- 初始化： 把输入的文本转化为字符级别的表示（即将单词分解为字符）。  
- 计算频率： 统计文本中所有相邻字符对（bigram）的频率。  
- 合并操作： 每次合并频率最高的字符对，形成一个新的“子词”单元，并替换原文中的该对字符。  
- 重复步骤： 重复合并步骤直到达到设定的词汇大小（vocab_size）。  

### 基于BPE算法训练LLM Tokenizer的流程：
- 文本预处理： 将训练数据进行预处理，例如分句、去除无用字符等，得到一个纯文本的字符串。  
- 初始化tokenizer： 创建一个空的tokenizer并初始化基本的词汇表。初始的词汇表通常包含所有单一字符和一个特殊的\<unk\>标记。  
- BPE训练： 使用BPE算法从训练数据中逐步合并字符对，更新词汇表直到达到目标词汇大小（vocab_size）。  
- 编码与解码： 使用训练好的tokenizer对新的文本进行编码和解码操作，将文本转化为模型可以处理的token ID序列，或将token ID序列转换回原始文本。  
## 【2分】实现一个基于BPE算法的tokenizer
见bpe/bpe_tokenizer.py内的class Tokenizer。

## 【1分】hw1-code/bpe/manual.txt是《北京大学研究生手册（2023版）》中提取出来的文字，请你用它来训练你的tokenizer，vocab_size为1024。

见bpe/main.py内的具体代码。


## 【1分】用它来encode再decode manual.txt，检查与原始manual.txt是否完全一致？

- 见bpe/main.py内的具体代码。
- 比较结果：完全一致。

## 【1分】学习使用huggingface transformers中的tokenizer，使用它加载GPT-2的tokenizer，然后使用它和你训练的tokenizer分别encode以下句子，比较两者的输出，简要解释长度上和具体token上不同的原因是什么。
- 比较结果（具体代码见bpe/main.py）
    - BPE Tokenizer for Sentence 1: Total tokens: 953
    - GPT-2 Tokenizer for Sentence 1: Total tokens: 185
    - BPE Tokenizer for Sentence 2: Total tokens: 149
    - GPT-2 Tokenizer for Sentence 2: Total tokens: 306
- 差异及原因
    - 长度差异
        - GPT-2的tokenizer是基于子词（subword）单元的，它会使用大量的预定义词汇表中的子词单元来表示输入句子。由于GPT-2的tokenizer已经有了大量的子词单元，它可能将一些常见的词（如Peking University、China等）映射为一个token，导致token数量较少。
        - 训练后的BPE tokenizer在词汇表大小较小的情况下，可能会更多地使用字符级别的token进行编码，这会导致token数量更多，编码结果也会有所不同。
    - 具体Token差异
        - GPT-2的tokenizer会使用其大规模的预训练词汇表来表示常见的单词和词组，因此它会更多地使用已知的子词单元（例如Peking University可能会作为一个token表示）
        - 而BPE tokenizer可能会拆分为更小的单元。
        - 例如，BPE可能会将"Peking University"分为多个token（如Pek, ing, Univer, sity等），而GPT-2 tokenizer可能会将其直接编码为一个token。



# 1.2 回答问题

## 1.2.1 Python中使用什么函数查看字符的Unicode，什么函数将Unicode转换成字符？并使用它们查看“北”“大”的Unicode，查看Unicode为22823、27169、22411对应的字符。

- **查看字符的Unicode：**
  在Python中，可以使用`ord()`函数查看字符的Unicode编码。`ord()`函数返回一个字符对应的Unicode码点。

  ```python
  print(ord('北'))  # 22823
  print(ord('大'))  # 27169
  ```

- **将Unicode转换成字符：**
  要将Unicode码点转换回字符，可以使用`chr()`函数。`chr()`函数接受一个Unicode码点，返回对应的字符。

  ```python
  print(chr(22823))  # 北
  print(chr(27169))  # 大
  print(chr(22411))  # 查
  ```

## 1.2.2 Tokenizer的vocab size大和小分别有什么好处和坏处？

- vocab size大：
  - 好处：
    - 能够表示更多的单词、词组或语法单元，因此对各种文本有更高的覆盖率。
    - 能够更准确地表示常见的单词和短语，不容易拆分成多个子词。
  - 坏处：
    - 需要更多的存储空间，词汇表更大，内存占用更高。
    - 词汇表过大时，稀疏性增大，可能出现词频分布不均，导致部分词汇频次较低的情况下学习效果不好。
    - 训练时间长，模型的推理速度可能变慢。

- vocab size小：
  - 好处：
    - 词汇表小，内存占用低，模型训练和推理速度更快。
    - 对低频词进行拆分，可以让模型学习更多的子词表示，提升泛化能力。
  - 坏处：
    - 可能无法完整表示一些常见的词，导致需要频繁地将词拆解为多个子词，影响模型性能。
    - 对于长尾词的处理不如大vocab size的tokenizer有效，导致模型的词汇覆盖率较低。

## 1.2.3 为什么 LLM 不能处理非常简单的字符串操作任务，比如反转字符串？

LLM主要是通过概率和上下文推测生成文本，而不是通过明确的编程逻辑来执行任务。反转字符串等简单的字符串操作通常需要明确的步骤和控制流逻辑，但LLM并不具备这种明确的逻辑推理能力。它处理的是基于语言的概率分布，并且对于类似编程任务，往往依赖上下文学习，但很难推理出正确的操作步骤。

## 1.2.4 为什么 LLM 在非英语语言（例如日语）上表现较差？

LLM在非英语语言上表现较差，通常是因为训练数据中英语占主导地位，非英语语言的训练数据较少。因此，模型对非英语语言的语法、词汇、句子结构等理解能力较弱。尤其是一些语言（如日语）具有复杂的书写系统和语法规则，且词汇表可能需要更多的子词切分，模型需要更多的非英语数据来进行有效训练。

## 1.2.5 为什么 LLM 在简单算术问题上表现不好？

LLM通常通过预测下一个词的概率来生成文本，而不是像传统的计算机程序那样执行具体的数值运算。对于简单的算术问题，LLM通常是基于语言模式来生成答案，而不是通过执行数学计算。因此，模型在算术问题上容易出错，特别是当问题涉及多步骤运算时，LLM更容易偏离正确答案。

## 1.2.6 为什么 GPT-2 在编写 Python 代码时遇到比预期更多的困难？

GPT-2是一个语言模型，主要通过预测文本来生成答案。虽然它可以生成合理的自然语言文本，但编写Python代码需要精确的语法、逻辑和结构。GPT-2没有显式的编程语法学习机制，也没有进行特定编程任务的优化训练。因此，它在编写代码时可能会犯一些语法错误，或者没有足够的上下文来推断出正确的逻辑，导致更多的困难。

## 1.2.7 为什么 LLM 遇到字符串 “<|endoftext|>” 时会突然中断？

`"<|endoftext|>"` 是GPT类模型等语言模型在训练时用来表示文本的结束标记。在训练过程中，模型被训练去预测文本直到遇到这个特殊标记，因此，当模型生成文本时遇到这个标记，表示生成任务的结束，模型会停止输出。

## 1.2.8 为什么当问 LLM 关于 “SolidGoldMagikarp” 的问题时 LLM 会崩溃？

"SolidGoldMagikarp" 是一种网络迷因（meme）或特定语境中的词汇，可能是某个特定领域的术语或者与特定知识库相关。LLM在遇到这种非常特殊的输入时，若没有相关训练数据或上下文，它可能无法正确处理这个词汇，从而出现崩溃或者生成不正确的答案。

## 1.2.9 为什么在使用 LLM 时应该更倾向于使用 YAML 而不是 JSON？

YAML相比JSON具有更简洁的语法，尤其是在处理复杂的嵌套结构时更加直观易读。LLM处理文本时可以更容易理解YAML格式的语义，并能够生成结构化的输出。JSON语法虽然更严格，但可能更难与自然语言交互。YAML对于人类编辑更加友好，因此在与LLM交互时，YAML更容易生成符合预期的结果。

## 1.2.10 为什么 LLM 实际上不是端到端的语言建模？

LLM虽然是在预训练阶段通过大量的文本数据进行训练的，但它的生成过程并不是完全端到端的语言建模。因为LLM的目标是预测下一个词或标记，并生成连贯的文本，而不是执行某种特定的任务或者逻辑推理。端到端的语言建模通常意味着模型能够在一个输入到输出的过程中，经过少量的手动干预或规则，直接得到正确的输出。而LLM往往需要较长的上下文信息，且它的输出依赖于训练数据中的概率分布，而不是明确的规则推理。


# 2. LLM Implementation（10分）


# 2.1 Initial Commit: basic architecture of GPT-2

## 2.1.2 实验过程

本部分代码的主要目标是复现GPT-2模型的核心架构，包括因果自注意力机制、自注意力模块（CausalSelfAttention）、前馈神经网络（MLP）、Transformer块（Block）、GPT配置类（GPTConfig）、GPT模型主体（GPT）、模型加载与验证，以及文本生成过程。具体实现步骤如下：
#### 因果自注意力机制（CausalSelfAttention）：

- **实现因果自注意力**：确保模型在生成文本时只能关注当前词及其之前的词，避免未来信息泄露。
- **线性投影**：通过线性层`c_attn`生成查询（Q）、键（K）、值（V）的投影。
- **多头分割和重组**：将Q、K、V分割为多个头，并进行转置以适应多头计算。
- **缩放点积注意力**：计算Q与K的点积，并进行缩放。
- **应用遮罩**：使用下三角遮罩`bias`确保因果性，避免未来信息的影响。
- **注意力权重计算**：通过softmax计算注意力权重，并与V相乘得到注意力输出。
- **输出线性层**：通过`c_proj`线性层将多头输出重新组合。
#### 前馈神经网络（MLP）：

- **两层线性变换**：第一层将嵌入维度扩展到4倍，第二层将其缩回原始维度。
- **GELU激活函数**：在两层线性变换之间应用GELU激活函数，增加模型的非线性表达能力。
#### Transformer块（Block）：

- **层归一化与残差连接**：每个Transformer块包含两次层归一化和残差连接，分别在自注意力和前馈神经网络之后，增强模型的稳定性和深度。
- **集成自注意力与MLP**：结合CausalSelfAttention和MLP模块，形成标准的Transformer层。
#### GPT配置类（GPTConfig）：

- **配置参数定义**：使用`@dataclass`定义模型的配置参数，如词汇表大小、嵌入维度、层数、头数等，支持不同规模的GPT-2模型配置。
#### GPT模型主体（GPT）：

- **嵌入层**：包括词嵌入（wte）和位置嵌入（wpe）。
- **多层Transformer块**：使用`ModuleList`构建多层Transformer块。
- **最终层归一化**：在所有Transformer块之后应用层归一化。
- **语言模型头**：通过线性层`lm_head`将嵌入维度转换为词汇表大小，输出logits。
- **从预训练模型加载权重**：实现`from_pretrained`方法，从Hugging Face加载预训练的GPT-2模型权重，通过参数对齐和必要的权重转置，确保模型能够正确加载预训练权重。
#### 模型加载与验证：

- **设备检测**：自动检测并选择可用的计算设备（CPU、CUDA、MPS）。
- **模型初始化**：加载预训练的GPT-2模型，并将其设置为评估模式。
- **前缀token处理**：使用`tiktoken`库对输入文本进行编码，准备生成任务的输入。
- **文本生成逻辑**：通过设置随机种子，使用top-k采样策略生成指定长度的文本序列，并打印生成结果。
## 2.1.2 代码运行结果
```
使用设备: cuda
> Hello, I'm a language model, scrubaliamut Zo retained Form deal nearbyESCODirectorchainslect.; homosexualityosukevision260 viewing HT interface paralyzedyo
> Hello, I'm a language model, presidentoxy phones skulladiqulptFDreporting Godd startersclasslect ال beginnings eclipsformationpodcastpeer workshop competenceampionorbit
> Hello, I'm a language model,593 SetTextColor fatalCVE RoryISSCaronomic fightershei radiationINSTwings396 Swords Cookadvert spe conjunction poemOV penetrating
> Hello, I'm a language model,igslistNotes Preferred restaur Islands absor hikes Reef enclosure sprint enormous CentOS arises comprised discsormon Observatory� cookedrypt Estateete
> Hello, I'm a language model,Auth aquPa amused�cks vert delegates`. bus patron.)430 reb.? bolts SU Jen confess Craigslist parked Ur
```
## 2.1.3 学习笔记
#### 多头注意力机制：

通过并行计算多个注意力头，模型能够捕捉序列中不同层次和不同方面的依赖关系。理解了如何通过线性变换分割和重组注意力头，以及如何通过拼接和线性层整合各头的信息。
#### 因果遮罩的实现：

使用下三角矩阵作为遮罩，确保在生成过程中模型只能关注当前及之前的位置，避免未来信息泄露。这是生成任务中保持生成因果性的关键。
#### 参数迁移与对齐：

在从预训练模型加载权重时，处理了参数名称和形状的不一致问题，如需要转置某些权重以匹配本地实现。这加深了对不同模型实现细节的理解，以及如何在不同框架或实现之间迁移权重。
#### 模块化设计：

通过将自注意力、MLP和Transformer块模块化，提升了代码的可读性和可维护性。同时，这种设计便于扩展和修改，适应不同规模或变种的GPT模型需求。
## 2.1.4 存在的问题与解决方法：
#### 权重转置问题：

- **问题**：从Hugging Face加载的某些权重（如`attn.c_attn.weight`、`attn.c_proj.weight`、`mlp.c_fc.weight`、`mlp.c_proj.weight`）需要进行转置才能与本地实现匹配。
- **解决方法**：在加载权重时，检查需要转置的权重名称，并在复制权重时应用转置操作，确保形状和数据正确对应。
#### 参数键不匹配：

- **问题**：在从预训练模型复制权重时，参数键的数量和名称可能存在不一致，导致加载失败。
- **解决方法**：通过断言确保两个模型的参数键数量一致，并过滤掉不需要的缓冲区参数（如`.attn.bias`），确保只复制有效的权重参数。
#### 配置灵活性：

- **问题**：当前`GPTConfig`类虽然支持基本的配置参数，但对于更复杂的模型变体，可能需要更多的配置选项。
- **解决方法**：扩展`GPTConfig`类，增加更多可配置的参数，如不同的激活函数选择、不同的层归一化方式等，以支持更广泛的模型需求。