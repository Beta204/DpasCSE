import re
import matplotlib.pyplot as plt

# 定义句子长度的四个类别
length_categories = {
    'Short': (0, 50),  # 短句子，长度在0到10个单词之间
    'Medium': (51, 100),  # 中等长度句子，长度在11到20个单词之间
    'Long': (101, 150),  # 长句子，长度在21到30个单词之间
    'Very Long': (151, float('inf'))  # 非常长句子，长度超过30个单词
}

# 读取文本文件并计算每行句子长度，然后归类
def sentence_length_visualization(file_path):
    # 初始化类别计数字典
    category_counts = {category: 0 for category in length_categories.keys()}

    # 打开文件并读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式匹配句子（以句号、问号、感叹号结尾）
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', line)
            for sentence in sentences:
                if sentence:  # 确保句子不为空
                    # 计算句子长度（不包括标点符号）
                    length = len(re.sub(r'[^\w\s]', '', sentence))
                    # 根据长度将句子归类
                    for category, (min_length, max_length) in length_categories.items():
                        if min_length <= length < max_length:
                            category_counts[category] += 1
                            break

    # 计算总句子数
    total_sentences = sum(category_counts.values())

    # 计算每个类别的百分比
    category_percentages = {category: (count / total_sentences) * 100 for category, count in category_counts.items()}

    # 可视化展示结果
    categories = list(category_percentages.keys())
    percentages = list(category_percentages.values())

    plt.figure(figsize=(10, 5))
    plt.bar(categories, percentages, color='skyblue')
    plt.title('Sentence Length Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 使用函数
# 请将 'example.txt' 替换为你的文本文件路径
sentence_length_visualization('wiki1m_for_simcse.txt')
