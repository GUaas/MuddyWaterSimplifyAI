import difflib
import re

def remove_consecutive_duplicates(text):
    """去除连续重复字符"""
    if not text:
        return ''
    cleaned = [text[0]]
    for char in text[1:]:
        if char != cleaned[-1]:
            cleaned.append(char)
    return ''.join(cleaned)

def remove_duplicate_punctuation(text):
    """处理重复的标点符号，只保留第一个"""
    # 定义正则表达式，用于匹配多个标点符号的组合
    punctuation_pattern = r'([^\w\s])[^\w\s]+'
    # 替换多个标点符号为第一个标点符号
    return re.sub(punctuation_pattern, r'\1', text)

def fix_text(original, simplified):
    # 预处理：去除简化文本中的连续重复
    simplified_clean = remove_consecutive_duplicates(simplified)

    # 使用SequenceMatcher进行差异比对
    matcher = difflib.SequenceMatcher(None, original, simplified_clean)
    fixed = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # 两文本相同部分直接保留
            fixed.append(simplified_clean[j1:j2])
        elif tag == 'replace':
            # 处理替换情况，保留简化文本内容
            orig_part = original[i1:i2]
            simp_part = simplified_clean[j1:j2]

            # 仅在简化文本过短时尝试从原始文本补充
            if len(simp_part) < 2 and len(orig_part) > 2:
                # 尝试从原始文本中提取可能的词首或词尾
                if i1 > 0 and i1 < len(original) - 1:
                    fixed.append(original[i1-1:i2])
                else:
                    fixed.append(orig_part)
            else:
                fixed.append(simp_part)
        elif tag == 'delete':
            # 处理缺失情况，仅在必要时补充单字
            deleted_text = original[i1:i2]
            if len(deleted_text) == 1 and i1 > 0 and i1 < len(original)-1:
                # 检查是否为可能的单字缺失
                prev_char = original[i1-1]
                next_char = original[i1+1]
                if prev_char != deleted_text and next_char != deleted_text:
                    fixed.append(deleted_text)
        elif tag == 'insert':
            # 处理插入情况，通常是不必要的添加，忽略
            pass

    result = ''.join(fixed)
    # 处理重复标点
    result = remove_duplicate_punctuation(result)

    return result

# 测试示例
#original = "好呀！日常中有好多有趣的事儿呢。最近有没有吃到什么特别好吃的美食，或者发现了新的好玩的地方呀？也可以和我说说生活里的小烦恼哦。"
#simplified = "最近有没有吃到什么好吃的美食，有新新好玩的地。"

#print(fix_text(original, simplified))