from flask import Flask, render_template, request, jsonify
import paddle
import pickle
from collections import Counter
import os
import re
from modelFramework.Seq2Seq import Seq2SeqInfer  # 保持原有模型定义
from util.characterMatching import fix_text


app = Flask(__name__)

# 加载预处理数据和模型（仅在启动时加载一次）
cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/v3-40w-64b/pretreatment_cache.pkl")
with open(cache_path, 'rb') as f:
    cache_data = pickle.load(f)

word2id_dict = cache_data['word2id_dict']
couplet_maxlen = cache_data['couplet_maxlen']
word_size = cache_data['word_size']
id2word_dict = cache_data['id2word_dict']

bos_id = word2id_dict['<start>']
eos_id = word2id_dict['<end>']

# 初始化推理模型
infer_model = paddle.Model(
    Seq2SeqInfer(word_size, 256, 128, 2, bos_id, eos_id, 10, couplet_maxlen)
)
infer_model.prepare()
model_path = os.path.join(os.path.dirname(__file__), "models/v3-40w-64b/MuddyWaterAI-Ss-V3-64b.pdopt")
infer_model.load(model_path)


def Chinesetoid(insrc, word2id_dict, couplet_maxlen=couplet_maxlen, start=bos_id, padid=eos_id):
    result = [start, ]
    for ch in insrc:
        result.append(word2id_dict.get(ch, padid))  # 添加OOV处理
    result.append(eos_id)
    result_len = len(result)
    if len(result) < couplet_maxlen:
        result += [padid] * (couplet_maxlen - len(result))
    result_len_tensor = paddle.to_tensor([result_len])
    return paddle.unsqueeze(paddle.to_tensor(result), axis=0), result_len_tensor


def get_second(inputs, id2word_dict):
    finished_seq = infer_model.predict_batch(inputs=list(inputs))[0][0]
    input_re = inputs[0][0][1:]
    input_re = paddle.tolist(input_re)

    in_input = []
    for subre in input_re:
        if subre == eos_id:
            break
        in_input.append(subre)

    result = []
    for subseq in finished_seq:
        resultid = Counter(list(subseq)).most_common(1)[0][0]
        if resultid == eos_id:
            break
        result.append(resultid)

    word_list_f = [id2word_dict.get(id, '<unk>') for id in in_input]  # 添加未知词处理
    word_list_s = [id2word_dict.get(id, '<unk>') for id in result]
    sequence = "".join(word_list_s)
    # 删除模型输出中的句号
    sequence = re.sub(r'[。]', '', sequence)
    return sequence


def split_text_v3(text,
                  max_seq_len=122,  # 使用训练集最大长度122
                  hard_max_len=50,  # 允许略超长时的智能处理
                  prefer_split_chars='。！？；∶!?;',  # 优先分割字符
                  secondary_split_chars='，,',  # 次要分割字符
                  no_split_chars='、'):  # 不分割字符
    """
    第三代智能分割算法，优先保证不超过max_seq_len，同时保持语义完整性
    """
    # 阶段一：基础分割
    segments = []
    current = []
    current_len = 0

    for i, char in enumerate(text):
        # 遇到不分割字符直接合并
        if char in no_split_chars:
            current.append(char)
            current_len += 1
            continue

        # 遇到优先分割字符时立即分割
        if char in prefer_split_chars:
            current.append(char)
            segments.append(''.join(current))
            current = []
            current_len = 0
            continue

        # 长度超限时的智能处理
        if current_len >= max_seq_len:
            # 逆向查找最近的合适分割点
            split_pos = None
            # 优先查找优先分割字符
            for j in range(len(current) - 1, max(-1, len(current) - 20), -1):
                if current[j] in prefer_split_chars + secondary_split_chars:
                    split_pos = j + 1  # 包含分割字符
                    break
            # 次选查找次要分割字符
            if split_pos is None:
                for j in range(len(current) - 1, max(-1, len(current) - 10), -1):
                    if current[j] in secondary_split_chars:
                        split_pos = j + 1
                        break
            # 最后进行安全分割
            if split_pos is None:
                split_pos = min(len(current), max_seq_len)

            segments.append(''.join(current[:split_pos]))
            current = current[split_pos:]
            current_len = len(current)

        current.append(char)
        current_len += 1

    if current:
        segments.append(''.join(current))

    # 阶段二：合并短片段
    merged_segments = []
    buffer = []
    buffer_len = 0

    for seg in segments:
        seg_len = len(seg)
        if buffer_len + seg_len <= max_seq_len:
            buffer.append(seg)
            buffer_len += seg_len
        else:
            if buffer:
                merged_segments.append(''.join(buffer))
            buffer = [seg]
            buffer_len = seg_len
    if buffer:
        merged_segments.append(''.join(buffer))

    # 阶段三：强制长度限制
    final_segments = []
    for seg in merged_segments:
        while len(seg) > hard_max_len:
            # 查找最佳分割点
            split_pos = None
            for i in range(hard_max_len, max(hard_max_len - 20, 0), -1):
                if seg[i] in prefer_split_chars + secondary_split_chars:
                    split_pos = i + 1
                    break
            if split_pos is None:
                split_pos = hard_max_len
            final_segments.append(seg[:split_pos])
            seg = seg[split_pos:]
        final_segments.append(seg)

    return final_segments


def process_segments_v3(segments):
    """第三代处理函数，增强标点保留机制"""
    results = []
    punctuation_stack = []

    for seg in segments:
        # 提取结尾标点
        last_punc = re.findall(r'([。！？；：,.!?;:]$)', seg)
        if last_punc:
            punctuation_stack.append(last_punc[0])
        else:
            punctuation_stack.append(None)

    # 第二步：分块处理并保留标点
    for i, seg in enumerate(segments):
        # 跳过纯标点片段
        if re.fullmatch(r'^\W+$', seg):
            continue

        # 智能处理标点前缀
        clean_seg = re.sub(r'^[，。、]+', '', seg)
        if not clean_seg:
            continue

        # 生成
        input_tensor = Chinesetoid(clean_seg, word2id_dict)
        sequence = get_second(input_tensor, id2word_dict)

        # 标点处理（核心改进）
        target_punc = punctuation_stack[i]
        if target_punc:
            # 移除生成内容自带的标点
            sequence = re.sub(r'[。！？；：,.!?;:]$', '', sequence)
            sequence += target_punc
        else:
            sequence = re.sub(r'[,，]*$', '，', sequence)

        results.append(sequence)

    # 第三步：智能合并
    final_text = ''.join(results)

    # 后处理规则
    final_text = re.sub(r'，$', '。', final_text)  # 最后一个逗号改句号
    #final_text = re.sub(r'([。！？])([^。！？”’])', r'\1\n\2', final_text)  # 添加换行

    return final_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        insrc = data['input_text']
        max_seq_len = int(data.get('max_seq_len', 122))
        hard_max_len = int(data.get('hard_max_len', 50))

        # 参数验证保持不变
        if not (20 <= max_seq_len <= 200):
            raise ValueError("最大分段长度需在20-200之间")
        if not (20 <= hard_max_len <= 100):
            raise ValueError("智能分段限制需在20-100之间")
        if hard_max_len > max_seq_len:
            raise ValueError("智能分段限制不能大于最大分段长度")

        if len(insrc) <= 2:
            return jsonify({
                "error": "输入需大于2个字符",
                "input_text": insrc,
                "raw_result": None,
                "optimized_result": None,
                "show_result": False
            })

        # 处理逻辑保持不变
        segments = split_text_v3(insrc,
                                 max_seq_len=max_seq_len,
                                 hard_max_len=hard_max_len)
        raw_result = process_segments_v3(segments)
        raw_result = re.sub(r'([，。])\1+', r'\1', raw_result)
        raw_result = re.sub(r'([！？])[！？]+', r'\1', raw_result)

        # 调用 fix_text 函数获取优化后的结果
        optimized_result = fix_text(insrc, raw_result)

        return jsonify({
            "error": None,
            "input_text": insrc,
            "result": raw_result,       # 原始生成结果
            "optimized_result": optimized_result,  # 优化后的结果
            "show_result": True
        })

    except ValueError as ve:
        return jsonify({
            "error": f"参数错误: {str(ve)}",
            "input_text": insrc,
            "result": None,
            "optimized_result": None,
            "show_result": False
        })

    except Exception as e:
        return jsonify({
            "error": f"处理错误: {str(e)}",
            "input_text": insrc,
            "result": None,
            "optimized_result": None,
            "show_result": False
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


