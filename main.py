import os
import time
import json
import random
import numpy as np

import torch
from jedi.debug import reset_time

from src.config import AllConfig
from src.data import DataManager
from src.path import LLM_PATH_QWEN, EMBEDDING_PATH_BGE_BASE_EN
from src.utils import get_parameters_range
from src.path import SAVE_DIR
from src.HippoRAG.src.hipporag.HippoRAG import HippoRAG

os.environ["DEVICE"] = "cuda:0"
params = {
    'dataset_name': "hotpotqa", # ["2wikimultihopqa", "hotpotqa", "musique", "narrativeqa"],边权范围[1-8, 1-10, 1-18, 1-72]
    'samples_num': 1000, # [1000, 1000, 1000, 300],
    'seed': 11,
}
params['samples_num'] = 1000 if params['dataset_name'] != 'narrativeqa' else 300
random.seed(params['seed'])
np.random.seed(params['seed'])

def main():
    print("-----------------------------------------")
    data_manager = DataManager(dataset_name=params['dataset_name'], samples_num=params['samples_num'])
    all_config = AllConfig(
        dataset_name = data_manager.dataset_name,       # 数据集名称：# 2wikimultihopqa hotpotqa musique narrativeqa
        llm_base_url = 'http://0.0.0.0:8000/v1',           # VLLM调用url，使用本地模型时该参数无用
        llm_name = LLM_PATH_QWEN,                       # 用于openie和对话的模型路径（加载本地模型用路劲）
        embedding_name = EMBEDDING_PATH_BGE_BASE_EN,    # 用于嵌入的模型路径（加载本地模型用路劲）
        openie_mode = 'online',                         # OpenIE 运行模式。在线模式：实时处理数据。离线模式：使用 VLLM 批量模式进行索引 choices=['online', 'offline']
        seed = params['seed'],                          # 随机种子
        corpus_len = data_manager.get_assign_corpus_len(increment_id=0),
        temperature=0.0,
    )
    print('--Range of parameters to be optimized :\n', get_parameters_range())
    data_manager.print_shape()

    hipporag = HippoRAG(
        global_config=all_config.global_config,
        save_dir=os.path.join(SAVE_DIR, all_config.dataset_name)
    )

    res = []
    res_time = []

    for i in range(3):
        a,b,c = hipporag.ec_index(data_manager.docs[i], params['samples_num'], i)

        time1 = time.time()
        hipporag.ec_create_graph(a,b,c)
        time2 = time.time()

        hipporag.ready_to_retrieve = False
        _, _, _, recall, f1 = hipporag.rag_qa(
            queries=data_manager.test_all_queries[i],
            gold_docs=data_manager.test_gold_docs[i],
            gold_answers=data_manager.test_gold_answers[i],
        )  # Retrieval and QA
        time3 = time.time()

        print(f"cache命中次数 / 访问次数 : {hipporag.llm_model.cache_hits} / {2*len(data_manager.test_all_queries[i])}")

        print("index花费时间",time2 - time1)
        print("rag_qa花费时间",time3 - time2)
        print(recall, f1)
        res.append([recall, f1])
        res_time.append([time2 - time1, time3 - time2])

    # with open(f"./outputs/test/{data_manager.dataset_name}_res_data.json", "w") as file:
    #     json.dump(res, file, indent=4)
    # with open(f"./outputs/test/{data_manager.dataset_name}_res_time_data.json", "w") as file:
    #     json.dump(res_time, file, indent=4)


    # for i in range(3):
    #     i=2
    #     start_time = time.time()
    #     chunk_ids, chunk_triples, chunk_triple_entities = hipporag.init_embedding(
    #         docs=data_manager.docs[i],
    #         samples_num=params['samples_num'],
    #         increment_i=i
    #     )
    #     hipporag.filter_syn_edges(chunk_ids, chunk_triples, chunk_triple_entities)
    #
    #     hipporag.ready_to_retrieve = False
    #     _, _, _, recall, f1 = hipporag.rag_qa(
    #         queries=data_manager.test_all_queries[2],
    #         gold_docs=data_manager.test_gold_docs[2],
    #         gold_answers=data_manager.test_gold_answers[2]
    #     )  # Retrieval and QA
    #     end_time = time.time()
    #     print(f"reall_f1 : {recall['Recall@5']}, {f1['F1']}")
    #     print(f"time : {recall['Recall@5']}, {f1['F1']}")
    #     break

#
# from src.utils import get_parameters_range
# print('--Range of parameters to be optimized :\n', get_parameters_range())
# print('--default T_syn :', all_config.global_config.synonymy_edge_sim_threshold)
# print('--default W_reset :', all_config.global_config.passage_node_weight)
# print('--default F_damp :', all_config.global_config.damping)
# print('--default embedding_max_seq_len :', all_config.global_config.embedding_max_seq_len)
# print('--default embedding_batch_size :', all_config.global_config.embedding_batch_size)
# print("--norm :", all_config.global_config.embedding_return_as_normalized)
# print("--data_manager :")
# data_manager.print_shape()
#
# ### 初始化 HippoRAG2
# from src.path import SAVE_DIR
# from src.HippoRAG.src.hipporag.HippoRAG import HippoRAG
#
#  # 初始化HippoRAG系统
# hipporag = HippoRAG(
#     global_config=all_config.global_config,
#     save_dir=os.path.join(SAVE_DIR, all_config.dataset_name)
#     # save_dir=os.path.join(RESULTS_SAVE_DIR)
# )
#
# chunk_ids, chunk_triples, chunk_triple_entities = hipporag.init_embedding(
#     docs=data_manager.docs[2],
#     samples_num=samples_num,
#     increment_i=2
# )
# print('---------wwaeawewae')


if __name__ == "__main__":
    """程序入口点判断"""
    main()
#
# def count_relations_number(chunk_triples):
#     triples = []
#     for i in chunk_triples:
#         triples.extend(i)
#     print(len(triples))
#     from collections import Counter
#     # triples 是 (h, r, t) 的 list
#     relation_counter = Counter([r for (_, r, _) in triples])
#     # relation_counter 是字典 {relation: count}
#     print("总共有多少种关系：", len(relation_counter))
#     # 输出出现次数最多的前 20 个 relation
#     print("Top-20 relations by frequency:")
#     aaa = 0
#     for rel, count in relation_counter.most_common(10):
#         print(f"{rel}: {count}")
#         aaa += count
#     print('总共有多少种关系213213123123', aaa, aaa / len(triples))
#     # 如果只取前10个关系做优化
#     top_relations = [rel for rel, _ in relation_counter.most_common(10)]
#     print("用于优化的 Top-10 relations:", top_relations)

## #测试余弦相似度
#     texts_to_encode = ['machine learning','deep learning','weather forecast']
#     embeddings = hipporag.embedding_model.batch_encode(texts_to_encode)
#     print(embeddings)
#
#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity
#     vectors = embeddings
#     # 计算所有向量之间的余弦相似度
#     similarity_matrix = cosine_similarity(vectors)
#     # 打印结果
#     print("余弦相似度矩阵:")
#     print(similarity_matrix)
#     print("\n解释:")
#     print(f"词语1与词语2的相似度: {similarity_matrix[0, 1]:.4f}")
#     print(f"词语1与词语3的相似度: {similarity_matrix[0, 2]:.4f}")
#     print(f"词语2与词语3的相似度: {similarity_matrix[1, 2]:.4f}")
