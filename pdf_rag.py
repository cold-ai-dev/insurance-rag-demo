"""Retrieve from the chunks produced by prepare_pdf.py."""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from deepseek_api import answer

CHUNKS_FILE = Path(__file__).with_name("policy_chunks.json")
EVAL_FILE = Path(__file__).with_name("pdf_eval.json")


def bigrams(text):
    # 恢复被 PDF 排版空白隔开的汉字，避免漏掉词中的字符组合。
    text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text.lower())
    for part in re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", text):
        for i in range(len(part) - 1):
            yield part[i:i + 2]


def vectorize(counts, idf):
    weights = {
        term: (1 + math.log(count)) * idf[term]
        for term, count in counts.items()
        if term in idf
    }
    if not weights:
        return {}
    length = math.hypot(*weights.values())
    return {term: weight / length for term, weight in weights.items()}


class PdfRetriever:
    def __init__(self, path=CHUNKS_FILE):
        if not path.is_file():
            raise ValueError("找不到 policy_chunks.json，请先运行 prepare_pdf.py。")
        self.chunks = json.loads(path.read_text(encoding="utf-8"))
        if not self.chunks:
            raise ValueError("PDF 片段为空，请检查 prepare_pdf.py 的提取结果。")

        counts = [Counter(bigrams(chunk["text"])) for chunk in self.chunks]
        document_frequency = Counter(term for count in counts for term in count)
        # 多数片段都有的词权重更低，少数片段出现的词权重更高。
        self.idf = {
            term: 1 + math.log((len(self.chunks) + 1) / (frequency + 1))
            for term, frequency in document_frequency.items()
        }
        self.vectors = [vectorize(count, self.idf) for count in counts]

    def search(self, question, top_k=3):
        query = question.replace("住院住宿费", "床位费")
        query_vector = vectorize(Counter(bigrams(query)), self.idf)
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            # 两个向量都已归一化，点积就是余弦相似度。
            score = sum(weight * vector.get(term, 0) for term, weight in query_vector.items())
            if score > 0:
                results.append((score, chunk))
        results.sort(key=lambda result: result[0], reverse=True)
        return results[:top_k]


def evaluate(retriever, top_k):
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    known_ids = {chunk["id"] for chunk in retriever.chunks}
    marked_ids = {chunk_id for case in cases for chunk_id in case["gold"]}
    if marked_ids - known_ids:
        raise ValueError("评测中的片段 ID 已变化，请重新核对并标注 pdf_eval.json。")

    hits = []
    recalls = []
    precisions = []
    out_of_scope = []
    print(f"真实 PDF 检索评测 · K={top_k}\n")
    for case in cases:
        question = case["question"]
        gold = set(case["gold"])
        found = [chunk["id"] for _, chunk in retriever.search(question, top_k)]
        print(question)
        print(f"  取回：{'、'.join(found) or '无'}")

        if not gold:
            out_of_scope.append(not found)
            print(f"  资料外题：{'未返回候选' if not found else '返回了无关候选'}")
            continue

        found_ids = set(found)
        correct = len(found_ids & gold)
        hits.append(int(correct > 0))
        recalls.append(correct / len(gold))
        precisions.append(correct / top_k)
        print(f"  命中 {hits[-1]} · 召回 {recalls[-1]:.1%} · 查准 {precisions[-1]:.1%}")
        print(f"  漏掉：{'、'.join(sorted(gold - found_ids)) or '无'}"
              f" · 多带：{'、'.join(chunk_id for chunk_id in found if chunk_id not in gold) or '无'}")

    if hits:
        count = len(hits)
        print(f"\n{count} 道可回答题平均 · Hit@{top_k} {sum(hits) / count:.1%}"
              f" · Recall@{top_k} {sum(recalls) / count:.1%}"
              f" · Precision@{top_k} {sum(precisions) / count:.1%}")
    if out_of_scope:
        print(f"资料外题返回空候选：{sum(out_of_scope)}/{len(out_of_scope)}")


def main():
    # Some PDF glyphs cannot be printed by a redirected Windows GBK terminal.
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="查找 policy.pdf 中的证据。")
    parser.add_argument("question", nargs="?", help="想查的条款问题")
    parser.add_argument("--k", type=int, choices=range(1, 11), default=3, metavar="1..10")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="将问题和候选证据发给 DeepSeek")
    mode.add_argument("--eval", action="store_true", help="离线运行真实 PDF 检索评测")
    args = parser.parse_args()
    if args.eval and args.question is not None:
        parser.error("--eval 使用预先标注的题目，请省略问题。")

    try:
        retriever = PdfRetriever()
        if args.eval:
            evaluate(retriever, args.k)
            return

        question = (args.question if args.question is not None else input("输入条款问题：")).strip()
        if not question:
            parser.error("问题不能为空。")
        results = retriever.search(question, args.k)
        if not results:
            print("没有找到候选条款。请换成保单里的关键词再试。")
            return

        print(f"已从 {len(retriever.chunks)} 个 PDF 片段中检索到以下候选证据。")
        for score, chunk in results:
            print(f"\n[{chunk['id']}] {chunk['source']} · PDF 第 {chunk['page']} 页 · 相似度 {score:.3f}")
            print(chunk["text"])

        if args.generate:
            print("\n【DeepSeek 回答 · 核对原文】")
            print(answer(question, [chunk for _, chunk in results]))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
