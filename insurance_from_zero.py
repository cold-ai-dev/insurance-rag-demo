import sys


clauses = [
    {
        "product": "594-1",
        "ref": "1.6.2",
        "text": "床位费以单人病房标准为上限。",
    },
    {
        "product": "2612-1",
        "ref": "1.2.3",
        "text": "床位费以双人病房标准为上限，单人病房除外。",
    },
    {
        "product": "2612-1",
        "ref": "1.2.1/规则",
        "text": "一般疾病等待期为主险合同生效起90日。",
    },
    {
        "product": "2612-1",
        "ref": "1.2.1/例外",
        "text": "等待期例外包括意外伤害、保证续保期间内续保，以及上一个保证续保期间届满后60日内重新投保。",
    },
]


def bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)}


def retrieve(product, question, top_k=1):
    search_query = question.replace("住院住宿费", "床位费")
    query_terms = bigrams(search_query)
    matches = []

    for clause in clauses:
        if clause["product"] != product:
            continue

        score = len(query_terms & bigrams(clause["text"]))
        if score > 0:
            matches.append((score, clause))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [clause for score, clause in matches[:top_k]]


def evaluate():
    tests = [
        ("2612-1", "住院住宿费怎么算？", "1.2.3"),
        ("2612-1", "酒店住宿费怎么算？", None),
        ("594-1", "住院住宿费怎么算？", "1.6.2"),
        ("未知版本", "床位费怎么算？", None),
    ]

    for product, question, expected in tests:
        found = retrieve(product, question)
        actual = found[0]["ref"] if found else None
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} | {product} | {question} | 预期 {expected}，实际 {actual}")

    # 床位费只是背景，标注的正确证据只包含等待期规则和例外。
    question = "背景提到床位费。只核对疾病等待期为多久，以及有哪些等待期例外？"
    expected_refs = {"1.2.1/规则", "1.2.1/例外"}

    for k in (1, 2, 3):
        found = retrieve("2612-1", question, top_k=k)
        found_refs = {clause["ref"] for clause in found}
        correct = len(found_refs & expected_refs)
        hit = int(correct > 0)
        recall = correct / len(expected_refs)
        precision = correct / k
        refs = "、".join(clause["ref"] for clause in found)
        print(f"K={k} | Hit={hit} | Recall={recall:.0%} | Precision@K={precision:.1%}")
        print(f"  取回条款：{refs}")


def ask():
    product = input("输入保单条款版本：").strip()
    question = input("输入客户的完整问题：").strip()
    found = retrieve(product, question, top_k=2)

    if not found:
        print("没找到条款，请核对版本和问题。")
        return

    evidence = []
    for clause in found:
        evidence.append(f"[{clause['product']} / {clause['ref']}] {clause['text']}")

    context = "\n".join(evidence)
    prompt = f"""你是医疗险坐席助手。只根据下方条款回答，并标注出处。
证据不足时，说明还需要核对什么。个案赔付交由人工核定。

客户问题：{question}

条款依据：
{context}
"""
    print(prompt)


if "--eval" in sys.argv:
    evaluate()
else:
    ask()
