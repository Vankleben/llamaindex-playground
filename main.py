from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 1) 读取数据
    documents = SimpleDirectoryReader("data").load_data()

    # 2) 建索引
    index = VectorStoreIndex.from_documents(documents)

    # 3) 检索
    retriever = index.as_retriever(similarity_top_k=5)
    query = "What is LlamaIndex?"
    nodes = retriever.retrieve(query)   # ✅ nodes 在这里定义

    # 4) 打印（分数+来源+前200字）
    print("\n=== Retrieved Chunks ===")
    for i, n in enumerate(nodes, 1):
        node = getattr(n, "node", n)  # 兼容不同版本
        score = getattr(n, "score", None)
        md = getattr(node, "metadata", {}) or {}
        source = md.get("file_path") or md.get("source") or "unknown"
        text = (getattr(node, "text", "") or "")[:200].replace("\n", " ")
        print(f"{i}. score={score} source={source}\n   {text}\n")


if __name__ == "__main__":
    main()