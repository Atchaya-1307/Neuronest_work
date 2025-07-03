from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util
import json
import os

# -------------------------------------------
# Groundedness Checker Class
# -------------------------------------------
class GroundednessChecker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(" Loading groundedness model...")
        self.model = SentenceTransformer(model_name)

    def check_groundedness(self, context: str, answer: str) -> dict:
        context_embedding = self.model.encode(context, convert_to_tensor=True)
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)

        score = util.cos_sim(context_embedding, answer_embedding).item()

        explanation = self._explain(score)
        return {
            "score": round(score, 4),
            "explanation": explanation
        }

    def _explain(self, score):
        if score > 0.85:
            return " Highly grounded in the retrieved content."
        elif score > 0.65:
            return " Moderately grounded. Consider refining the response."
        else:
            return " Weakly grounded or off-topic."

# -------------------------------------------
# Load JSON Data and Build FAISS Index
# -------------------------------------------
json_path = r"C:\Users\ahaly\OneDrive\Desktop\Neuronest_work\Arbitration.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

ACT_NAME = "Arbitration Act"
docs = []
for chapter in data:
    act_name = chapter.get("actName", ACT_NAME)
    chapter_number = chapter.get("chapterNumber", "Unknown Chapter")
    chapter_title = chapter.get("chapterTitle", "Untitled Chapter")

    for section in chapter.get("sections", []):
        section_number = section.get("sectionNumber", "Unknown Section")
        section_content = section.get("sectionContent", "No Content Available")

        text_to_embed = (
            f"Act: {act_name}\n"
            f"Chapter {chapter_number}: {chapter_title}\n"
            f"Section {section_number}: {section_content}"
        )

        doc = Document(
            page_content=text_to_embed,
            metadata={
                "act": act_name,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "section_number": section_number,
            },
        )
        docs.append(doc)

print(f"\n Loaded {len(docs)} sections from JSON.")

# Embeddings and FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
save_dir = "faiss_index"
vectorstore.save_local(save_dir)
print(f"✅ FAISS index saved to: {os.path.abspath(save_dir)}")

# -------------------------------------------
# Search Loop + Groundedness
# -------------------------------------------
checker = GroundednessChecker()

print("\n You can now search! Type your query below.")
while True:
    query = input("\nEnter your query (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    results = vectorstore.similarity_search(query, k=3)
    if not results:
        print(" No results found.")
    else:
        for i, res in enumerate(results, 1):
            meta = res.metadata
            print(f"\n📄 Result {i}:")
            print(f"Act: {meta['act']}")
            print(f"Chapter {meta['chapter_number']} - {meta['chapter_title']}")
            print(f"Section {meta['section_number']}")
            print("Content:")
            print(res.page_content)
            print("-" * 50)

        # Perform groundedness check with top result
        print("\n🔗 Groundedness Check (Top Result):")
        top_content = results[0].page_content
        result = checker.check_groundedness(top_content, query)
        print(f"Score: {result['score']}")
        print(f"Explanation: {result['explanation']}")
