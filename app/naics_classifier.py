import json
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load the NAICS data from a local JSON file
with open("./data/naics_2022.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Extract the list of codes from the "naicsCodes" key
naics_data = data["naicsCodes"]

# 3. Build a list of NAICS “documents” (title + description) and keep track of codes
naics_docs = []
naics_codes = []
for entry in naics_data:
    combined_text = f"{entry['title']} {entry['description']}"
    naics_docs.append(combined_text)
    naics_codes.append(entry["code"])

# 4. Initialize a Sentence Transformer model
#    'all-MiniLM-L6-v2' is a common lightweight model. 
#    You can choose a larger model for potentially higher accuracy.
model_name = 'all-MiniLM-L6-v2'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = SentenceTransformer(model_name, device=device)

# 5. Precompute embeddings for each NAICS document
naics_embeddings = model.encode(naics_docs, convert_to_tensor=True)

# 6. Define your tags (query terms) and build a single “query string”
tags = [
    "cloud-computing",
    "software-development",
    "it-services-and-it-consulting",
    "it-system-custom-software-development",
    "it-system-data-services",
    "it-system-design-services",
    "it-system-installation-and-disposal"
]

query_string = " ".join(tags)

# 7. Compute the embedding for the query
query_embedding = model.encode(query_string, convert_to_tensor=True)

# 8. Compute cosine similarities
similarities = util.cos_sim(query_embedding, naics_embeddings)[0]  # Shape: [num_documents]

# 9. Sort indices in descending order of similarity
sorted_indices = similarities.argsort(descending=True)

# 10. Print out the top matches
print("Top NAICS code matches for the given tags:\n")
for rank, idx in enumerate(sorted_indices[:10], start=1):
    # Convert similarity to a float for printing
    sim_score = float(similarities[idx])
    # Show only partial text for brevity
    partial_doc = naics_docs[idx].replace("\n", " ")[:80] + "..."
    print(f"Rank {rank}:")
    print(f"  NAICS Code: {naics_codes[idx]}")
    print(f"  Document (partial): {partial_doc}")
    print(f"  Similarity Score: {sim_score:.4f}\n")
