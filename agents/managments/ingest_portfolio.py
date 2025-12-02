import re
import json
import traceback
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "agents/chroma_db"
JSON_FILE = BASE_DIR / "advanced_portfolio_data.json"
COLLECTION_NAME = "project_portfolio"

llm = ChatOpenAI(model="gpt-4o-mini")

# FIXED: Use consistent embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072 dimensions


def build_taxonomy_chunks_from_project_json(project_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate 5 taxonomy chunks from project JSON"""
    
    prompt = f"""
    You are a Top 1% AI Engineer. Analyze ONLY the provided Project Portfolio JSON.

    📌 **STRICT RULES:**
    1. Use **only** information found in `project_json`.  
    2. Do NOT infer or create imaginary details.  
    3. Reference actual project names, descriptions, technologies from JSON.
    4. Keep writing concise and factual.

    ## 🎯 **Generate 5 Master Taxonomies:**
    1. **Technical_Capability**  
    2. **Domain_Expertise**  
    3. **Business_Impact_Trust**  
    4. **Engagement_Hiring**  
    5. **Process_Communication**

    ## ✅ **OUTPUT FORMAT:**
    [
    {{
        "content": "<One paragraph strictly based on project_json>",
        "metadata": {{
        "category": "<One of the 5 taxonomy categories>",
        "sub_type": "<Short label extracted from JSON>",
        "keywords": ["List", "Of", "Exact", "Phrases"],
        "project_ref": "<Project name from JSON>"
        }}
    }},
    ...
    ]

    ## 📌 **Input Project Portfolio:**
    {json.dumps(project_json, indent=2)}

    Generate the taxonomy chunks now.
    """

    try:
        llm_response = llm.invoke(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM invocation failed: {e}")

    raw = llm_response.content.strip()

    # Extract JSON
    json_text = None
    array_match = re.search(r"(\[\s*\{.*\}\s*\])", raw, flags=re.DOTALL)
    if array_match:
        json_text = array_match.group(1)
    else:
        obj_match = re.search(r"(\{\s*\"content\".*\}\s*)", raw, flags=re.DOTALL)
        if obj_match:
            json_text = f"[{obj_match.group(1)}]"

    if not json_text:
        try:
            parsed_try = json.loads(raw)
            taxonomy_list = [parsed_try] if isinstance(parsed_try, dict) else parsed_try
        except:
            raise ValueError(f"Could not parse JSON from LLM output:\n{raw}")
    else:
        try:
            parsed = json.loads(json_text)
            taxonomy_list = [parsed] if isinstance(parsed, dict) else parsed
        except json.JSONDecodeError:
            # Cleanup and retry
            cleaned = json_text.replace(""", "\"").replace(""", "\"")
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                parsed = json.loads(cleaned)
                taxonomy_list = [parsed] if isinstance(parsed, dict) else parsed
            except:
                raise ValueError(f"Failed to parse JSON:\n{json_text}")

    # Validate
    required_categories = {
        "Technical_Capability",
        "Domain_Expertise",
        "Business_Impact_Trust",
        "Engagement_Hiring",
        "Process_Communication",
    }

    seen_categories = set()
    validated_chunks: List[Dict[str, Any]] = []

    def _validate_item(item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("Each entry must be a dict")

        if "content" not in item or "metadata" not in item:
            raise ValueError("Missing 'content' or 'metadata'")

        content = item["content"]
        metadata = item["metadata"]

        if not isinstance(content, str) or not content.strip():
            raise ValueError("'content' must be non-empty string")

        if not isinstance(metadata, dict):
            raise ValueError("'metadata' must be dict")

        for k in ("category", "sub_type", "keywords", "project_ref"):
            if k not in metadata:
                raise ValueError(f"Missing metadata.{k}")

        category = metadata["category"]
        if category not in required_categories:
            raise ValueError(f"Invalid category: {category}")

        return {
            "content": content.strip(),
            "metadata": {
                "category": category,
                "sub_type": metadata["sub_type"].strip(),
                "keywords": [k.strip() for k in metadata["keywords"] if k.strip()],
                "project_ref": metadata["project_ref"].strip(),
            },
        }

    for item in taxonomy_list:
        validated = _validate_item(item)
        cat = validated["metadata"]["category"]
        seen_categories.add(cat)
        validated_chunks.append(validated)

    missing = required_categories - seen_categories
    if missing:
        raise ValueError(f"Missing categories: {sorted(list(missing))}")

    return validated_chunks


def convert_to_documents(taxonomy_chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert taxonomy chunks to LangChain Documents"""
    documents = []
    
    for chunk in taxonomy_chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "category": chunk["metadata"]["category"],
                "sub_type": chunk["metadata"]["sub_type"],
                "keywords": ", ".join(chunk["metadata"]["keywords"]),
                "project_ref": chunk["metadata"]["project_ref"],
                "timestamp": datetime.now().isoformat(),
            }
        )
        documents.append(doc)
    
    return documents


def upsert_documents_to_vectorstore(documents: List[Document]) -> None:
    """Upsert documents to Chroma vector store with dimension matching"""
    if not documents:
        print("❌ No documents to upsert")
        return

    try:
        print(f"\n💾 Upserting {len(documents)} documents to vector store...")

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DB_PATH),
        )

        vectorstore.delete_collection()
        
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,  # Uses text-embedding-3-large (3072)
            persist_directory=str(CHROMA_DB_PATH),
        )
        
        # Add documents - Chroma handles embeddings with correct dimension
        vectorstore.add_documents(documents=documents)
        
        print(f"✅ Successfully upserted {len(documents)} documents!")
        print(f"📊 Collection: {COLLECTION_NAME}")
        print(f"📁 Directory: {CHROMA_DB_PATH}")
        print(f"🔢 Embedding dimension: 3072 (text-embedding-3-large)")
        
    except Exception as e:
        print(f"❌ Error upserting documents: {e}")
        print(traceback.format_exc())
        raise


def load_and_process_data() -> None:
    """Load JSON, generate taxonomy, convert to docs, upsert to vector store"""
    print(f"🚀 Starting Advanced Portfolio Extraction...")
    
    # Load JSON
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            portfolio_json = json.load(f)
        print(f"✅ Loaded portfolio data from {JSON_FILE}")
    except FileNotFoundError:
        print(f"❌ JSON file not found: {JSON_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return

    # Generate taxonomy
    try:
        print("\n🤖 Generating taxonomy chunks with LLM...")
        taxonomy_chunks = build_taxonomy_chunks_from_project_json(portfolio_json)
        print(f"✅ Generated {len(taxonomy_chunks)} taxonomy chunks")
        print("\n📋 Taxonomy Chunks:")
        print(json.dumps(taxonomy_chunks, indent=2))
    except Exception as e:
        print(f"❌ Error generating taxonomy: {e}")
        print(traceback.format_exc())
        return

    # Convert to documents
    try:
        print("\n📄 Converting chunks to documents...")
        documents = convert_to_documents(taxonomy_chunks)
        print(f"✅ Converted {len(documents)} chunks to documents")
        
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(f"  Content: {doc.page_content[:80]}...")
            print(f"  Metadata: {doc.metadata}")
    except Exception as e:
        print(f"❌ Error converting chunks: {e}")
        print(traceback.format_exc())
        return

    # Upsert to vector store
    try:
        upsert_documents_to_vectorstore(documents)
    except Exception as e:
        print(f"❌ Error upserting to vector store: {e}")
        return

    print("\n✅ Portfolio processing complete!")



load_and_process_data()