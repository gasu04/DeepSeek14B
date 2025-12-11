#!/usr/bin/env python3
"""
Google Drive RAG with DeepSeek R1 14B

This script:
1. Connects to Google Drive using your credentials
2. Downloads and processes documents (PDF, DOCX, TXT, MD, TEX)
3. Creates embeddings using Jina AI cloud embeddings
4. Stores in ChromaDB vector database
5. Allows querying with DeepSeek R1 14B

Requirements:
- credentials.json in the same directory
- Ollama running with deepseek-r1:14b
- Jina API key (cloud embeddings)
"""

import os
import io
import requests
from pathlib import Path
from typing import List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_ollama import OllamaLLM
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Jina Embeddings API wrapper
class JinaEmbeddings(Embeddings):
    """Jina AI Embeddings API wrapper for LangChain"""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v2-base-en"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": texts
        }

        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

class GoogleDriveRAG:
    def __init__(self, credentials_path="credentials.json"):
        self.credentials_path = credentials_path
        self.token_path = "token.json"
        self.download_dir = Path("./gdrive_downloads")
        self.download_dir.mkdir(exist_ok=True)
        # Base directory for all embeddings on 990PRO 4T SSD
        self.embeddings_base_dir = Path("/Volumes/990PRO 4T/DeepSeek/chroma_embeddings")
        self.embeddings_base_dir.mkdir(exist_ok=True)
        self.vector_db_path = None  # Will be set when selecting/creating embedding
        self.current_folder_name = None  # Track which folder we're working with

        print("🚀 Initializing Google Drive RAG System...\n")

        # Initialize components
        self.drive_service = self._authenticate_gdrive()
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vectorstore = None

    def _authenticate_gdrive(self):
        """Authenticate with Google Drive"""
        print("🔐 Authenticating with Google Drive...")
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        service = build('drive', 'v3', credentials=creds)
        print("✓ Google Drive authenticated\n")
        return service

    def _init_embeddings(self):
        """Initialize Jina AI embeddings"""
        print("📊 Loading Jina AI Embeddings (jina-embeddings-v2-base-en)...")

        # Get API key from environment or prompt user
        api_key = os.getenv('JINA_API_KEY')
        if not api_key:
            print("   ⚠️  JINA_API_KEY not found in environment")
            api_key = input("   Enter your Jina API key: ").strip()
            if not api_key:
                raise ValueError("Jina API key is required!")

        embeddings = JinaEmbeddings(
            api_key=api_key,
            model="jina-embeddings-v2-base-en"
        )

        # Test the connection works
        print("   Testing Jina API connection...")
        try:
            test_result = embeddings.embed_query("connection test")
            print(f"   ✓ Connection test passed ({len(test_result)} dimensions)")
            print(f"   ✓ Using Jina AI cloud embeddings\n")
        except Exception as e:
            print(f"   ❌ Connection test failed: {e}\n")
            raise

        return embeddings

    def _init_llm(self):
        """Initialize DeepSeek model"""
        print("🤖 Loading DeepSeek R1 14B...")
        llm = OllamaLLM(
            model="deepseek-r1:14b",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        print("✓ DeepSeek loaded\n")
        return llm

    def list_files(self, folder_id=None, file_types=None):
        """List files from Google Drive"""
        print("📂 Listing files from Google Drive...")

        query_parts = []
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        if file_types:
            mime_queries = []
            for ft in file_types:
                if ft == 'pdf':
                    mime_queries.append("mimeType='application/pdf'")
                elif ft == 'docx':
                    mime_queries.append("mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'")
                    # Also include regular .doc files
                    mime_queries.append("mimeType='application/msword'")
                    # Include Google Docs
                    mime_queries.append("mimeType='application/vnd.google-apps.document'")
                elif ft == 'txt':
                    mime_queries.append("mimeType='text/plain'")
                elif ft == 'md':
                    mime_queries.append("mimeType='text/markdown'")
                elif ft == 'tex':
                    mime_queries.append("mimeType='application/x-latex'")
                    mime_queries.append("mimeType='text/x-tex'")
                    # Also match as plain text since some systems store .tex as text/plain
                    mime_queries.append("(mimeType='text/plain' and name contains '.tex')")
            if mime_queries:
                query_parts.append(f"({' or '.join(mime_queries)})")

        query_parts.append("trashed=false")
        # Exclude folders from results
        query_parts.append("mimeType!='application/vnd.google-apps.folder'")
        query = ' and '.join(query_parts)

        print(f"   Query: {query[:100]}...")

        results = self.drive_service.files().list(
            q=query,
            pageSize=100,
            fields="files(id, name, mimeType, size)"
        ).execute()

        files = results.get('files', [])

        # Debug: Show what we found
        if len(files) == 0:
            print("   ⚠️  No files found with specified types")
            print("   Let me check ALL files in this folder...\n")

            # Try listing ALL files to see what's there
            debug_query = f"'{folder_id}' in parents and trashed=false" if folder_id else "trashed=false"
            debug_results = self.drive_service.files().list(
                q=debug_query,
                pageSize=20,
                fields="files(name, mimeType)"
            ).execute()

            debug_files = debug_results.get('files', [])
            if debug_files:
                print(f"   Found {len(debug_files)} items total (any type):")
                for f in debug_files[:10]:
                    mime = f.get('mimeType', 'unknown')
                    print(f"      • {f['name']} ({mime})")
                if len(debug_files) > 10:
                    print(f"      ... and {len(debug_files) - 10} more")
                print()
            else:
                print("   No items at all in this folder (it's truly empty)\n")

        print(f"✓ Found {len(files)} matching files\n")
        return files

    def list_folders(self):
        """List ROOT-LEVEL folders only from Google Drive (exact names as shown in web interface)"""
        print("📁 Listing ROOT-LEVEL folders from Google Drive...")
        print("   (Only showing folders at the top level, not subfolders)")

        # Only get folders that are direct children of root (My Drive)
        query = "mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false"

        # Get all folders (paginated if more than 1000)
        all_folders = []
        page_token = None

        try:
            while True:
                results = self.drive_service.files().list(
                    q=query,
                    pageSize=1000,  # Maximum allowed by API
                    fields="nextPageToken, files(id, name)",
                    orderBy="name",
                    pageToken=page_token
                ).execute()

                folders = results.get('files', [])
                all_folders.extend(folders)

                if len(all_folders) > 0 and len(all_folders) % 100 == 0:
                    print(f"   ...found {len(all_folders)} so far...")

                page_token = results.get('nextPageToken')
                if not page_token:
                    break

            print(f"✓ Found {len(all_folders)} folder(s)\n")
            return all_folders

        except Exception as e:
            print(f"\n❌ Error listing folders: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            print("\n💡 Troubleshooting:")
            print("   1. Check your internet connection")
            print("   2. Verify Google Drive API is enabled")
            print("   3. Try refreshing authentication (delete token.json)")
            return []

    def get_folder_by_name(self, folder_name):
        """Get folder ID by folder name"""
        folders = self.list_folders()

        # Exact match
        for folder in folders:
            if folder['name'].lower() == folder_name.lower():
                return folder['id'], folder['name']

        # Partial match
        matches = [f for f in folders if folder_name.lower() in f['name'].lower()]
        if len(matches) == 1:
            return matches[0]['id'], matches[0]['name']
        elif len(matches) > 1:
            print(f"\n⚠️  Multiple folders match '{folder_name}':")
            for i, f in enumerate(matches, 1):
                print(f"  {i}. {f['name']}")
            return None, None

        return None, None

    def select_folder_interactively(self):
        """Let user select a folder interactively"""
        folders = self.list_folders()

        if not folders:
            print("⚠️  No folders found in Google Drive.")
            print("   Only showing FOLDERS (files are not listed here)")
            return None

        print("📂 ROOT-LEVEL FOLDERS ONLY (names match Google Drive exactly):")
        print("   (Not showing nested subfolders)")
        print("=" * 60)
        print("  0. 📁 [Entire Drive - All files from all folders]")
        for i, folder in enumerate(folders, 1):
            print(f"  {i}. 📁 {folder['name']}")
        print("=" * 60)

        print("\n💡 Selection options:")
        print("  • Enter a number (e.g., 1, 2, 3)")
        print("  • Type folder name (exact or partial)")
        print("  • Press Enter for root (all Drive)")

        while True:
            choice = input("\n📁 Your choice: ").strip()

            if not choice or choice == "0":
                print("✓ Selected: ROOT folder (entire Drive)\n")
                return None

            # Try as number
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(folders):
                    selected = folders[idx - 1]
                    print(f"✓ Selected folder: '{selected['name']}'\n")
                    return selected['id']
                else:
                    print(f"❌ Invalid number. Choose 0-{len(folders)}")
                    continue

            # Try as folder name
            folder_id, folder_name = self.get_folder_by_name(choice)
            if folder_id:
                print(f"✓ Selected folder: '{folder_name}'\n")
                return folder_id
            else:
                print(f"❌ Folder '{choice}' not found.")
                print("   Tip: Names are case-insensitive, partial match works")
                print("   Or press Enter for root folder")

    def list_existing_embeddings(self):
        """List all existing embeddings"""
        if not self.embeddings_base_dir.exists():
            return []

        embeddings = []
        for item in self.embeddings_base_dir.iterdir():
            if item.is_dir():
                # Check if it's a valid ChromaDB directory
                if (item / "chroma.sqlite3").exists():
                    embeddings.append({
                        'name': item.name,
                        'path': item,
                        'created': item.stat().st_ctime
                    })

        # Sort by creation time (newest first)
        embeddings.sort(key=lambda x: x['created'], reverse=True)
        return embeddings

    def select_existing_embedding(self):
        """Let user select an existing embedding"""
        embeddings = self.list_existing_embeddings()

        if not embeddings:
            print("⚠️  No existing embeddings found.")
            return None

        print("\n📚 EXISTING EMBEDDINGS:")
        print("=" * 60)
        for i, emb in enumerate(embeddings, 1):
            from datetime import datetime
            created_date = datetime.fromtimestamp(emb['created']).strftime('%Y-%m-%d %H:%M')
            print(f"  {i}. 📁 {emb['name']} (created: {created_date})")
        print("=" * 60)

        while True:
            choice = input("\n📁 Select embedding (number or name): ").strip()

            if not choice:
                print("❌ Please select an embedding.")
                continue

            # Try as number
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(embeddings):
                    selected = embeddings[idx - 1]
                    print(f"✓ Selected embedding: '{selected['name']}'\n")
                    return selected['name'], selected['path']
                else:
                    print(f"❌ Invalid number. Choose 1-{len(embeddings)}")
                    continue

            # Try as name (exact or partial match)
            matches = [e for e in embeddings if choice.lower() in e['name'].lower()]
            if len(matches) == 1:
                selected = matches[0]
                print(f"✓ Selected embedding: '{selected['name']}'\n")
                return selected['name'], selected['path']
            elif len(matches) > 1:
                print(f"⚠️  Multiple embeddings match '{choice}':")
                for i, e in enumerate(matches, 1):
                    print(f"  {i}. {e['name']}")
                print("Please be more specific.")
            else:
                print(f"❌ Embedding '{choice}' not found.")

    def download_file(self, file_id, file_name, mime_type=None):
        """Download or export a file from Google Drive"""

        # Check if it's a Google Docs file (needs export, not download)
        google_doc_types = {
            'application/vnd.google-apps.document': ('text/plain', '.txt'),
            'application/vnd.google-apps.spreadsheet': ('text/csv', '.csv'),
            'application/vnd.google-apps.presentation': ('text/plain', '.txt'),
        }

        if mime_type in google_doc_types:
            # Export Google Doc to text format
            export_mime, extension = google_doc_types[mime_type]
            file_path = self.download_dir / f"{file_name}{extension}"

            request = self.drive_service.files().export_media(
                fileId=file_id,
                mimeType=export_mime
            )

            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.close()
            return file_path

        else:
            # Regular file download
            request = self.drive_service.files().get_media(fileId=file_id)
            file_path = self.download_dir / file_name

            fh = io.FileIO(file_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.close()
            return file_path

    def load_document(self, file_path):
        """Load document based on file type"""
        ext = file_path.suffix.lower()

        try:
            if ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif ext == '.docx':
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif ext in ['.txt', '.md', '.tex']:
                loader = TextLoader(str(file_path))
            else:
                print(f"⚠️  Unsupported file type: {ext}")
                return []

            return loader.load()
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {str(e)}")
            return []

    def process_documents(self, folder_id=None, folder_name=None, file_types=['pdf', 'docx', 'txt', 'md', 'tex']):
        """Download and process documents from Google Drive"""
        print("=" * 60)
        print("📥 Processing documents from Google Drive...")
        print("=" * 60 + "\n")

        # Set folder name for embedding storage
        if folder_name:
            self.current_folder_name = folder_name
        else:
            self.current_folder_name = "entire_drive"

        # Set vector database path based on folder name
        self.vector_db_path = str(self.embeddings_base_dir / self.current_folder_name)
        print(f"📁 Embedding will be saved as: '{self.current_folder_name}'\n")

        # List files
        files = self.list_files(folder_id, file_types)

        if not files:
            print("⚠️  No files found!")
            return

        # Display files
        print("Files to process:")
        for i, f in enumerate(files, 1):
            size_mb = int(f.get('size', 0)) / (1024 * 1024)
            print(f"  {i}. {f['name']} ({size_mb:.2f} MB)")
        print()

        # Download and load documents
        all_documents = []
        for file in files:
            print(f"📄 Processing: {file['name']}...")

            try:
                # Download or export (handles Google Docs)
                mime_type = file.get('mimeType')
                file_path = self.download_file(file['id'], file['name'], mime_type)
                print(f"  ✓ Downloaded/exported to {file_path}")

                # Load
                docs = self.load_document(file_path)
                if docs:
                    all_documents.extend(docs)
                    print(f"  ✓ Loaded {len(docs)} chunks")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")

            print()

        if not all_documents:
            print("❌ No documents loaded successfully!")
            return

        print(f"✓ Total documents loaded: {len(all_documents)}\n")

        # Split documents
        print("✂️  Splitting documents into chunks...")
        # Optimized for M4 Pro with DeepSeek's 131k context window
        # 3000 chars ≈ 750 tokens per chunk
        # With k=5: 3,750 tokens total context (~3% of 131k capacity)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,      # 3x larger for better context
            chunk_overlap=500     # More overlap for continuity
        )
        splits = text_splitter.split_documents(all_documents)
        print(f"✓ Created {len(splits)} chunks\n")

        # Create vector store with batching for large document sets
        print("💾 Creating ChromaDB vector store...")
        print(f"   Processing {len(splits)} chunks in batches...")

        # Process in batches to avoid overwhelming Ollama
        # Using small batches to prevent crashes
        batch_size = 10

        try:
            if len(splits) > batch_size:
                # Create initial vectorstore with first batch
                print(f"   Batch 1/{(len(splits)-1)//batch_size + 1}: Processing chunks 1-{batch_size}...")
                self.vectorstore = Chroma.from_documents(
                    documents=splits[:batch_size],
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )

                # Add remaining batches
                for i in range(batch_size, len(splits), batch_size):
                    batch_num = i//batch_size + 1
                    end_idx = min(i + batch_size, len(splits))
                    print(f"   Batch {batch_num}/{(len(splits)-1)//batch_size + 1}: Processing chunks {i+1}-{end_idx}...")

                    batch = splits[i:end_idx]

                    # Retry logic for Ollama crashes
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.vectorstore.add_documents(batch)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"      ⚠️  Retry {attempt+1}/{max_retries} after error...")
                                import time
                                time.sleep(2)
                            else:
                                raise

                    # Delay to let Ollama breathe
                    import time
                    time.sleep(1)
            else:
                # Small dataset, process all at once
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )

            print("✓ Vector store created!\n")

        except Exception as e:
            print(f"\n❌ Error creating vector store: {str(e)}")
            print("\n💡 Troubleshooting:")
            print("   1. Check if Ollama is running: curl http://localhost:11434")
            print("   2. Restart Ollama: ollama serve &")
            print("   3. Try with fewer documents")
            raise

        return len(splits)

    def load_existing_vectorstore(self, folder_name=None, db_path=None):
        """Load existing vector store"""
        # Set the path based on what was provided
        if db_path:
            self.vector_db_path = str(db_path)
            self.current_folder_name = folder_name
        elif folder_name:
            self.vector_db_path = str(self.embeddings_base_dir / folder_name)
            self.current_folder_name = folder_name

        if self.vector_db_path and os.path.exists(self.vector_db_path):
            print(f"📂 Loading existing vector store: '{self.current_folder_name}'...")
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            print("✓ Vector store loaded\n")
            return True
        return False

    def query(self, question, k=5):
        """Query the RAG system - optimized for M4 Pro hardware"""
        if not self.vectorstore:
            print("❌ No vector store available. Process documents first!")
            return None

        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt with context
        prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Get answer from LLM
        answer = self.llm.invoke(prompt)

        return {
            "result": answer,
            "source_documents": docs
        }

def main():
    # Initialize the RAG system
    rag = GoogleDriveRAG()

    # Main menu: Embed new or use existing
    print("=" * 60)
    print("🚀 GOOGLE DRIVE RAG - MAIN MENU")
    print("=" * 60)

    # Check if there are existing embeddings
    existing_embeddings = rag.list_existing_embeddings()

    if existing_embeddings:
        print(f"\n📚 Found {len(existing_embeddings)} existing embedding(s)")
        print("\nOptions:")
        print("  1. 📥 Embed a new Google Drive folder")
        print("  2. 📂 Use existing embedding")
        print("=" * 60)

        while True:
            choice = input("\nYour choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("❌ Please enter 1 or 2")

        embed_new = (choice == '1')
    else:
        print("\n⚠️  No existing embeddings found.")
        print("   Let's create your first embedding!\n")
        embed_new = True

    # Either embed new folder or load existing
    if embed_new:
        # Let user select a folder from Google Drive
        print("\n" + "=" * 60)
        print("📁 SELECT FOLDER FROM GOOGLE DRIVE")
        print("=" * 60 + "\n")

        folder_id = rag.select_folder_interactively()

        # Get the folder name for the embedding
        if folder_id:
            # Look up the folder name
            folders = rag.list_folders()
            folder_name = None
            for f in folders:
                if f['id'] == folder_id:
                    folder_name = f['name']
                    break
            if not folder_name:
                folder_name = "unknown_folder"
        else:
            folder_name = "entire_drive"

        print(f"\n📁 Embedding will be saved as: '{folder_name}'")

        # Check if this embedding already exists
        if (rag.embeddings_base_dir / folder_name).exists():
            print(f"\n⚠️  Warning: An embedding named '{folder_name}' already exists!")
            overwrite = input("Do you want to overwrite it? (yes/no): ").strip().lower()
            if overwrite not in ['yes', 'y']:
                print("❌ Cancelled. Please select a different folder or use the existing embedding.")
                return

        chunks = rag.process_documents(
            folder_id=folder_id,
            folder_name=folder_name,
            file_types=['pdf', 'docx', 'txt', 'md', 'tex']
        )

        if not chunks:
            print("❌ No documents processed. Exiting.")
            return

    else:
        # Use existing embedding
        result = rag.select_existing_embedding()
        if not result:
            print("❌ No embedding selected. Exiting.")
            return

        folder_name, db_path = result

        if not rag.load_existing_vectorstore(folder_name=folder_name, db_path=db_path):
            print("❌ Failed to load embedding. Exiting.")
            return

    # Interactive query loop
    print("=" * 60)
    print(f"💬 DeepSeek RAG - Ready for questions!")
    print(f"📁 Using embedding: '{rag.current_folder_name}'")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'help' for more info")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("❓ Your question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if question.lower() == 'help':
                print("\n📖 Help:")
                print("  - Ask questions about your documents")
                print("  - The system will retrieve relevant context and answer")
                print("  - Adjust the 'k' parameter in the code to retrieve more/fewer docs\n")
                continue

            print()
            print("🔍 Searching documents...")
            result = rag.query(question, k=5)  # Retrieve 5 chunks for better context

            if result:
                print(f"\n🤖 DeepSeek Answer:\n{result['result']}\n")

                print(f"📚 Sources ({len(result['source_documents'])} documents):")
                for i, doc in enumerate(result['source_documents'], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"  {i}. {Path(source).name}")
                print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

if __name__ == "__main__":
    main()
