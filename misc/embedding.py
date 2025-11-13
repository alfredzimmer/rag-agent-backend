from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
  file_path = './IEEE/IEEE Std 493 -2007.pdf'
  loader = PyPDFLoader(file_path)

  documents = loader.load()
  # print(documents[0].page_content[:250])

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)

  chunks = text_splitter.split_documents(documents)
  print(f"Generated {len(chunks)} chunks from your document(s).")

  # print('chunk 50: ', chunks[50].page_content)



