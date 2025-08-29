from langchain.text_splitter import CharacterTextSplitter

text='Blockchain is a decentralized, distributed digital ledger that records transactions in blocks linked together chronologically using cryptography. Each block contains a cryptographic hash of the previous block, creating a secure, immutable, and transparent chain of records that are resistant to tampering. By using consensus mechanisms, the network validates new data, providing a single, trustworthy source of truth without needing a central authority. Blockchain technology serves as the foundation for various applications, including cryptocurrencies, smart contracts, and supply chain management. '

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result= splitter.split_text(text)

print(result