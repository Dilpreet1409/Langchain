from langchain_community.document_loaders import PyPDFLoader

loader= PyPDFLoader('curriculum.pdf')

docs=loader.load()

print(docs)

''' PDFS with Tables/Columns - PDFPlumberLoader 
    Scanned/Image PDFs - UnstructuredPDFLoader or AmazonTextractPDFLoader
    Need Layout and Image data - PyMuPDFLoader
    Want Best structure extraction - UnstructuredPDFLoader '''