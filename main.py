import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InvoiceMatchingSystem:
    def __init__(self):
        self.invoices = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def add_invoice(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        self.invoices.append({
            'path': pdf_path,
            'text': text
        })
        self._update_tfidf_matrix()

    def _update_tfidf_matrix(self):
        texts = [invoice['text'] for invoice in self.invoices]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def find_similar_invoice(self, new_invoice_path):
        new_text = self.extract_text_from_pdf(new_invoice_path)
        new_tfidf = self.vectorizer.transform([new_text])
        
        similarities = cosine_similarity(new_tfidf, self.tfidf_matrix)
        most_similar_index = similarities.argmax()
        similarity_score = similarities[0][most_similar_index]
        
        return self.invoices[most_similar_index], similarity_score

def main():
    system = InvoiceMatchingSystem()
    
   
    database_dir = "document similarity/train"
    for filename in os.listdir(database_dir):
        if filename.endswith('.pdf'):
            system.add_invoice(os.path.join(database_dir, filename))
    
   
    test_invoice = "document similarity/test/invoice_77098.pdf"
    similar_invoice, score = system.find_similar_invoice(test_invoice)
    
    print(f"Most similar invoice: {similar_invoice['path']}")
    print(f"Similarity score: {score}")

if __name__ == "__main__":
    main()