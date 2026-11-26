
from task1 import faqs, search_by_keyword

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# FAQ Matcher Class
class FAQMatcher:

    def __init__(self, faqs):
        """Create TF-IDF index for FAQs."""
        self.faqs = faqs

        corpus = [
            faq["question"] + " " + " ".join(faq["keywords"])
            for faq in faqs
        ]

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(corpus)

    def match(self, query, top_k=3):
        """Find FAQs using cosine similarity."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        results = [
            (self.faqs[i], round(float(scores[i]), 4))
            for i in range(len(self.faqs))
            if scores[i] > 0
        ]

        return sorted(
            results, key=lambda x: x[1], reverse=True
        )[:top_k]

    def best_match(self, query, threshold=0.15):
        """Return the best FAQ using hybrid search."""
        results = hybrid_search(self.faqs, query, 1)

        if results and results[0][1] >= threshold:
            return results[0]

        return None

    def explain_match(self, query):
        """Display top 3 matching FAQs."""
        results = self.match(query, 3)

        return "\n".join(
            f"{i+1}. [{score}] {faq['question']}"
            for i, (faq, score) in enumerate(results)
        ) or "No matching FAQs found."


# Hybrid Search Function
def hybrid_search(faqs, query, top_k=3):
    """Combine keyword and TF-IDF search results."""

    matcher = FAQMatcher(faqs)
    combined = {}

    # Keyword search
    for faq in search_by_keyword(faqs, query):
        combined[faq["id"]] = (faq, 0.5)

    # TF-IDF search
    for faq, score in matcher.match(query, len(faqs)):
        old_score = combined.get(faq["id"], (faq, 0))[1]

        combined[faq["id"]] = (
            faq, max(old_score, score)
        )

    return sorted(
        combined.values(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


# Demonstration
if __name__ == "__main__":

    matcher = FAQMatcher(faqs)

    queries = [
        "I forgot my login credentials",
        "Can I get my money back?",
        "package delivery time"
    ]

    for query in queries:

        print("\nQuery:", query)

        print("\n[Keyword Search]")
        keyword_results = search_by_keyword(faqs, query)

        if keyword_results:
            for faq in keyword_results[:3]:
                print("-", faq["question"])
        else:
            print("No results")

        print("\n[TF-IDF Matching]")
        print(matcher.explain_match(query))

        print("\n[Hybrid Search]")
        hybrid_results = hybrid_search(faqs, query)

        if hybrid_results:
            for i, (faq, score) in enumerate(hybrid_results):
                print(f"{i+1}. [{score}] {faq['question']}")
        else:
            print("No matching FAQs found.")

        best = matcher.best_match(query)

        if best:
            print(
                f"\nBest match: {best[0]['question']} "
                f"(confidence: {best[1]})"
            )
        else:
            print("\nNo confident match found.")
